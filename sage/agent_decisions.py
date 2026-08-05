"""Arm B/C 的 LLM 决策: ASS (选参数子空间) 与 warm-start (初始配置生成)。

Arm B: 提示词只含参数名+一句话描述 (与 GRIMIP 论文的 ASS 信息量一致)。
Arm C: 额外注入 RAG 检索到的手册条目/论文/专家经验 (由 rag 模块提供 context)。
"""
import json

from sage.llm import chat_json

SYSTEM = ("You are a world-class Gurobi tuning expert with deep knowledge of "
          "MIP solver internals and unit commitment problems.")

ASS_PROMPT = """You are tuning Gurobi parameters for a specific MIP instance (unit commitment problem).

## Instance Feature Analysis
{feature_text}

## Tunable Parameter List (name: brief description)
{param_list}
{rag_context}
## Task
Select the {k} parameters predicted to have the most significant impact on
solving performance for THIS instance. Consider the instance's structure
(variable types, constraint structure, root relaxation behavior, cut activity).

Return ONLY valid JSON (each parameter may appear at most once, keys must be unique):
{{"selections": {{"<param_name>": "<why, citing evidence>", ...}}}}"""

# 确定性补位优先级: GRIMIP 论文/经典 AC 文献中的高影响参数
ASS_FALLBACK = ["MIPFocus", "Heuristics", "Cuts", "Presolve", "Method",
                "VarBranch", "MIRCuts", "FlowCoverCuts", "GomoryPasses", "RINS"]

WS_PROMPT = """You are tuning Gurobi parameters for a specific MIP instance (unit commitment problem).

## Instance Feature Analysis
{feature_text}

## Selected Tunable Parameters (name, type, range, default)
{param_space}
{rag_context}
## Task
Recommend {n} diverse, high-potential parameter configurations as the initial
population for Bayesian optimization. They should cover distinct strategies
(e.g., feasibility-oriented, bound-oriented, cut-aggressive, heuristic-heavy).
Every value must lie within the given range. Include only the selected
parameters; unlisted parameters keep their defaults.

Return ONLY valid JSON:
{{"configs": [{{"<param>": <value>, ...}}, ...]}}"""


def _fmt_param_list(tunable: list[dict]) -> str:
    return "\n".join(f"- {p['name']}: {p['brief']}" for p in tunable)


def _fmt_param_space(selection: list[dict], tunable: list[dict]) -> str:
    by_name = {p["name"]: p for p in tunable}
    lines = []
    for s in selection:
        p = by_name[s["name"]]
        lines.append(f"- {p['name']} ({p['type']}, default {p['default']}, "
                     f"range [{p['min']}, {p['max']}]): {p['brief']}")
    return "\n".join(lines)


def select_space(feature_text: str, tunable: list[dict], k: int = 6,
                 rag_context: str = "", model: str | None = None,
                 return_meta: bool = False):
    """ASS: LLM 从可调参数中选 k 个。

    输出契约: {"selections": {name: reason}} —— JSON 对象键天然唯一,
    从结构上排除重复参数名。不足 k 个时按 ASS_FALLBACK 确定性补位
    (不掷骰子重试), 补位情况记入 meta 供实验审计。
    """
    prompt = ASS_PROMPT.format(feature_text=feature_text,
                               param_list=_fmt_param_list(tunable),
                               k=k, rag_context=rag_context)
    valid = {p["name"] for p in tunable}
    out = chat_json(prompt, system=SYSTEM, temperature=0.7, model=model)
    sels = out.get("selections", {})
    selected = [{"name": n, "reason": str(r)}
                for n, r in sels.items() if n in valid]
    meta = {"invalid_names": [n for n in sels if n not in valid],
            "fallback_used": []}
    for name in ASS_FALLBACK:                    # 确定性补位
        if len(selected) >= k:
            break
        if name in valid and name not in {s["name"] for s in selected}:
            selected.append({"name": name,
                             "reason": "fallback: canonical high-impact parameter"})
            meta["fallback_used"].append(name)
    selected = selected[:k]
    if return_meta:
        return selected, meta
    return selected


def warmstart_configs(feature_text: str, selection: list[dict],                      tunable: list[dict], n: int = 5,
                      rag_context: str = "", model: str | None = None) -> list[dict]:
    """WS: LLM 生成 n 个初始配置，取值裁剪到手册范围。"""
    prompt = WS_PROMPT.format(feature_text=feature_text,
                              param_space=_fmt_param_space(selection, tunable),
                              n=n, rag_context=rag_context)
    out = chat_json(prompt, system=SYSTEM, temperature=0.9, model=model)
    by_name = {p["name"]: p for p in tunable}
    configs = []
    for cfg in out.get("configs", []):
        clean = _clamp_config(cfg, selection, tunable)
        if clean:
            configs.append(clean)
    if not configs:
        raise RuntimeError(f"warm-start 未产出任何合法配置: {out}")
    return configs[:n]


# ---------------- Arm E: LLM 候选生成 (论文 candidate generation) ----------------

CG_PROMPT = """You are a world-class Gurobi tuning expert optimizing solver parameters
for a specific MIP instance (unit commitment).

## Instance Feature Analysis
{feature_text}

## Tunable Parameter Space (selected subset; name, type, range, default)
{param_space}

## Optimization History (config -> PDI, lower is better; best first)
{history_text}

## Task
Propose the NEXT configuration to evaluate. Balance exploitation (refine what
works) and exploration (try meaningfully different parameter values). Use only
the selected parameters; every value must lie within its range.

Return ONLY valid JSON:
{{"config": {{"<param>": <value>, ...}}, "rationale": "<one line>"}}"""


def _fmt_history(history: list[dict], limit: int = 25) -> str:
    """history: [{"config": {...}, "pdi": ...}]，按 PDI 升序输出。"""
    valid = [h for h in history if h.get("pdi") is not None]
    valid.sort(key=lambda h: h["pdi"])
    lines = []
    for h in valid[:limit]:
        cfg = ", ".join(f"{k}={v}" for k, v in h["config"].items())
        lines.append(f"- PDI {h['pdi']:.5g}  <-  {cfg}")
    return "\n".join(lines)


def _clamp_config(cfg: dict, selection: list[dict], tunable: list[dict]) -> dict:
    from sage.params import INF_CAP, MAXINT_CAP
    by_name = {p["name"]: p for p in tunable}
    selected = {s["name"] for s in selection}
    clean = {}
    for name, val in cfg.items():
        if name not in by_name or name not in selected:
            continue
        p = by_name[name]
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        lo = float(p["min"]) if p["min"] not in ("MAXINT", None) else -1e9
        hi = float(p["max"]) if p["max"] not in ("MAXINT", None) else 1e9
        # 与 ConfigSpace 构建口径一致 (params.build_subspace 的截断)
        if p["type"] == "int":
            lo, hi = min(lo, MAXINT_CAP), min(hi, MAXINT_CAP)
        else:
            lo, hi = max(lo, -INF_CAP), min(hi, INF_CAP)
        v = min(max(v, lo), hi)
        clean[name] = int(round(v)) if p["type"] == "int" else v
    return clean


def llm_propose(feature_text: str, selection: list[dict], tunable: list[dict],
                history: list[dict], existing: list[dict],
                model: str | None = None) -> dict | None:
    """LLM 候选生成 (GRIMIP candidate generation): 基于优化历史提出下一个配置。
    与已有配置重复时重试一次; 仍重复则返回 None (调用方回退 SMAC)。"""
    for _ in range(2):
        out = chat_json(CG_PROMPT.format(
            feature_text=feature_text,
            param_space=_fmt_param_space(selection, tunable),
            history_text=_fmt_history(history)), system=SYSTEM,
            temperature=0.8, model=model)
        cfg = _clamp_config(out.get("config", {}), selection, tunable)
        if cfg and cfg not in existing:
            return cfg
    return None
