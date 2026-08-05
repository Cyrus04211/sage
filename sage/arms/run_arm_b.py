"""Arm B: GRIMIP 三件套 (ASS + warm-start) + SMAC 子空间搜索。

流程: 特征提取 -> LLM 选 k 个参数 (ASS) -> LLM 生成 n 个初始配置 (WS, 另算预算)
-> 评估初始配置 -> tell 进 SMAC -> SMAC 在子空间再调 20 次。

用法:
  PYTHONPATH=. .venv/bin/python -m sage.arms.run_arm_b \
      --instance data/instances/118/model237.lp --out results/arm_b/model237.json

Arm C 通过 --rag 开启: ASS/WS 提示词中注入检索到的专家经验上下文。
--model 可指定 LLM (默认读 .env 的 LLM_MODEL)。
"""
import argparse
import json
import time
from pathlib import Path

from ConfigSpace import Configuration
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from sage.agent_decisions import select_space, warmstart_configs
from sage.evaluator import InstanceEvaluator
from sage.observe import dynamic_features, feature_text, static_features
from sage.params import (build_subspace, filter_settable, load_params,
                         tunable_params)

HERE = Path(__file__).resolve().parents[2]
PARAMS_JSON = HERE / "rag" / "params.json"


def run(instance: str, trials: int, n_ws: int, k: int, time_limit: float,
        seed: int, out: str, use_rag: bool = False, model: str | None = None):
    params = load_params(PARAMS_JSON)
    keep, _ = tunable_params(params)
    keep, _ = filter_settable(keep)

    # 1. 特征 (动态特征优先读画像缓存)
    import gurobipy as gp
    m = gp.read(instance)
    s_feats = static_features(m)
    del m
    d_feats = dynamic_features(instance, probe_seconds=60)
    f_text = feature_text({"static": s_feats, "dynamic": d_feats})
    print(f"[features]\n{f_text}", flush=True)

    rag_context = ""
    if use_rag:
        from sage.rag_retriever import build_context
        rag_context = build_context(f_text, params=keep)
        print(f"[rag] context {len(rag_context)} chars", flush=True)

    # 2. ASS: LLM 选参数
    selection, ass_meta = select_space(f_text, keep, k=k,
                                       rag_context=rag_context, model=model,
                                       return_meta=True)
    print(f"[ass] selected: {[s['name'] for s in selection]} "
          f"(fallback: {ass_meta['fallback_used']})", flush=True)

    # 3. WS: LLM 生成初始配置
    ws_configs = warmstart_configs(f_text, selection, keep, n=n_ws,
                                   rag_context=rag_context, model=model)
    print(f"[ws] {len(ws_configs)} configs", flush=True)

    # 4. 评估初始配置 (另算预算, 不计入 SMAC 的 20 次)
    evaluator = InstanceEvaluator(instance, time_limit=time_limit, threads=1)
    ws_results = []
    for cfg in ws_configs:
        res = evaluator.evaluate(cfg)
        ws_results.append(res)
        print(f"[ws-eval] pdi={res.pdi:.4g} runtime={res.runtime:.0f}s "
              f"cfg={cfg}", flush=True)

    # 5. SMAC 子空间搜索 (tell 注入 warm-start 结果)
    cs = build_subspace(keep, selection, seed=seed)
    history = []

    def target(config, seed):
        t0 = time.time()
        try:
            res = evaluator.evaluate(dict(config))
            cost = res.pdi
            rec = {"config": dict(config), "pdi": res.pdi,
                   "runtime": res.runtime, "status": res.status,
                   "obj": res.obj, "bound": res.bound}
        except Exception as e:
            cost = 1e18
            rec = {"config": dict(config), "error": repr(e)}
        rec["walltime"] = time.time() - t0
        history.append(rec)
        return cost

    scenario = Scenario(cs, n_trials=trials + len(ws_configs),
                        deterministic=True, seed=seed)
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario, n_configs=0)
    smac = HyperparameterOptimizationFacade(
        scenario, target, initial_design=initial_design, overwrite=True)
    defaults = dict(cs.get_default_configuration())
    for cfg, res in zip(ws_configs, ws_results):
        full_cfg = defaults | cfg      # ConfigSpace 要求所有活跃参数都有值
        smac.tell(TrialInfo(config=Configuration(cs, full_cfg), seed=seed),
                  TrialValue(cost=res.pdi, time=res.runtime))
    t0 = time.time()
    incumbent = smac.optimize()
    walltime = time.time() - t0

    result = {
        "arm": "C" if use_rag else "B",
        "instance": instance, "seed": seed,
        "trials_smac": trials, "n_warmstart": len(ws_configs), "k": k,
        "llm_model": model,
        "feature_text": f_text,
        "selection": selection, "ass_meta": ass_meta, "warmstart_configs": ws_configs,
        "warmstart_results": [r.__dict__ for r in ws_results],
        "incumbent": dict(incumbent),
        "history": history,
        "walltime_h": walltime / 3600,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1, default=str))
    all_pdi = [r.pdi for r in ws_results] + [h.get("pdi", float("inf"))
                                             for h in history]
    print(f"[done] best_pdi={min(all_pdi):.4g} "
          f"(ws best={min(r.pdi for r in ws_results):.4g}) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--n-ws", type=int, default=5)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rag", action="store_true", help="Arm C: 启用 RAG 上下文")
    ap.add_argument("--model", default=None, help="LLM 模型名 (默认读 .env)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.instance, args.trials, args.n_ws, args.k, args.time_limit,
        args.seed, args.out, use_rag=args.rag, model=args.model)


if __name__ == "__main__":
    main()
