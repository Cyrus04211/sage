"""Arm E: ASS + warm-start + SMAC/LLM 交替候选生成。

与 Arm B 的唯一区别: 20 次搜索评估中, 偶数轮由 SMAC 提议候选 (RF+EI),
奇数轮由 LLM 基于优化历史提议候选 (GRIMIP 的 candidate generation),
两者共用同一份 runhistory (SMAC tell)。

用法:
  PYTHONPATH=. .venv/bin/python -m sage.arms.run_arm_e \
      --instance data/instances/118/model2797.lp --out results/arm_e/model2797_s0.json
"""
import argparse
import json
import time
from pathlib import Path

from ConfigSpace import Configuration
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from sage.agent_decisions import llm_propose, select_space, warmstart_configs
from sage.evaluator import InstanceEvaluator
from sage.observe import dynamic_features, feature_text, static_features
from sage.params import (build_subspace, filter_settable, load_params,
                         tunable_params)

HERE = Path(__file__).resolve().parents[2]
PARAMS_JSON = HERE / "rag" / "params.json"


def _noop_target(config, seed):
    """占位 target (SMAC 签名校验用); 实际评估在交替循环中手动进行。"""
    return 0.0


def run(instance: str, trials: int, n_ws: int, k: int, time_limit: float,
        seed: int, out: str, model: str | None = None):
    params = load_params(PARAMS_JSON)
    keep, _ = tunable_params(params)
    keep, _ = filter_settable(keep)

    import gurobipy as gp
    m = gp.read(instance)
    s_feats = static_features(m)
    del m
    d_feats = dynamic_features(instance, probe_seconds=60)
    f_text = feature_text({"static": s_feats, "dynamic": d_feats})
    print(f"[features]\n{f_text}", flush=True)

    selection, ass_meta = select_space(f_text, keep, k=k, model=model,
                                       return_meta=True)
    print(f"[ass] selected: {[s['name'] for s in selection]}", flush=True)
    ws_configs = warmstart_configs(f_text, selection, keep, n=n_ws, model=model)

    evaluator = InstanceEvaluator(instance, time_limit=time_limit, threads=1)

    def evaluate_cfg(cfg):
        try:
            res = evaluator.evaluate(cfg)
            return res.pdi, res
        except Exception as e:
            return 1e18, None

    # warm-start 评估 (另算预算)
    history = []
    ws_results = []
    for cfg in ws_configs:
        res = evaluator.evaluate(cfg)
        ws_results.append(res)
        history.append({"config": cfg, "pdi": res.pdi, "source": "warmstart"})
        print(f"[ws-eval] pdi={res.pdi:.4g} cfg={cfg}", flush=True)

    cs = build_subspace(keep, selection, seed=seed)
    defaults = dict(cs.get_default_configuration())
    scenario = Scenario(cs, n_trials=trials + len(ws_configs),
                        deterministic=True, seed=seed)
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario, n_configs=0)
    smac = HyperparameterOptimizationFacade(
        scenario, _noop_target, initial_design=initial_design,
        overwrite=True)
    for cfg, res in zip(ws_configs, ws_results):
        smac.tell(TrialInfo(config=Configuration(cs, defaults | cfg), seed=seed),
                  TrialValue(cost=res.pdi, time=res.runtime))

    # SMAC / LLM 交替提议
    t0 = time.time()
    for i in range(trials):
        if i % 2 == 0:
            info = smac.ask()
            source = "smac"
        else:
            existing = [h["config"] for h in history]
            cfg = llm_propose(f_text, selection, keep, history, existing,
                              model=model)
            if cfg is None:                       # 重复/非法 -> 回退 SMAC
                info = smac.ask()
                source = "smac-fallback"
            else:
                try:
                    info = TrialInfo(config=Configuration(cs, defaults | cfg),
                                     seed=seed)
                    source = "llm"
                except Exception:                 # 范围外取值 -> 回退 SMAC
                    info = smac.ask()
                    source = "smac-fallback"
        cfg_dict = dict(info.config)
        t1 = time.time()
        try:
            res = evaluator.evaluate(cfg_dict)
            cost, pdi, rt = res.pdi, res.pdi, res.runtime
            rec = {"config": cfg_dict, "pdi": pdi, "runtime": rt,
                   "status": res.status, "source": source}
        except Exception as e:
            cost = 1e18
            rec = {"config": cfg_dict, "error": repr(e), "source": source}
        rec["walltime"] = time.time() - t1
        history.append(rec)
        smac.tell(info, TrialValue(cost=cost, time=rec["walltime"]))
        print(f"[{i+1}/{trials}] {source}: pdi={rec.get('pdi', float('nan')):.4g}",
              flush=True)
    walltime = time.time() - t0

    valid = [h for h in history if h.get("pdi") is not None]
    best = min(valid, key=lambda h: h["pdi"])
    n_llm = sum(1 for h in history if h.get("source") == "llm")
    result = {
        "arm": "E", "instance": instance, "seed": seed,
        "trials_smac_loop": trials, "n_warmstart": len(ws_configs), "k": k,
        "llm_model": model, "n_llm_proposals": n_llm,
        "feature_text": f_text,
        "selection": selection, "ass_meta": ass_meta,
        "warmstart_configs": ws_configs,
        "warmstart_results": [r.__dict__ for r in ws_results],
        "incumbent": best["config"],
        "history": history,
        "walltime_h": walltime / 3600,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1, default=str))
    print(f"[done] best_pdi={best['pdi']:.4g} llm_proposals={n_llm} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--n-ws", type=int, default=5)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.instance, args.trials, args.n_ws, args.k, args.time_limit,
        args.seed, args.out, model=args.model)


if __name__ == "__main__":
    main()
