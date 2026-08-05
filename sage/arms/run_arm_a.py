"""Arm A: SMAC 在全参数空间上 per-instance 调参 (20 次评估)。

用法:
  python -m sage.arms.run_arm_a --instance data/instances/118/model237.lp \
      --trials 20 --time-limit 300 --seed 0 --out results/arm_a/model237.json
"""
import argparse
import json
import time
from pathlib import Path

from smac import HyperparameterOptimizationFacade, Scenario

from sage.evaluator import InstanceEvaluator
from sage.params import build_full_space, filter_settable, load_params, tunable_params

HERE = Path(__file__).resolve().parents[2]   # 项目根 (含 rag/, sage/ 包)
PARAMS_JSON = HERE / "rag" / "params.json"


def run(instance: str, trials: int, time_limit: float, seed: int, out: str):
    params = load_params(PARAMS_JSON)
    keep, drop = tunable_params(params)
    keep, drop2 = filter_settable(keep)   # 剔除环境级/只读参数 (如 ServerTimeout)
    cs = build_full_space(keep, seed=seed)
    print(f"[space] tunable={len(keep)} excluded={len(drop) + len(drop2)}")

    evaluator = InstanceEvaluator(instance, time_limit=time_limit, threads=1)
    crash_cost = None  # 首次评估后按 default PDI 的倍数定

    history = []

    def target(config, seed):
        nonlocal crash_cost
        t0 = time.time()
        try:
            res = evaluator.evaluate(dict(config))
            cost = res.pdi
            rec = {"config": dict(config), "pdi": res.pdi, "runtime": res.runtime,
                   "status": res.status, "obj": res.obj, "bound": res.bound,
                   "nodes": res.nodes, "timed_out": res.timed_out}
        except Exception as e:  # Gurobi 数值错误等: 按最差计入
            cost = crash_cost if crash_cost is not None else 1e18
            rec = {"config": dict(config), "error": repr(e)}
        rec["walltime"] = time.time() - t0
        history.append(rec)
        if crash_cost is None and "pdi" in rec and rec["pdi"] > 0:
            crash_cost = rec["pdi"] * 10.0   # 粗略基准: 首个有效 PDI 的 10 倍
        return cost

    # 不传 trial_walltime_limit: SMAC 2.4 在该参数下会异常耗尽 trials 预算;
    # 单次评估时限由评估器通过 Gurobi TimeLimit 自行保证
    scenario = Scenario(cs, n_trials=trials, deterministic=True, seed=seed)
    smac = HyperparameterOptimizationFacade(scenario, target, overwrite=True)
    t0 = time.time()
    incumbent = smac.optimize()
    walltime = time.time() - t0

    result = {
        "arm": "A", "instance": instance, "seed": seed,
        "trials": trials, "time_limit": time_limit,
        "n_params": len(keep),
        "incumbent": dict(incumbent),
        "history": history,
        "walltime_h": walltime / 3600,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1, default=str))
    best = min((h.get("pdi", float("inf")) for h in history))
    print(f"[done] incumbent={dict(incumbent)}")
    print(f"[done] best_pdi={best:.4g} walltime={walltime/3600:.2f}h -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.instance, args.trials, args.time_limit, args.seed, args.out)


if __name__ == "__main__":
    main()
