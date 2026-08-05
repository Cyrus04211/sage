"""Arm D: SMAC 在论文固定六维空间 Θ 上 per-instance 调参 (20 次)。
即 GRIMIP 论文的 SMAC-I 基线设置。

用法:
  PYTHONPATH=. .venv/bin/python -m sage.arms.run_arm_d \
      --instance data/instances/118/model237.lp --out results/arm_d/model237.json
"""
import argparse
import json
import time
from pathlib import Path

from smac import HyperparameterOptimizationFacade, Scenario

from sage.evaluator import InstanceEvaluator
from sage.params import build_theta_space


def run(instance: str, trials: int, time_limit: float, seed: int, out: str,
        resume: bool = False):
    cs = build_theta_space(seed=seed)
    evaluator = InstanceEvaluator(instance, time_limit=time_limit, threads=1)
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

    old_hist, old_walltime = [], 0.0
    if resume and Path(out).exists():
        old = json.loads(Path(out).read_text())
        old_hist = old.get("history", [])
        old_walltime = old.get("walltime_h", 0) * 3600

    scenario = Scenario(cs, n_trials=trials, deterministic=True, seed=seed)
    kwargs = {}
    if resume:
        kwargs["initial_design"] = \
            HyperparameterOptimizationFacade.get_initial_design(
                scenario, n_configs=0)
    smac = HyperparameterOptimizationFacade(scenario, target,
                                            overwrite=True, **kwargs)
    if resume and old_hist:
        # 重放旧评估作为 runhistory, 等效续跑 (RF 代理按全量数据重建)
        from ConfigSpace import Configuration
        from smac.runhistory.dataclasses import TrialInfo, TrialValue
        first_pdi = next((h["pdi"] for h in old_hist if h.get("pdi")), None)
        crash = (first_pdi or 1e17) * 10
        defaults = dict(cs.get_default_configuration())
        for h in old_hist:
            cfg = Configuration(cs, defaults | h["config"])
            smac.tell(TrialInfo(config=cfg, seed=seed),
                      TrialValue(cost=h.get("pdi", crash),
                                 time=h.get("walltime", 0.0)))
    t0 = time.time()
    incumbent = smac.optimize()
    walltime = time.time() - t0 + old_walltime
    history = old_hist + history            # 合并旧轨迹

    result = {"arm": "D", "instance": instance, "seed": seed,
              "trials": trials, "time_limit": time_limit,
              "space": "theta6",
              "incumbent": dict(incumbent), "history": history,
              "walltime_h": walltime / 3600}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1, default=str))
    best = min((h.get("pdi", float("inf")) for h in history))
    print(f"[done] best_pdi={best:.4g} walltime={walltime/3600:.2f}h -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="从已有 runhistory 续跑 (trials 为总预算)")
    args = ap.parse_args()
    run(args.instance, args.trials, args.time_limit, args.seed, args.out,
        resume=args.resume)


if __name__ == "__main__":
    main()
