"""最终配置的双指标评估: 对每实例×每臂的 incumbent 配置复跑 N 次,
报告 PDI 与求解时间（runtime）的均值。

用法: PYTHONPATH=. .venv/bin/python scripts/final_eval.py [实例名...] [--reps 3]
"""
import json
import sys
from pathlib import Path

from sage.evaluator import InstanceEvaluator

ARMS = ["a", "b", "c", "d", "e"]


def find_result(arm, inst):
    for p in [Path(f"results/arm_{arm}/{inst}_s0.json"),
              Path(f"results_luna/arm_{arm}/{inst}.json"),
              Path(f"results/arm_{arm}/{inst}.json")]:
        if p.exists():
            return json.load(open(p))
    return None


def main(insts, reps=3):
    outdir = Path("results/final_eval")
    outdir.mkdir(parents=True, exist_ok=True)
    for inst in insts:
        entry = {}
        d = json.load(open(f"results/default/{inst}.json"))
        entry["default"] = {"pdi": d["pdi"], "runtime": d["runtime"]}
        ev = InstanceEvaluator(f"data/instances/118/{inst}.lp",
                               time_limit=300, threads=1)
        for arm in ARMS:
            r = find_result(arm, inst)
            if r is None:
                continue
            cfg = r["incumbent"]
            pdis, rts = [], []
            for _ in range(reps):
                res = ev.evaluate(cfg)
                pdis.append(res.pdi)
                rts.append(res.runtime)
            entry[f"arm_{arm}"] = {
                "pdi": sum(pdis) / reps, "runtime": sum(rts) / reps,
                "pdi_all": pdis, "runtime_all": rts, "config": cfg}
            print(f"{inst} arm_{arm}: pdi={entry[f'arm_{arm}']['pdi']:.4g} "
                  f"runtime={entry[f'arm_{arm}']['runtime']:.1f}s", flush=True)
        (outdir / f"{inst}.json").write_text(
            json.dumps(entry, indent=1, default=str))
        print(f"-> {outdir}/{inst}.json", flush=True)


if __name__ == "__main__":
    insts = [a for a in sys.argv[1:] if not a.startswith("--")]
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3
    main(insts or ["model2797", "model1337"], reps)
