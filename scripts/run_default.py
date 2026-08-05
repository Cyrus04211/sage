"""默认配置基线: 对实例跑默认参数 300s, 记录 PDI。
用法: PYTHONPATH=. .venv/bin/python scripts/run_default.py data/instances/118/model2797.lp
"""
import json
import sys
from pathlib import Path

from sage.evaluator import InstanceEvaluator

def main():
    lp = sys.argv[1]
    name = Path(lp).stem
    ev = InstanceEvaluator(lp, time_limit=300, threads=1)
    res = ev.evaluate({})
    out = Path(f"results/default/{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res.__dict__, indent=1, default=str))
    print(f"{name}: pdi={res.pdi:.4g} runtime={res.runtime:.1f}s "
          f"status={res.status} obj={res.obj}")

if __name__ == "__main__":
    main()
