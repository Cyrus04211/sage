"""2746 实例难度探测: 120s 默认参数运行, 串行, 每次只加载一个实例。
用法: PYTHONPATH=. .venv/bin/python scripts/profile_2746.py
"""
import gc
import json
import time
from pathlib import Path

from sage.observe import dynamic_features

INST_DIR = Path("data/instances/2746")
OUT = Path("results/profiles_2746.json")

def main():
    out = {}
    for lp in sorted(INST_DIR.glob("*.lp")):
        t0 = time.time()
        try:
            f = dynamic_features(str(lp), probe_seconds=120)
            f["file_mb"] = lp.stat().st_size // 2**20
        except Exception as e:
            f = {"error": repr(e)}
        out[lp.stem] = f
        print(f"{lp.stem}: status={f.get('status')} gap={f.get('mip_gap')} "
              f"nodes={f.get('nodes_processed')} sols={f.get('sol_count')} "
              f"root_lp_time={f.get('root_lp_time')} "
              f"({time.time()-t0:.0f}s)", flush=True)
        gc.collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print("->", OUT)

if __name__ == "__main__":
    main()
