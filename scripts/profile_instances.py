"""对全部 118 实例做快速画像: 60s 默认参数探测运行, 输出难度与动态特征。

串行执行, 每次只加载一个实例 (峰值内存 < 数 GB), 不会 OOM。
用法: python scripts/profile_instances.py
"""
import gc
import json
import time
from pathlib import Path

from sage.observe import dynamic_features

INST_DIR = Path("data/instances/118")
OUT = Path("results/profiles.json")

def main():
    out = {}
    lps = sorted(INST_DIR.glob("*.lp"), key=lambda p: p.stat().st_size)
    print(f"{len(lps)} instances")
    for lp in lps:
        t0 = time.time()
        try:
            feats = dynamic_features(str(lp), probe_seconds=60)
            feats["file_mb"] = lp.stat().st_size // 2**20
        except Exception as e:
            feats = {"error": repr(e)}
        out[lp.stem] = feats
        print(f"{lp.stem}: status={feats.get('status')} gap={feats.get('mip_gap')} "
              f"nodes={feats.get('nodes_processed')} sols={feats.get('sol_count')} "
              f"({time.time()-t0:.0f}s)", flush=True)
        gc.collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print("->", OUT)

if __name__ == "__main__":
    main()
