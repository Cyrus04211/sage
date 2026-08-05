"""三臂对比汇总。用法: PYTHONPATH=. .venv/bin/python scripts/compare.py [实例名...]"""
import json
import sys
from pathlib import Path

def best_pdi(r):
    pdis = [h.get("pdi") for h in r.get("history", []) if h.get("pdi") is not None]
    pdis += [x["pdi"] for x in r.get("warmstart_results", [])]
    return min(pdis) if pdis else float("inf")

def main(insts):
    for inst in insts:
        d = json.load(open(f"results/default/{inst}.json"))
        print(f"=== {inst} | default PDI {d['pdi']:.4g} (runtime {d['runtime']:.0f}s) ===")
        for arm in ["a", "b", "c"]:
            f = Path(f"results/arm_{arm}/{inst}.json")
            if not f.exists():
                print(f"  Arm {arm.upper()}: running...")
                continue
            r = json.load(open(f))
            bp = best_pdi(r)
            imp = (d["pdi"] - bp) / d["pdi"] * 100
            extra = ""
            ws = [x["pdi"] for x in r.get("warmstart_results", [])]
            if ws:
                sel = [s["name"] for s in r.get("selection", [])]
                extra = f" | ws best {min(ws):.4g} | space {sel}"
            print(f"  Arm {arm.upper()}: best {bp:.4g} ({imp:+.1f}%){extra}")

if __name__ == "__main__":
    main(sys.argv[1:] or ["model2797", "model1337"])
