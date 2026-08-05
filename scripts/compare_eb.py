"""E vs B 快速对比。用法: PYTHONPATH=. .venv/bin/python scripts/compare_eb.py"""
import json
from pathlib import Path

def best(r):
    pdis = [h.get("pdi") for h in r.get("history", []) if h.get("pdi") is not None]
    return min(pdis) if pdis else float("inf")

def main():
    rows = []
    for lp in sorted(Path("data/instances/118").glob("*.lp")):
        inst = lp.stem
        d = json.load(open(f"results/default/{inst}.json"))["pdi"]
        b = best(json.load(open(f"results/arm_b/{inst}_s0.json")))
        er = json.load(open(f"results/arm_e/{inst}_s0.json"))
        e = best(er)
        rows.append((inst, (d - b) / d * 100, (d - e) / d * 100,
                     er["n_llm_proposals"]))
    print("%-12s %8s %8s %6s" % ("instance", "B imp", "E imp", "nLLM"))
    for inst, b, e, n in rows:
        mark = "E" if e > b else "B"
        print("%-12s %+7.1f%% %+7.1f%% %6d  <- %s" % (inst, b, e, n, mark))
    bw = sum(1 for _, b, e, _ in rows if b > e)
    print()
    print("B mean %+.1f%% | E mean %+.1f%% | B wins %d/16" % (
        sum(r[1] for r in rows) / 16, sum(r[2] for r in rows) / 16, bw))

if __name__ == "__main__":
    main()
