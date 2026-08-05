"""绘制三臂 best-so-far PDI 收敛曲线（每实例一张图）。

x 轴 = 评估序号（Arm B/C 的前 5 个点为 warm-start 配置）；y 轴 = 至此最佳 PDI。
默认配置 PDI 画水平虚线。输出 results/figs/convergence_<instance>.png

用法: PYTHONPATH=. .venv/bin/python scripts/plot_convergence.py [实例名...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("a", "Arm A (SMAC, 131 params)", "#d62728"),
        ("d", "Arm D (SMAC, fixed 6 params)", "#9467bd"),
        ("b", "Arm B (ASS+WS+SMAC)", "#1f77b4"),
        ("c", "Arm C (B + RAG)", "#2ca02c"),
        ("e", "Arm E (B + LLM candidates)", "#ff7f0e")]


def best_so_far(r):
    seq = [x["pdi"] for x in r.get("warmstart_results", [])]
    seq += [h.get("pdi") for h in r.get("history", [])]
    curve, best = [], float("inf")
    for p in seq:
        if p is not None and p < best:
            best = p
        curve.append(best)
    return curve


def main(insts):
    outdir = Path("results/figs")
    outdir.mkdir(parents=True, exist_ok=True)
    for inst in insts:
        d = json.load(open(f"results/default/{inst}.json"))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arm, label, color in ARMS:
            f = None
            for cand in [Path(f"results/arm_{arm}/{inst}_s0.json"),
                         Path(f"results/arm_{arm}/{inst}.json"),
                         Path(f"results_luna/arm_{arm}/{inst}.json")]:
                if cand.exists():
                    f = cand
                    break
            if f is None:
                continue
            curve = best_so_far(json.load(open(f)))
            ax.plot(range(1, len(curve) + 1), curve, label=label,
                    color=color, lw=1.8)
        ax.axhline(d["pdi"], ls="--", c="gray", label="Default")
        ax.set_yscale("log")   # PDI 跨数量级, 对数轴才能看清各臂差异
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Best PDI so far (log)")
        ax.set_title(inst)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"convergence_{inst}.png", dpi=150)
        plt.close(fig)
        print("saved", outdir / f"convergence_{inst}.png")


if __name__ == "__main__":
    main(sys.argv[1:] or ["model2797", "model1337"])
