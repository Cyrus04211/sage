"""四臂汇总表 (PDI 相对默认改善率 + Wins 行), 输出 markdown + PNG。

用法: PYTHONPATH=. .venv/bin/python scripts/summary_table.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("a", "SMAC3\n(full 131 params)"),
        ("d", "SMAC3\n(fixed 6 params)"),
        ("b", "Ours\n(ASS+WS+SMAC)"),
        ("c", "Ours+RAG"),
        ("e", "Ours\n+LLM candidates")]


def find_result(arm, inst):
    for cand in [Path(f"results/arm_{arm}/{inst}_s0.json"),
                 Path(f"results/arm_{arm}/{inst}.json")]:
        if cand.exists():
            return json.load(open(cand))
    return None


def best_pdi(r):
    pdis = [h.get("pdi") for h in r.get("history", []) if h.get("pdi") is not None]
    pdis += [x["pdi"] for x in r.get("warmstart_results", [])]
    return min(pdis) if pdis else float("inf")


def main():
    insts = sorted(p.stem for p in Path("data/instances/118").glob("*.lp"))
    rows, wins = [], {a: 0 for a, _ in ARMS}
    improvements = {a: [] for a, _ in ARMS}
    for inst in insts:
        d = json.load(open(f"results/default/{inst}.json"))
        row, best_arm, best_val = [], None, float("inf")
        for arm, _ in ARMS:
            r = find_result(arm, inst)
            if r is None:
                row.append(None)
                continue
            bp = best_pdi(r)
            imp = (d["pdi"] - bp) / d["pdi"] * 100
            row.append(imp)
            improvements[arm].append(imp)
            if bp < best_val:
                best_val, best_arm = bp, arm
        if best_arm:
            wins[best_arm] += 1
        rows.append((inst, row))

    # markdown
    md = ["| 实例 | " + " | ".join(l.replace("\n", " ") for _, l in ARMS) + " |",
          "|" + "---|" * (len(ARMS) + 1)]
    for inst, row in rows:
        cells = [f"{v:+.1f}%" if v is not None else "—" for v in row]
        best_i = max((i for i, v in enumerate(row) if v is not None),
                     key=lambda i: row[i], default=None)
        if best_i is not None:
            cells[best_i] = "**" + cells[best_i] + "**"
        md.append(f"| {inst} | " + " | ".join(cells) + " |")
    n = len(rows)
    md.append("| **Wins** | " + " | ".join(
        f"**{wins[a]}/{n}**" for a, _ in ARMS) + " |")
    md.append("| 平均改善 | " + " | ".join(
        f"{sum(improvements[a]) / max(len(improvements[a]),1):+.1f}%"
        for a, _ in ARMS) + " |")
    text = "\n".join(md)
    Path("results/summary_table.md").write_text(text)
    print(text)

    # PNG
    fig, ax = plt.subplots(figsize=(10, 0.42 * (n + 2) + 1))
    ax.axis("off")
    cell_text = []
    for inst, row in rows:
        cell_text.append([inst] + [f"{v:+.1f}%" if v is not None else "—"
                                   for v in row])
    cell_text.append(["Wins"] + [f"{wins[a]}/{n}" for a, _ in ARMS])
    cell_text.append(["Mean"] + [
        f"{sum(improvements[a]) / max(len(improvements[a]),1):+.1f}%"
        for a, _ in ARMS])
    tbl = ax.table(cellText=cell_text,
                   colLabels=["Instance"] + [l for _, l in ARMS],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.5)
    # 颜色: 正绿负红, 每行最优加粗
    for i, (inst, row) in enumerate(rows, start=1):
        valid = [(j, v) for j, v in enumerate(row) if v is not None]
        if not valid:
            continue
        bj = max(valid, key=lambda x: x[1])[0]
        for j, v in valid:
            c = tbl[i, j + 1]
            c.set_facecolor("#e8f5e9" if v >= 0 else "#ffebee")
            if j == bj:
                c.set_text_props(weight="bold")
    for j in range(len(ARMS) + 1):
        tbl[n + 1, j].set_text_props(weight="bold")
        tbl[n + 2, j].set_text_props(weight="bold")
    ax.set_title("UC Tuner: PDI Improvement over Default (25-eval budget)",
                 fontsize=13, pad=14)
    fig.tight_layout()
    fig.savefig("results/figs/summary_table.png", dpi=160,
                bbox_inches="tight")
    print("-> results/figs/summary_table.png")


if __name__ == "__main__":
    main()
