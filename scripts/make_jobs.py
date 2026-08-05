"""批量并发运行三臂实验的任务驱动器。

从 jobs.txt 读取任务 (每行: arm instance seed), 用 xargs -P 控制并发数。
每个任务内部仍是单进程单线程 Gurobi; 峰值内存 ≈ 并发数 × ~3GB,
503GB 机器上并发 16 也很安全。

用法 (服务器, 项目根目录):
  bash scripts/run_batch.sh jobs.txt 12        # 12 路并发

jobs.txt 每行格式:  <arm:A|B|C> <instance.lp 相对路径> <seed>
示例:
  A data/instances/118/model2797.lp 0
  B data/instances/118/model2797.lp 0
  C data/instances/118/model2797.lp 0
"""
import subprocess
import sys
from pathlib import Path

def make_jobs(instances, arms, seeds, out="jobs.txt"):
    lines = []
    for inst in instances:
        for arm in arms:
            for seed in seeds:
                lines.append(f"{arm} {inst} {seed}")
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {out}")

if __name__ == "__main__":
    # 示例: python scripts/make_jobs.py
    insts = sorted(str(p) for p in Path("data/instances/118").glob("*.lp"))
    make_jobs(insts, ["A", "B", "C"], [0])
