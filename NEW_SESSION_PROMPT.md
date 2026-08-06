# 任务：在新的 5 个 MIP 数据集上跑 SAGE 五臂调参对比（目标：最小化求解时间）

## 背景

我们有一个已完成的 Gurobi 调参实验框架 SAGE（代码在 GitHub 仓库 Cyrus04211/sage，私有）。它实现了 5 个调参臂，在 16 个 IEEE-118 UC 实例上完成了 PDI 口径和求解时间口径的全量对比。现在要把同样的五臂实验搬到 5 个新数据集上：**set cover / maximum independent set / multiple knapsack / corlat / mik**，目标为**最小化求解时间**（300s 封顶，超时计 300s），与默认 Gurobi 对比。

## 两台服务器（本机公钥免密均可直连）

- **实验服务器**（跑实验）：`ssh -p 222 lgl@10.26.17.9`，项目目录 `/home/lgl/projects/sage`，venv 在 `.venv`（全部依赖已装好），Gurobi 13.0.1 + 学术 license 可用，LLM API 配置在 `.env`（OpenAI 兼容端点 + 本机代理 127.0.0.1:7890 才能访问外网，openai SDK 自动读 HTTP_PROXY）。当前使用模型 `gpt-5.6-terra`。
- **数据服务器**（实例所在）：`ssh -p 221 wcg@10.26.17.8`，实例信息在 `/nas1/wcg/SCIPAgent/EvoSCIP/InstanceEvoSCIP/experiment_state/five_family_http_100ep_pat20_v1/campaigns`。先用 ssh 探查该目录结构，弄清 5 个数据集的实例文件（.lp/.mps）在哪、各多少个、规模多大。**两台服务器之间是否能直接互传不确定，必要时经本机中转（rsync -A 路径经本地）**。实例小的话复制到实验服务器 `/home/lgl/nas1/instances/` 下的新目录；大的话软链接、禁止复制。

## 五臂定义（全部 per-instance，评估预算统一 25 次）

- **Arm A**：SMAC3 在全参数空间（131 个性能相关参数，`sage/params.py` 从手册解析+过滤生成）调 25 次；
- **Arm D**：SMAC3 在固定六维 Θ={MIPFocus, Heuristics, Cuts, Presolve, Method, VarBranch} 调 25 次（GRIMIP 论文基线设置）；
- **Arm B**：LLM 观察实例特征（静态+60s 探测运行动态）→ 选 6 参数子空间（ASS）→ 生成 5 个 warm-start 配置（评估另算）→ SMAC 在子空间调 20 次（warm-start 结果 tell 进 SMAC）；
- **Arm C**：Arm B + RAG v2（Gurobi 13.0 手册 + 18 源专家语料，BM25+HyDE+LLM 重排，注入 ASS/WS 提示词）；
- **Arm E**：Arm B + LLM 候选生成（20 次搜索中 SMAC/LLM 交替提议，各约 10 次）。

所有臂：Gurobi 单线程、单次评估 300s 上限、metric=runtime。

## 怎么跑（全部已实现，直接复用）

```bash
# 实验服务器 /home/lgl/projects/sage 下
# 1) 把新实例放到 data/ 下（软链接或复制小实例）
# 2) 生成任务单: 每行 "<臂> <实例相对路径> <种子>"，臂 ∈ A B C D E
# 3) 批量跑（tmux 脱离会话；OUTDIR 分开避免覆盖）:
tmux new-session -d -s rt_sc "cd /home/lgl/projects/sage && OUTDIR=results_rt_sc METRIC=runtime TRIALS=25 bash scripts/run_batch.sh jobs_sc.txt 12 > results_rt_sc/batch.log 2>&1"
```

- 并发 12 路以内（每进程约 3GB，内存 503GB；CPU 128 核）；实例小的话并发可以更高。
- 默认基线：对每个实例跑 `scripts/run_default.py`（先确认它对非 118 目录的路径处理，必要时小改）。
- 汇总脚本：`scripts/summary_runtime.py`（求解时间口径表）、`scripts/plot_convergence.py`（best-so-far 曲线）、`scripts/cross_metric.py`（交叉口径）。这些脚本里实例目录/结果目录目前是写死的 118 路径，用前改成参数或复制改一份。

## 已踩过的坑（别再踩）

- SMAC 2.4.0 需要 `scikit-learn>=1.6.1,<1.8`；`Scenario` 里**不要传** `trial_walltime_limit`（会静默耗尽 trials 预算）；时限由评估器 Gurobi TimeLimit 保证。
- 参数空间必须过 `filter_settable`（剔除 ServerTimeout 等环境级参数）；LLM 输出裁剪用 `_clamp_config`（与 ConfigSpace 的 MAXINT_CAP=1000/INF_CAP 口径一致）。
- SMAC `tell()` 预评估计入 n_trials；续跑用"重放 tell"+`--resume`。
- 服务器无外网直连，pip 用清华镜像，LLM/HF 走 127.0.0.1:7890 代理。
- 长时间任务一律 tmux + 输出重定向，不要靠 ssh 前台。
- RAG 索引已建在服务器 `rag/index/`（18 源语料+手册 214 参数）。如果新实例不是 UC，`observe.py` 里的 UC 专属特征会自动退化，无需改；但建议检查 `feature_text` 模板对新族是否通顺。

## 重要：先做可行性检查

之前 118 UC 实例全部太简单（默认 60s 内近优），导致"调参 vs 默认"的差异空间很小。**开工前先对每个新数据集抽样做 60s 默认参数探测**（`sage.observe.dynamic_features`），报告各族的求解时间分布；如果某族全部秒解，要么放弃该族、要么延长/缩短时限，先和我商量再放量。

## 第一批交付

1. 数据服务器目录探查结果 + 5 族实例清单与规模统计；
2. 各族 2-3 个实例的默认探测结果（难度分布）;
3. 和我确认实例选择后，挂 5 臂 × 5 族的后台批次；
4. 完成后出：求解时间五臂对比表（按族分组）、收敛曲线、与 118 UC 结果的横向比较。
