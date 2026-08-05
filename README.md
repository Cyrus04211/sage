# SAGE: Agentic Gurobi Tuning for Unit Commitment MIPs

LLM 代理 + SMAC3 的 Gurobi 求解器参数调优系统，面向机组组合（UC）MIP 实例。
三臂对比实验框架：

- **Arm A**：SMAC3 直接在全参数空间（131 个性能相关参数）调 20 次——高维基线；
- **Arm B**：LLM 观察实例结构（静态特征 + 60s 探测运行动态特征）→ 选出 6 参数子空间（ASS）
  → 生成 5 个 warm-start 配置（另算预算）→ SMAC3 在子空间调 20 次；
- **Arm C**：Arm B + RAG 专家先验（Gurobi 13.0 手册 + 相关论文 + 官方调参经验 +
  开源项目实践，BM25 检索注入 ASS/WS 提示词）。

## 目录结构

```
sage/               核心包
├── evaluator.py    单实例评估器（模型驻留 + reset 复用，PDI callback）
├── params.py       手册参数解析 → ConfigSpace（全空间/子空间，settable 探测过滤）
├── observe.py      实例特征提取（ISAC 静态特征 + UC 专属 + 探测运行动态特征）
├── agent_decisions.py  LLM 决策（ASS 选参 + warm-start 生成，JSON 校验/范围裁剪/确定性补位）
├── llm.py          LLM API 客户端（OpenAI 兼容协议，读 .env）
├── rag_retriever.py    BM25 检索与上下文构建
└── arms/           run_arm_a.py（全空间）/ run_arm_b.py（ASS+WS+SMAC，--rag 为 Arm C）
rag/                手册解析（parse_manual.py）+ 语料抓取（scripts/fetch_corpus.py）+ 索引构建
scripts/            批量驱动 run_batch.sh、任务生成 make_jobs.py、对比 compare.py 等
docs/DESIGN.md      完整框架设计文档
```

## 初步实验结果（2 实例，IEEE-118 UC，PDI 越小越好）

默认配置基线：model2797 PDI 3.35e7（36.0s 证优）；model1337 PDI 2.82e7（39.4s 证优）。

| 实例 | 臂 | 最佳 PDI | 相对默认 | 评估次数 | 调参耗时 |
|---|---|---|---|---|---|
| model2797 | Arm A（全 131 参数） | 4.94e7 | **−47.6%** | 20 | 1.33h |
| model2797 | Arm B（6 参数子空间） | **1.87e7** | **+44.1%** | 5 ws + 20 | 0.24h |
| model1337 | Arm A（全 131 参数） | 5.56e8 | **−1873%** | 20 | 1.26h |
| model1337 | Arm B（6 参数子空间） | **2.03e7** | **+27.9%** | 5 ws + 20 | 0.39h |

Arm B 的 LLM 选参（model2797: MIPFocus/MIPSepCuts/MIRCuts/ImpliedCuts/VarBranch/Heuristics；
model1337: Presolve/Aggregate/MIRCuts/CutPasses/Symmetry/MIPFocus）均为实例自适应选择。

## 运行环境

- 求解器：Gurobi 13.0.1（`gurobipy==13.0.1`）
- Python 3.12 venv；依赖：`smac==2.4.0`（需 `scikit-learn>=1.6.1,<1.8`）、`ConfigSpace`、
  `openai`、`langchain-openai`、`langgraph`、`bm25s`、`optuna-fast-fanova`、`numpy<2`
- LLM：OpenAI 兼容 API（`.env` 配置 `OPENAI_BASE_URL`/`OPENAI_API_KEY`/`LLM_MODEL`，不进库）

## 用法

```bash
# 手册解析 + RAG 索引（一次性）
python3 rag/parse_manual.py && python3 scripts/fetch_corpus.py && python3 rag/build_index.py

# 单臂运行
PYTHONPATH=. python -m sage.arms.run_arm_a --instance <lp> --trials 20 --out results/arm_a/x.json
PYTHONPATH=. python -m sage.arms.run_arm_b --instance <lp> --trials 20 --rag --out results/arm_c/x.json

# 批量并发（jobs.txt 每行: arm instance seed）
python scripts/make_jobs.py && bash scripts/run_batch.sh jobs.txt 12

# 结果对比
PYTHONPATH=. python scripts/compare.py model2797 model1337
```
