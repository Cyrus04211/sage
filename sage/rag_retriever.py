"""Arm C 的 RAG 检索 v2: 多查询扩展 + HyDE + LLM 重排过滤。

链路: 特征描述 -> 多路查询 (症状关键词 + HyDE 假设性专家笔记) -> BM25 各取 top-N
-> 并集去重 -> LLM 重排过滤 (只保留对本实例调参决策真正有用的条目, 拒绝噪声)
-> 注入 ASS/WS 提示词。

不做向量检索: 语料 ~2000 chunks 量级, BM25 + LLM 重排足够且无需嵌入模型。
"""
import json
import re
from pathlib import Path

import bm25s

HERE = Path(__file__).resolve().parents[1]
INDEX = HERE / "rag" / "index"

_chunks = None
_retriever = None


def _load():
    global _chunks, _retriever
    if _chunks is None:
        _chunks = [json.loads(l) for l in
                   (INDEX / "chunks.jsonl").read_text().splitlines()]
        _retriever = bm25s.BM25.load(str(INDEX / "bm25"), load_corpus=False)
    return _chunks, _retriever


def search(query: str, top_k: int = 5) -> list[dict]:
    chunks, retriever = _load()
    qtokens = bm25s.tokenize([query], stopwords="en")
    results, scores = retriever.retrieve(qtokens, k=top_k, corpus=chunks)
    return [{"score": float(s), **r} for r, s in zip(results[0], scores[0])]


def _symptom_queries(feature_text: str) -> list[str]:
    """从特征描述中提取症状关键词查询。"""
    qs = ["unit commitment MIP Gurobi parameter tuning"]
    if re.search(r"根节点.*?(?:[3-9]\d|\d{3,})s", feature_text):   # 根 LP 超 30s
        qs.append("slow root LP relaxation barrier method NodeMethod crossover")
    if "可行解 0 个" in feature_text or "sol_count 0" in feature_text:
        qs.append("find feasible solution quickly MIPFocus heuristics")
    if re.search(r"gap (?:0\.[1-9]|[1-9])", feature_text):        # gap 偏大
        qs.append("close optimality gap cutting planes aggressive cuts")
    qs.append("Gurobi MIPFocus Heuristics Cuts Presolve VarBranch tuning advice")
    return qs


_HYDE_PROMPT = """You are a Gurobi tuning expert. Given this MIP instance analysis, write a short
technical note (150 words, English) on which Gurobi parameters should be tuned
and why. Be concrete about parameter names and expected effects.

## Instance Feature Analysis
{feature_text}"""

_RERANK_PROMPT = """You are curating reference material for a Gurobi parameter tuning decision.

## Instance Feature Analysis
{feature_text}

## Candidate excerpts
{snippets}

## Task
Select the {n} excerpts that contain the MOST actionable, trustworthy guidance
for tuning THIS instance (parameter-level advice only). Reject excerpts that are
off-topic, marketing, or too generic to act on.

Return ONLY valid JSON: {{"picked": [<index>, ...], "reason": "<one line>"}}"""


def build_context(feature_text: str, params: list[dict] | None = None,
                  per_query: int = 4, final_n: int = 6,
                  max_chars: int = 4500, model: str | None = None) -> str:
    """多查询 + HyDE 检索, LLM 重排过滤后构建上下文。"""
    from sage.llm import chat, chat_json

    # 1. 多路查询: 症状关键词 + HyDE
    queries = _symptom_queries(feature_text)
    try:
        hyde = chat(_HYDE_PROMPT.format(feature_text=feature_text),
                    temperature=0.3, model=model)
        queries.append(hyde)
    except Exception:
        pass   # HyDE 失败不阻塞主链路

    # 2. 并集召回
    seen, pool = set(), []
    for q in queries:
        for h in search(q, top_k=per_query):
            if h["id"] not in seen:
                seen.add(h["id"])
                pool.append(h)

    # 3. LLM 重排过滤 (噪声防火墙)
    snippets = "\n\n".join(f"[{i}] ({h['title']})\n{h['text'][:400]}"
                           for i, h in enumerate(pool))
    try:
        out = chat_json(_RERANK_PROMPT.format(
            feature_text=feature_text[:1500], snippets=snippets,
            n=final_n), temperature=0.0, model=model)
        picked = [pool[i] for i in out.get("picked", [])
                  if isinstance(i, int) and 0 <= i < len(pool)]
    except Exception:
        picked = sorted(pool, key=lambda h: -h["score"])[:final_n]
    if not picked:
        picked = sorted(pool, key=lambda h: -h["score"])[:final_n]

    lines = ["## Retrieved Expert Knowledge (cite when used)"]
    n = 0
    for h in picked:
        block = f"\n[{h['title']} | {h['kind']}]\n{h['text'][:700]}"
        if n + len(block) > max_chars:
            break
        lines.append(block)
        n += len(block)
    return "\n".join(lines)
