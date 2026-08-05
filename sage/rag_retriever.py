"""Arm C 的 RAG 检索: 从手册+论文+专家经验语料中为 ASS/WS 构建上下文。

不做向量检索: 语料 ~2000 chunks, BM25 足够; "症状反查"由多个关键词查询覆盖。
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


def build_context(feature_text: str, params: list[dict] | None = None,
                  per_query: int = 3, max_chars: int = 4000) -> str:
    """为 LLM 决策构建检索上下文 (带引用来源)。"""
    seen, hits = set(), []
    for q in _symptom_queries(feature_text):
        for h in search(q, top_k=per_query):
            if h["id"] not in seen:
                seen.add(h["id"])
                hits.append(h)
    lines = ["## Retrieved Expert Knowledge (cite when used)"]
    n = 0
    for h in hits:
        block = f"\n[{h['title']} | {h['kind']}]\n{h['text'][:600]}"
        if n + len(block) > max_chars:
            break
        lines.append(block)
        n += len(block)
    return "\n".join(lines)
