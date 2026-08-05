"""检查 Arm C 某次运行的 RAG 检索内容。用法:
PYTHONPATH=. .venv/bin/python scripts/inspect_rag.py results/arm_c/model2797.json
"""
import json
import sys

from sage.rag_retriever import _symptom_queries, search

def main(path):
    r = json.load(open(path))
    ft = r["feature_text"]
    print("=== 实例特征描述(前 400 字符) ===")
    print(ft[:400])
    print("\n=== 检索查询与命中 (★ = 去重后入选上下文) ===")
    seen = set()
    for q in _symptom_queries(ft):
        print("Q:", q)
        for h in search(q, top_k=3):
            tag = "*" if h["id"] not in seen else " "
            seen.add(h["id"])
            print(" %s [%s] %s (score %.1f)" % (tag, h["kind"], h["title"][:70], h["score"]))
    print("\n=== 该次运行的 ASS 选择 ===")
    for s in r.get("selection", []):
        print(" ", s["name"], "-", s["reason"][:110])

if __name__ == "__main__":
    main(sys.argv[1])
