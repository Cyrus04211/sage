"""构建扩展 RAG 索引: Gurobi 手册参数 + 论文/官方经验/开源项目语料。

语料来源:
  rag/params.json           214 个参数的结构化记录 (一参数一 chunk)
  rag/corpus/txt/*.txt      外部文档 (按段落切块, ~1200 字符上限)

输出: rag/index/chunks.jsonl + rag/index/bm25/ (bm25s 持久化)
用法: python3 rag/build_index.py   (服务器或本机均可)
"""
import json
import re
from pathlib import Path

import bm25s

HERE = Path(__file__).resolve().parent
OUT = HERE / "index"
CHUNK_MAX = 1200


def chunk_text(text: str, source_id: str, title: str, kind: str) -> list[dict]:
    paras = re.split(r"\n\s*\n", text)
    chunks, buf = [], ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) > CHUNK_MAX and buf:
            chunks.append(buf)
            buf = p
        else:
            buf = (buf + "\n" + p).strip()
    if buf:
        chunks.append(buf)
    return [{"id": f"{source_id}#{i}", "source_id": source_id, "title": title,
             "kind": kind, "text": c} for i, c in enumerate(chunks)]


def main():
    chunks = []
    # 1. 手册参数
    params = json.loads((HERE / "params.json").read_text())
    for p in params:
        text = (f"{p['name']}: {p['brief']}\n"
                f"Type: {p['type']}, Default: {p['default']}, "
                f"Range: [{p['min']}, {p['max']}]\n{p['description'][:800]}")
        chunks.append({"id": f"manual#{p['name']}", "source_id": "gurobi_manual",
                       "title": f"Gurobi 参数 {p['name']}", "kind": "manual",
                       "text": text})
    # 2. 外部语料
    manifest = json.loads((HERE / "corpus" / "manifest.json").read_text())
    for s in manifest:
        if s["status"] != "ok":
            continue
        txt = (HERE / "corpus" / "txt" / f"{s['id']}.txt").read_text(errors="ignore")
        chunks += chunk_text(txt, s["id"], s["title"], s["kind"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks))

    corpus_tokens = bm25s.tokenize([c["text"] for c in chunks], stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(OUT / "bm25"))
    print(f"indexed {len(chunks)} chunks -> {OUT}")


if __name__ == "__main__":
    main()
