"""解析 Gurobi 13.0 参数参考页 (parameters.html) 为结构化 JSON。

每个参数一条记录:
  name        参数名 (h2 标题)
  brief       一句话功能描述
  type        int / double / string
  default     默认值 (原始字符串)
  min / max   取值范围 (原始字符串; MAXINT/Infinity 保留原文)
  description 完整说明文字 (纯文本, 已去 HTML)
  mip_only    说明中是否标注 "Only affects mixed integer programming (MIP) models"
  notes       其他限制说明 (如 "Barrier only")

用法: python3 sage/rag/parse_manual.py  (读取同目录 parameters.html, 输出 params.json)
"""
import json
import re
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).parent


class TextExtractor(HTMLParser):
    """把一段 HTML 转成纯文本(保留段落换行)。"""

    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "li", "tr"):
            self.parts.append("\n")

    def text(self):
        t = "".join(self.parts)
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n\s*\n+", "\n", t)
        return t.strip()


def parse(html: str) -> list[dict]:
    # 顶层 <section id="..."> 对应一个参数; 跳过非参数 section
    sections = re.split(r'<section id="', html)[1:]
    records = []
    for sec in sections:
        sec_id, _, rest = sec.partition('"')
        # 参数 section 内都有 <span id="parameter...">
        if f'id="parameter{sec_id.lower()}"' not in rest:
            continue
        body = rest.split("</section>", 1)[0]

        m = re.search(r"<h2>(.*?)<a class=\"headerlink\"", body, re.S)
        if not m:
            continue
        name = re.sub(r"<[^>]+>", "", m.group(1)).strip()

        # Type / Default / Min / Max
        fields = {}
        for label, key in [("Type", "type"), ("Default value", "default"),
                           ("Minimum value", "min"), ("Maximum value", "max")]:
            fm = re.search(
                label + r':\s*<code[^>]*>\s*<span class="pre">(.*?)</span>',
                body, re.S)
            fields[key] = re.sub(r"<[^>]+>", "", fm.group(1)).strip() if fm else None

        # 一句话功能: h2 之后的第一个 <p>
        bm = re.search(r"</h2>\s*<p>(.*?)</p>", body, re.S)
        brief = TextExtractor()
        if bm:
            brief.feed(bm.group(1))
        brief_text = brief.text().replace("\n", " ")

        # 完整说明: 整个 section 纯文本化, 去掉 h2/brief/字段清单行
        full = TextExtractor()
        full.feed(body)
        desc_lines = []
        for line in full.text().split("\n"):
            s = line.strip()
            if not s or s == name or s == brief_text:
                continue
            if re.match(r"^(Type|Default value|Minimum value|Maximum value):", s):
                continue
            desc_lines.append(s)
        description = "\n".join(desc_lines)

        notes = []
        if "Only affects mixed integer programming (MIP) models" in description:
            notes.append("MIP only")
        for tag in ("Barrier only", "Barrier and PDHG only", "Cluster Manager only",
                    "Compute Server only"):
            if tag in description:
                notes.append(tag)

        records.append({
            "name": name,
            "brief": brief_text,
            "type": fields["type"],
            "default": fields["default"],
            "min": fields["min"],
            "max": fields["max"],
            "mip_only": "MIP only" in notes,
            "notes": notes,
            "description": description,
            "source": f"https://docs.gurobi.com/projects/optimizer/en/13.0/reference/parameters.html#{sec_id}",
        })
    return records


def main():
    html = (HERE / "parameters.html").read_text(encoding="utf-8")
    records = parse(html)
    out = HERE / "params.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=1))
    typed = sum(1 for r in records if r["type"])
    mip = sum(1 for r in records if r["mip_only"])
    print(f"parsed {len(records)} params ({typed} with type info, {mip} MIP-only)")
    for r in records[:3]:
        print(" -", r["name"], "|", r["type"], "|", r["brief"][:60])


if __name__ == "__main__":
    main()
