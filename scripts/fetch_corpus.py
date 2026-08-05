"""抓取 Arm C 扩展 RAG 语料: 论文 / Gurobi 官方经验 / 开源项目文档。

输出: rag/corpus/raw/ (原始文件) + rag/corpus/txt/ (纯文本, 供切块索引)
用法: python3 scripts/fetch_corpus.py   (在本机执行后 rsync 到服务器)
"""
import json
import re
import subprocess
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "rag" / "corpus" / "raw"
TXT = HERE / "rag" / "corpus" / "txt"

SOURCES = [
    # (id, 类型, 标题, url; url=None 表示本地已有)
    {"id": "grimip", "kind": "paper",
     "title": "GRIMIP: Instance-Specific Configuration of MIP Solvers Using LLMs",
     "url": None, "local": "paper.txt"},   # 项目根目录已有
    {"id": "hutter2010", "kind": "paper",
     "title": "Automated Configuration of Mixed Integer Programming Solvers (Hutter et al., CPAIOR 2010)",
     "url": "https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/10-CPAIOR-MIP-Config.pdf"},
    {"id": "knueven2020", "kind": "paper",
     "title": "On Mixed Integer Programming Formulations for the Unit Commitment Problem",
     "url": "https://optimization-online.org/wp-content/uploads/2018/11/6930.pdf"},
    {"id": "gurobi_tuning", "kind": "official",
     "title": "Gurobi Parameter Tuning Tool 官方文档",
     "url": "https://docs.gurobi.com/projects/optimizer/en/13.0/features/tuning.html"},
    {"id": "gurobi_support_tuning", "kind": "official",
     "title": "Gurobi Support: What is parameter tuning",
     "url": "https://support.gurobi.com/hc/en-us/articles/19998635021713-What-is-parameter-tuning"},
    {"id": "gurobi_uc_talk", "kind": "official",
     "title": "Gurobi 用户会议: 能源系统/UC 调参实践 (Yfantis)",
     "url": "https://cdn.gurobi.com/wp-content/uploads/Vassilios-Yfantis_Gurobi.pdf?x81293"},
    {"id": "genbench_milp", "kind": "repo",
     "title": "GenBench-MILP / EVA-MILP (含 SMAC3 调 Gurobi 8 参数的实践)",
     "url": "https://arxiv.org/html/2505.24779v2"},
    {"id": "ucjl", "kind": "repo",
     "title": "UnitCommitment.jl 文档 (UC 实例与求解器设置)",
     "url": "https://anl-ceeesa.github.io/UnitCommitment.jl/0.3/instances/"},
    {"id": "openmod_mip", "kind": "forum",
     "title": "openmod 论坛: MIP 求解器参数设置讨论",
     "url": "https://forum.openmod.org/t/cplex-settings-for-mip-optimization/2482"},
    {"id": "klotz_newman_mip", "kind": "paper",
     "title": "Practical Guidelines for Solving Difficult Mixed Integer Linear Programs (Klotz & Newman, SORMS 2013)",
     "url": "https://people.mines.edu/anewman/wp-content/uploads/sites/158/2019/11/28-MIP_practice120212.pdf"},
    {"id": "klotz_newman_lp", "kind": "paper",
     "title": "Practical Guidelines for Solving Difficult Linear Programs (Klotz & Newman, SORMS 2013)",
     "url": "https://people.mines.edu/anewman/wp-content/uploads/sites/158/2019/11/27-LP_practice123112.pdf"},
    {"id": "berthold_heuristics", "kind": "paper",
     "title": "A computational study of primal heuristics inside an MI(NL)P solver (Berthold, JGO 2018)",
     "url": "https://link.springer.com/content/pdf/10.1007/s10898-017-0600-3.pdf"},
    {"id": "fischetti_variability", "kind": "paper",
     "title": "Improving branch-and-cut performance by random sampling (Fischetti et al., MPC 2016)",
     "url": "https://www.dei.unipd.it/~salvagni/pdf/ksample.pdf"},
    {"id": "gamrath_presolve", "kind": "paper",
     "title": "Progress in presolving for mixed integer programming (Gamrath et al., MPC 2015)",
     "url": "https://link.springer.com/content/pdf/10.1007/s12532-015-0083-5.pdf"},
    {"id": "gurobi_logs", "kind": "official",
     "title": "Gurobi Support: Interpreting Gurobi Logs (如何读 MIP 日志判断瓶颈)",
     "url": "https://support.gurobi.com/hc/en-us/articles/39810125521937-Interpreting-Gurobi-Logs"},
    {"id": "gurobi_cuts", "kind": "official",
     "title": "Gurobi Support: Understanding cutting planes and how to make use of them",
     "url": "https://support.gurobi.com/hc/en-us/articles/44924553715345-Understanding-cutting-planes-and-how-to-make-use-of-them"},
    {"id": "gurobi_numericguide", "kind": "official",
     "title": "Gurobi 官方文档: Guidelines for Numerical Issues (NumericFocus/ScaleFlag 等)",
     "url": "https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide.html"},  # txt 另合并了 numericguide/numeric_parameters.html 子页
    {"id": "miplib_selection", "kind": "official",
     "title": "MIPLIB 2017 实例选择方法 (特征与难度评估)",
     "url": "https://miplib.zib.de/Selection_Methodology.html"},
]


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer"):
            self.skip += 1
        elif tag in ("p", "li", "h1", "h2", "h3", "h4", "tr", "section"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self):
        t = "".join(self.parts)
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n\s*\n+", "\n\n", t)
        return t.strip()


def fetch(url: str, dest: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    TXT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for s in SOURCES:
        try:
            if s.get("local"):
                text = Path(s["local"]).read_text(errors="ignore")
            elif s["url"].endswith(".pdf") or ".pdf?" in s["url"]:
                raw = RAW / f"{s['id']}.pdf"
                if not raw.exists():
                    fetch(s["url"], raw)
                subprocess.run(["pdftotext", str(raw), str(TXT / f"{s['id']}.txt")],
                               check=True)
                manifest.append({**s, "status": "ok"})
                print(f"[ok] {s['id']} (pdf)")
                continue
            else:
                raw = RAW / f"{s['id']}.html"
                if not raw.exists():
                    fetch(s["url"], raw)
                ex = TextExtractor()
                ex.feed(raw.read_text(errors="ignore"))
                text = ex.text()
            (TXT / f"{s['id']}.txt").write_text(text)
            manifest.append({**s, "status": "ok", "chars": len(text)})
            print(f"[ok] {s['id']} ({len(text)} chars)")
        except Exception as e:
            manifest.append({**s, "status": f"failed: {e}"})
            print(f"[fail] {s['id']}: {e}")
    (HERE / "rag" / "corpus" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
