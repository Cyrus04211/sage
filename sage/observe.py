"""实例特征提取: 静态特征 (模型结构) + 动态特征 (默认参数短时探测运行)。

供 Arm B/C 的 ASS (LLM 选参数) 与 warm-start (LLM 生成初始配置) 使用。
特征集合对齐 GRIMIP 论文附录 D 的 Table 4 / Table 5, 并增加 UC 专属特征。
动态特征通过解析 Gurobi 日志获得 (message callback 捕获)。
"""
import re
import time
from dataclasses import dataclass, field

import gurobipy as gp
import numpy as np


def static_features(m: gp.Model) -> dict:
    """从已加载的模型提取静态特征。"""
    feats = {
        "n_vars": m.NumVars,
        "n_constrs": m.NumConstrs,
        "n_nz": m.NumNZs,
        "n_bin": m.NumBinVars,
        "n_int": m.NumIntVars,
        "n_cont": m.NumVars - m.NumIntVars,   # NumIntVars 含 binary
        "n_sos": m.NumSOS,
        "density": m.NumNZs / max(m.NumVars * m.NumConstrs, 1),
        "binary_ratio": m.NumBinVars / max(m.NumVars, 1),
        "int_ratio": m.NumIntVars / max(m.NumVars, 1),
    }
    try:
        feats.update({
            "max_coeff": m.MaxCoeff, "min_coeff": m.MinCoeff,
            "max_rhs": m.MaxRHS, "min_rhs": m.MinRHS,
            "max_obj_coeff": m.MaxObjCoeff, "min_obj_coeff": m.MinObjCoeff,
        })
    except (gp.GurobiError, AttributeError):
        pass

    A = m.getA()                       # scipy CSR
    row_nnz = np.diff(A.indptr)
    feats["row_nnz_mean"] = float(row_nnz.mean())
    feats["row_nnz_std"] = float(row_nnz.std())

    senses = [c.Sense for c in m.getConstrs()]
    n = max(len(senses), 1)
    feats["pct_le"] = senses.count("<") / n
    feats["pct_ge"] = senses.count(">") / n
    feats["pct_eq"] = senses.count("=") / n

    objs = np.array([v.Obj for v in m.getVars()])
    feats["obj_nonzero_ratio"] = float((objs != 0).mean())
    nz_objs = objs[objs != 0]
    if len(nz_objs):
        feats["obj_coeff_mean"] = float(nz_objs.mean())
        feats["obj_coeff_std"] = float(nz_objs.std())

    feats.update(uc_features(m))
    return feats


def uc_features(m: gp.Model) -> dict:
    """UC 专属: 从变量命名 (如 delta_P(t,g,k)#i, u(g,t) 等) 推断机组数/时段数。"""
    names = [v.VarName for v in m.getVars()[:200000]]   # 采样上限防慢
    text = "\n".join(names)
    # 提取名字前缀 (去掉索引与 # 后缀)
    prefixes = re.findall(r"^([A-Za-z_][A-Za-z_0-9]*)\(", text, re.M)
    from collections import Counter
    top = Counter(prefixes).most_common(8)
    feats = {"var_prefixes": top}
    # 时段数: 对最常见前缀统计第一个索引的取值数
    if top:
        p0 = re.escape(top[0][0])
        idx = re.findall(p0 + r"\((\d+),", text)
        if idx:
            feats["uc_horizon_T"] = len(set(idx))
        gens = re.findall(p0 + r"\(\d+,(\d+),", text)
        if gens:
            feats["uc_n_generators"] = len(set(gens))
    return feats


# ---------------- 动态特征 (日志解析) ----------------

CUT_PATTERNS = {
    "Gomory": r"Gomory:\s*(\d+)",
    "Cover": r"Cover:\s*(\d+)",
    "MIR": r"MIR:\s*(\d+)",
    "Clique": r"Clique:\s*(\d+)",
    "FlowCover": r"Flow cover:\s*(\d+)",
    "ZeroHalf": r"Zero half:\s*(\d+)",
    "ImpliedBound": r"Implied bound:\s*(\d+)",
    "RLT": r"RLT:\s*(\d+)",
    "PSD": r"PSD:\s*(\d+)",
}


def dynamic_features(lp_path: str, probe_seconds: float = 60.0) -> dict:
    """默认参数探测运行, 解析日志提取动态特征。独立读模型 (评估器驻留的是另一份)。
    日志经 LogFile 落盘后解析 (MESSAGE callback 在部分 gurobipy 版本上抛错)。"""
    import tempfile
    m = gp.read(lp_path)
    m.Params.Threads = 1
    m.Params.TimeLimit = probe_seconds
    m.Params.LogToConsole = 0
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=True) as lf:
        m.Params.LogFile = lf.name
        m.optimize()
        m.Params.LogFile = ""
        lf.seek(0)
        log = lf.read()

    feats = {
        "probe_seconds": probe_seconds,
        "status": m.Status,
        "nodes_processed": int(m.NodeCount),
        "sol_count": m.SolCount,
        "mip_gap": None,
        "obj_bound": None,
        "best_obj": None,
    }
    try:
        feats["obj_bound"] = m.ObjBound
        if m.SolCount > 0:
            feats["best_obj"] = m.ObjVal
            feats["mip_gap"] = abs(m.ObjVal - m.ObjBound) / max(abs(m.ObjVal), 1e-10)
    except gp.GurobiError:
        pass

    # 根节点 LP 松弛
    rm = re.search(r"Root relaxation: objective ([\d.eE+-]+), (\d+) iterations, ([\d.]+) seconds", log)
    if rm:
        feats["root_lp_obj"] = float(rm.group(1))
        feats["root_lp_iters"] = int(rm.group(2))
        feats["root_lp_time"] = float(rm.group(3))

    pm = re.search(r"Presolve removed (\d+) rows and (\d+) columns", log)
    if pm:
        feats["presolve_rows_removed"] = int(pm.group(1))
        feats["presolve_cols_removed"] = int(pm.group(2))
    pt = re.search(r"Presolve time: ([\d.]+)s", log)
    if pt:
        feats["presolve_time"] = float(pt.group(1))

    for name, pat in CUT_PATTERNS.items():
        cm = re.search(pat, log)
        feats[f"cuts_{name}"] = int(cm.group(1)) if cm else 0
    feats["cuts_total"] = sum(v for k, v in feats.items() if k.startswith("cuts_"))

    fm = re.search(r"Found heuristic solution: objective ([\d.eE+-]+)", log)
    if fm:
        feats["first_heuristic_obj"] = float(fm.group(1))
    return feats


def feature_text(feats: dict) -> str:
    """把特征字典渲染成供 LLM 阅读的中文描述。"""
    lines = []
    s = feats.get("static", {})
    d = feats.get("dynamic", {})
    if s:
        lines.append(
            f"规模: {s.get('n_vars')} 变量 / {s.get('n_constrs')} 约束 / "
            f"{s.get('n_nz')} 非零元 (密度 {s.get('density', 0):.2e})")
        lines.append(
            f"变量结构: 二进制 {s.get('n_bin')} ({s.get('binary_ratio', 0):.1%}), "
            f"整数 {s.get('n_int')}, 连续 {s.get('n_cont')}")
        lines.append(
            f"约束类型: <= {s.get('pct_le', 0):.1%}, >= {s.get('pct_ge', 0):.1%}, "
            f"== {s.get('pct_eq', 0):.1%}; 每行非零均值 {s.get('row_nnz_mean', 0):.1f}")
        if s.get("uc_n_generators"):
            lines.append(
                f"UC 结构: 约 {s.get('uc_n_generators')} 台机组, "
                f"{s.get('uc_horizon_T')} 个时段; "
                f"主要变量族: {s.get('var_prefixes', [])[:4]}")
    if d:
        lines.append(
            f"探测运行 ({d.get('probe_seconds')}s 默认参数): "
            f"处理节点 {d.get('nodes_processed')}, 可行解 {d.get('sol_count')} 个, "
            f"gap {d.get('mip_gap')}")
        if d.get("root_lp_time") is not None:
            lines.append(
                f"根节点: LP 松弛 {d.get('root_lp_time')}s / {d.get('root_lp_iters')} 次迭代, "
                f"松弛目标 {d.get('root_lp_obj')}")
        lines.append(
            f"割平面: 总计 {d.get('cuts_total')} "
            f"(Gomory {d.get('cuts_Gomory', 0)}, MIR {d.get('cuts_MIR', 0)}, "
            f"Cover {d.get('cuts_Cover', 0)}, Clique {d.get('cuts_Clique', 0)}); "
            f"presolve 删 {d.get('presolve_rows_removed')} 行 / "
            f"{d.get('presolve_cols_removed')} 列")
    return "\n".join(lines)
