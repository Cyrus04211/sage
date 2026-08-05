"""从 params.json (手册解析结果) 构建 SMAC 搜索空间。

Arm A 使用"全部参数"空间: 手册中全部影响单次本地 MIP 求解性能的参数。
剔除规则 (有据可查, 全部记录在 excluded.json):
  - string 类型参数 (license/文件路径/服务器地址等)
  - 基础设施类: Cloud*/CS*/ComputeServer/Worker*/TokenServer/WLS*/Group/Router 等
  - 日志输出类: OutputFlag/LogFile/DisplayInterval/LogToConsole/Record 等
  - 调参工具自身: Tune* 参数
  - 终止/截断准则: TimeLimit/NodeLimit/SolutionLimit/WorkLimit/MemLimit/
    IterationLimit/BarIterLimit/Cutoff/BestObjStop/BestBdStop/MIPGap/MIPGapAbs
    (调它们改变求解语义而非搜索行为)
  - 并行相关: Threads/Concurrent*/Distributed* (协议固定单线程)
  - 结果写出类: ResultFile/SolFiles/JsonSolDetail/Write* 等

范围处理: MAXINT 截断为 1000; Infinity 截断为 1e12;
跨度大且非负的数值范围用 log 标度。
"""
import json
from pathlib import Path

from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

MAXINT_CAP = 1000
INF_CAP = 1e12

EXCLUDE_EXACT = {
    # 终止/截断
    "TimeLimit", "NodeLimit", "IterationLimit", "BarIterLimit", "SolutionLimit",
    "WorkLimit", "MemLimit", "Cutoff", "BestObjStop", "BestBdStop",
    "MIPGap", "MIPGapAbs", "ObjectiveLimit",
    # 并行 (协议固定单线程)
    "Threads", "ConcurrentMIP", "ConcurrentJobs", "ConcurrentMethod",
    "ConcurrentSettings", "DistributedMIPJobs",
    # 日志/输出
    "OutputFlag", "LogFile", "DisplayInterval", "LogToConsole", "Record",
    "ResultFile", "SolFiles", "JsonSolDetail",
    # 调参工具自身
    "TuneCriterion", "TuneJobs", "TuneMetric", "TuneOutput", "TuneResults",
    "TuneTargetMIPGap", "TuneTargetTime", "TuneTimeLimit", "TuneTrials",
    "TuneBaseSettings", "TuneParams", "TuneIgnoreSettings",
    # 其他非性能
    "Seed", "RandomSeed", "IgnoreNames", "ScenarioNumber", "ObjNumber",
    "LicenseID", "Username", "ServerPassword", "WLSAccessID", "WLSSecret",
    "CSAPIAccessID", "CSAPISecret", "CSAuthToken", "CSAppName", "CSBatchMode",
    "CSClientLog", "CSGroup", "CSIdleTimeout", "CSManager", "CSPriority",
    "CSQueueTimeout", "CSRouter", "CSTLSInsecure",
    "CloudAccessID", "CloudHost", "CloudSecretKey", "CloudPool",
    "ComputeServer", "WorkerPool", "WorkerPassword", "TokenServer",
    "ServerTimeout", "ConnectionName",
    "MultiObjMethod", "MultiObjPre",  # 单目标 UC 用不到
}
EXCLUDE_PREFIX = ("Tune", "Cloud", "CS", "Worker")


def _parse_num(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if s in ("MAXINT", "INFINITY", "Infinity", "+Infinity"):
        return float("inf")
    if s in ("-Infinity", "-INFINITY"):
        return float("-inf")
    if s in ("", '""'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_params(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def tunable_params(params: list[dict]) -> tuple[list[dict], list[dict]]:
    """返回 (纳入的参数, 被剔除的参数及原因)。"""
    keep, drop = [], []
    for p in params:
        name, ptype = p["name"], p["type"]
        if ptype == "string" or ptype is None:
            drop.append({**p, "reason": "string/untyped"})
            continue
        if name in EXCLUDE_EXACT or any(name.startswith(pre) for pre in EXCLUDE_PREFIX):
            drop.append({**p, "reason": "excluded (non-performance/termination/infra)"})
            continue
        lo, hi = _parse_num(p["min"]), _parse_num(p["max"])
        default = _parse_num(p["default"])
        if lo is None or hi is None or lo >= hi:
            drop.append({**p, "reason": "bad range"})
            continue
        keep.append({**p, "lo": lo, "hi": hi, "default_val": default})
    return keep, drop


def filter_settable(params: list[dict]) -> tuple[list[dict], list[dict]]:
    """探测过滤: 在空模型上用非默认值试 setParam, 剔除环境级/只读参数。

    注意必须用非默认值: Gurobi 对部分参数(如 ServerTimeout)在设置为某些
    取值时才报 "Unable to modify ... after environment started"。
    """
    import gurobipy as gp
    probe = gp.Model()
    probe.Params.OutputFlag = 0
    keep, drop = [], []
    for p in params:
        d = p["default_val"]
        v = p["hi"] if d is None or p["hi"] != d else p["lo"]
        if p["type"] == "int":
            v = int(round(min(v, MAXINT_CAP)))
        else:
            v = float(min(v, INF_CAP))
        try:
            probe.setParam(p["name"], v)
            keep.append(p)
        except gp.GurobiError as e:
            drop.append({**p, "reason": f"not settable: {e}"})
    return keep, drop


def build_full_space(params: list[dict], seed: int = 0) -> ConfigurationSpace:
    """Arm A 的全参数 ConfigSpace。"""
    cs = ConfigurationSpace(seed=seed)
    for p in params:
        name, lo, hi, default = p["name"], p["lo"], p["hi"], p["default_val"]
        lo_c = min(lo, MAXINT_CAP) if p["type"] == "int" else lo
        hi_c = min(hi, MAXINT_CAP) if p["type"] == "int" else min(hi, INF_CAP)
        lo_c = max(lo_c, -INF_CAP)
        if default is None or not (lo_c <= default <= hi_c):
            default = lo_c
        span_pos = lo_c > 0 and hi_c / lo_c > 100
        if p["type"] == "int":
            cs.add(Integer(name, (int(round(lo_c)), int(round(hi_c))),
                           default=int(round(default)), log=span_pos))
        else:
            cs.add(Float(name, (lo_c, hi_c), default=default, log=span_pos))
    return cs


def build_subspace(params: list[dict], selection: list[dict],
                   seed: int = 0) -> ConfigurationSpace:
    """Arm B/C: 按 LLM 选择构建子空间。selection = [{"name":..., "min":..., "max":...}]，
    范围必须 ⊆ 手册范围 (调用前由 agent 层校验)。"""
    by_name = {p["name"]: p for p in params}
    cs = ConfigurationSpace(seed=seed)
    for sel in selection:
        p = by_name[sel["name"]]
        lo = max(float(sel.get("min", p["lo"])), p["lo"])
        hi = min(float(sel.get("max", p["hi"])), p["hi"])
        if p["type"] == "int":
            lo, hi = min(lo, MAXINT_CAP), min(hi, MAXINT_CAP)
        else:
            hi = min(hi, INF_CAP)
        default = p["default_val"]
        if default is None or not (lo <= default <= hi):
            default = lo
        if p["type"] == "int":
            cs.add(Integer(p["name"], (int(round(lo)), int(round(hi))),
                           default=int(round(default))))
        else:
            cs.add(Float(p["name"], (lo, hi), default=default))
    return cs
