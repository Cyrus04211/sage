"""单实例 Gurobi 评估器: 读模型一次, 多次评估不同参数配置, 输出 PDI。

PDI 计算 (与 GRIMIP 附录 E 一致, 左黎曼和):
  PDI = sum_i  Gap(t_i) * (t_{i+1} - t_i),  Gap(t) = Z_primal(t) - Z_dual(t)

关键处理: 找到首个可行解之前 Gurobi 的 primal bound 为 +inf (1e100),
直接代入会使 PDI 爆炸。采用追溯填充: 首个可行解出现之前的时段,
primal bound 取首个可行解的目标值; 整轮无可行解时使用 reference_obj
(该实例历史最优目标值) 作为虚拟 primal bound, 若也无参考则按
Gap = |dual(t)| + 1 兜底 (保证 PDI 为正且量级可控)。
"""
import time
from dataclasses import dataclass, field

import gurobipy as gp


@dataclass
class EvalResult:
    pdi: float
    runtime: float
    status: int                 # GRB status code
    obj: float | None           # 最优可行解目标值 (无可行解为 None)
    bound: float | None         # 最终对偶界
    nodes: int
    timed_out: bool
    n_points: int = 0           # PDI 采样点数
    extra: dict = field(default_factory=dict)


class InstanceEvaluator:
    """对单个 MIP 实例反复评估参数配置。模型驻留内存, 评估间 reset 复用。"""

    def __init__(self, lp_path: str, time_limit: float = 300.0,
                 threads: int = 1, reference_obj: float | None = None,
                 output: bool = False):
        self.lp_path = lp_path
        self.time_limit = time_limit
        self.threads = threads
        self.reference_obj = reference_obj
        self.output = output
        self._model: gp.Model | None = None

    def _load(self):
        if self._model is None:
            self._model = gp.read(self.lp_path)
            self._model.Params.OutputFlag = 0

    @property
    def model(self) -> gp.Model:
        self._load()
        return self._model

    def evaluate(self, config: dict | None = None,
                 time_limit: float | None = None) -> EvalResult:
        """评估一个参数配置。config 之外的参数保持 Gurobi 默认。"""
        self._load()
        m = self._model
        m.reset()
        m.resetParams()
        m.Params.OutputFlag = 1 if self.output else 0
        m.Params.Threads = self.threads
        m.Params.TimeLimit = time_limit or self.time_limit
        for k, v in (config or {}).items():
            m.setParam(k, v)

        rec = {"t": [0.0], "primal": [None], "dual": [None]}
        t0 = time.time()

        def cb(model, where):
            if where == gp.GRB.Callback.MIP:
                rec["t"].append(time.time() - t0)
                bst = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                bnd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
                inf = gp.GRB.INFINITY
                rec["primal"].append(bst if abs(bst) < inf / 10 else None)
                rec["dual"].append(bnd if abs(bnd) < inf / 10 else None)

        m.optimize(cb)
        runtime = time.time() - t0

        pdi, n_points = self._compute_pdi(rec)
        obj = m.ObjVal if m.SolCount > 0 else None
        bound = None
        try:
            bound = m.ObjBound
        except gp.GurobiError:
            pass
        if obj is not None and (self.reference_obj is None
                                or obj < self.reference_obj):
            self.reference_obj = obj
        return EvalResult(
            pdi=pdi, runtime=runtime, status=m.Status, obj=obj, bound=bound,
            nodes=int(m.NodeCount), timed_out=(m.Status == gp.GRB.Status.TIME_LIMIT),
            n_points=n_points,
        )

    def _compute_pdi(self, rec) -> tuple[float, int]:
        ts = rec["t"][1:]
        primals = rec["primal"][1:]
        duals = rec["dual"][1:]
        if not ts:
            return float("inf"), 0
        # 首个可行解目标值 (追溯填充用)
        first_inc = next((p for p in primals if p is not None), None)
        virtual = first_inc if first_inc is not None else self.reference_obj

        pdi = 0.0
        prev_t = 0.0
        for t, p, d in zip(ts, primals, duals):
            if d is None:
                prev_t = t
                continue
            if p is None:
                p = virtual if virtual is not None else d + abs(d) + 1.0
            gap = max(p - d, 0.0)          # 最小化问题 primal >= dual
            pdi += gap * (t - prev_t)
            prev_t = t
        # 结尾到 time_limit 的尾段按最后 gap 延伸 (左黎曼和语义)
        limit = self.time_limit
        if ts[-1] < limit and duals[-1] is not None:
            p = primals[-1]
            if p is None:
                p = virtual if virtual is not None else duals[-1] + abs(duals[-1]) + 1.0
            pdi += max(p - duals[-1], 0.0) * (limit - ts[-1])
        return pdi, len(ts)
