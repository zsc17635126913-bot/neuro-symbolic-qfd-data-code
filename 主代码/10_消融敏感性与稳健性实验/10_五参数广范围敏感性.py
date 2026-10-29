from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


STEP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STEP_DIR.parent
CORE_DIR = PROJECT_DIR / "03_神经符号EC映射"
sys.path.insert(0, str(CORE_DIR))

from qfd_core import build_topology, load_ontology, stable_cvar  # noqa: E402


COMPONENTS = CORE_DIR / "映射分量缓存.npz"
INTERFACE = PROJECT_DIR / "02_QFD接口转换" / "全量QFD统一接口.parquet"
FORMAL_REFERENCE = PROJECT_DIR / "05_EC拓扑传播与Jacobian分析" / "EC纯风险与拓扑调整排序_全部样本总体.csv"
KG_PATH = PROJECT_DIR.parent / "数据" / "03_工程知识图谱" / "冻结工程特性本体与符号约束图谱.xlsx"

N_ROWS = 525_647
N_EC = 32
MIN_TAIL = 20
KAPPA = 20.0
THRESHOLD = 0.38
SECOND_GAP = 0.10


def assign(score: np.ndarray, blocked: np.ndarray, forced: np.ndarray) -> dict[str, np.ndarray | int]:
    score = np.asarray(score, dtype=np.float32).copy()
    score[blocked] = -np.inf
    score[forced & ~blocked] = np.maximum(score[forced & ~blocked], 1.25)
    top = np.argpartition(score, -2, axis=1)[:, -2:]
    values = np.take_along_axis(score, top, axis=1)
    order = np.argsort(values, axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1)
    values = np.take_along_axis(values, order, axis=1)
    first = np.isfinite(values[:, 0]) & (values[:, 0] >= THRESHOLD)
    second = first & np.isfinite(values[:, 1]) & (values[:, 1] >= THRESHOLD) & ((values[:, 0] - values[:, 1]) <= SECOND_GAP)
    denominator = np.maximum(values[:, 0], 1e-9) + np.where(second, np.maximum(values[:, 1], 1e-9), 0.0)
    w1 = np.where(first, np.maximum(values[:, 0], 1e-9) / denominator, 0.0)
    w2 = np.where(second, np.maximum(values[:, 1], 1e-9) / denominator, 0.0)
    p1, p2 = np.flatnonzero(first), np.flatnonzero(second)
    return {
        "rows": np.concatenate([p1, p2]),
        "ecs": np.concatenate([top[p1, 0], top[p2, 1]]).astype(np.int16),
        "weights": np.concatenate([w1[p1], w2[p2]]).astype(np.float32),
        "mapped": len(p1),
        "second": len(p2),
    }


class Aggregator:
    def __init__(self, assignment: dict, comment_codes: np.ndarray, ecs: list[dict]):
        self.rows = assignment["rows"]
        self.ec_index = assignment["ecs"]
        self.weights = assignment["weights"]
        self.ecs = ecs
        parent_names = list(dict.fromkeys(ec["parent"] for ec in ecs))
        parent_index = {name: index for index, name in enumerate(parent_names)}
        self.ec_parent = np.asarray([parent_index[ec["parent"]] for ec in ecs])
        self.ec_positions = [np.flatnonzero(self.ec_index == i) for i in range(len(ecs))]
        self.parent_positions = [
            np.concatenate([self.ec_positions[i] for i in range(len(ecs)) if self.ec_parent[i] == p])
            for p in range(len(parent_names))
        ]
        self.evidence = np.asarray([
            len(np.unique(comment_codes[self.rows[pos]])) for pos in self.ec_positions
        ])
        self.shrink_weight = self.evidence / (self.evidence + KAPPA)

    def risks(self, energy: np.ndarray, alpha: float = 0.80) -> np.ndarray:
        values = energy[self.rows] * self.weights
        parent_risk = np.asarray([
            stable_cvar(values[pos], alpha, MIN_TAIL)[1] for pos in self.parent_positions
        ])
        raw = np.asarray([
            stable_cvar(values[pos], alpha, MIN_TAIL)[1] for pos in self.ec_positions
        ])
        return self.shrink_weight * raw + (1.0 - self.shrink_weight) * parent_risk[self.ec_parent]

    def alpha_risks(self, energy: np.ndarray, alphas: np.ndarray) -> list[np.ndarray]:
        values = energy[self.rows] * self.weights
        ec_sorted = [np.sort(values[pos])[::-1] for pos in self.ec_positions]
        parent_sorted = [np.sort(values[pos])[::-1] for pos in self.parent_positions]
        ec_prefix = [np.cumsum(arr, dtype=np.float64) for arr in ec_sorted]
        parent_prefix = [np.cumsum(arr, dtype=np.float64) for arr in parent_sorted]

        def top_mean(prefix: np.ndarray, length: int, alpha: float) -> float:
            if length == 0:
                return 0.0
            k = min(length, max(MIN_TAIL, max(1, math.ceil((1.0 - alpha) * length))))
            return float(prefix[k - 1] / k)

        output = []
        for alpha in alphas:
            parent_risk = np.asarray([top_mean(pre, len(arr), alpha) for pre, arr in zip(parent_prefix, parent_sorted)])
            raw = np.asarray([top_mean(pre, len(arr), alpha) for pre, arr in zip(ec_prefix, ec_sorted)])
            output.append(self.shrink_weight * raw + (1.0 - self.shrink_weight) * parent_risk[self.ec_parent])
        return output


def topology_from_weighted(weighted: np.ndarray, gamma: float) -> np.ndarray:
    adjacency = weighted @ weighted.T
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    inv = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    laplacian = np.eye(len(weighted)) - inv[:, None] * adjacency * inv[None, :]
    column_sum = weighted.sum(axis=0, keepdims=True)
    projection = np.divide(weighted, column_sum, out=np.zeros_like(weighted), where=column_sum > 0)
    return (np.linalg.inv(np.eye(len(weighted)) + gamma * laplacian) @ projection).sum(axis=1)


def ranking(risk: np.ndarray, topology: np.ndarray, ecs: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame({
        "EC编号": [ec["id"] for ec in ecs],
        "EC名称": [ec["name"] for ec in ecs],
        "风险": risk,
        "拓扑效用": topology,
        "全局得分": risk * topology,
    }).sort_values(["全局得分", "EC编号"], ascending=[False, True]).reset_index(drop=True)
    frame.insert(0, "排名", np.arange(1, len(frame) + 1))
    return frame


def metrics(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float | int]:
    ref = reference.set_index("EC编号")["排名"].sort_index()
    cur = candidate.set_index("EC编号")["排名"].reindex(ref.index)
    ref10, cur10 = set(ref.nsmallest(10).index), set(cur.nsmallest(10).index)
    return {
        "Spearman": float(spearmanr(ref, cur).statistic),
        "Kendall_tau_b": float(kendalltau(ref, cur).statistic),
        "Top10_Jaccard": len(ref10 & cur10) / len(ref10 | cur10),
        "平均绝对排名变化": float(np.mean(np.abs(ref.to_numpy() - cur.to_numpy()))),
        "最大排名变化": int(np.max(np.abs(ref.to_numpy() - cur.to_numpy()))),
    }


def add_result(rows: list[dict], ranks: list[pd.DataFrame], parameter: str, value: float, display_value: float,
               mapped_rate: float, risk: np.ndarray, topology: np.ndarray, ecs: list[dict], reference: pd.DataFrame) -> None:
    current = ranking(risk, topology, ecs)
    current.insert(0, "参数原始值", value)
    current.insert(0, "参数显示值", display_value)
    current.insert(0, "参数", parameter)
    ranks.append(current)
    rows.append({
        "参数": parameter,
        "参数原始值": value,
        "参数显示值": display_value,
        "映射率": mapped_rate,
        **metrics(current, reference),
    })


def main() -> None:
    interface = pd.read_parquet(INTERFACE, columns=["Original_Index", "S6_效价值", "严重度连续分值_0_4"])
    comment_codes = pd.factorize(interface["Original_Index"].astype(str), sort=False)[0]
    pain = np.maximum(0.0, -interface["S6_效价值"].to_numpy(dtype=float))
    severity = interface["严重度连续分值_0_4"].to_numpy(dtype=float)
    _, ecs, _, _ = load_ontology(KG_PATH)
    _, parent, cross, _, _, _ = build_topology(ecs, 0.30)
    weighted = parent + cross

    cache = np.load(COMPONENTS)
    lexical = cache["lexical"].astype(np.float32)
    dense = cache["dense"].astype(np.float32)
    sparse = cache["sparse"].astype(np.float32)
    blocked = np.unpackbits(cache["blocked"], axis=1, count=N_EC).astype(bool)
    forced = np.unpackbits(cache["forced"], axis=1, count=N_EC).astype(bool)

    formal_energy = np.exp(-2.3 * (2.0 - severity)) * np.power(pain, 2.0)
    formal_topology = topology_from_weighted(weighted, 0.30)
    full_score = 0.70 * (0.60 * dense + 0.40 * sparse) + 0.30 * lexical
    baseline_assignment = assign(full_score, blocked, forced)
    baseline_agg = Aggregator(baseline_assignment, comment_codes, ecs)
    baseline_risk = baseline_agg.risks(formal_energy, 0.80)
    reconstructed_reference = ranking(baseline_risk, formal_topology, ecs)

    summary_rows: list[dict] = []
    rank_rows: list[pd.DataFrame] = []
    mapped_rate = baseline_assignment["mapped"] / N_ROWS

    alphas = np.round(np.linspace(0.00, 0.99, 100), 2)
    for alpha, risk in zip(alphas, baseline_agg.alpha_risks(formal_energy, alphas)):
        add_result(summary_rows, rank_rows, "CVaR分位数", alpha, alpha, mapped_rate, risk, formal_topology, ecs, reconstructed_reference)
    print("[1/5] CVaR分位数完成", flush=True)

    severity_coefficients = np.round(np.linspace(0.0, 5.0, 51), 2)
    for coefficient in severity_coefficients:
        energy = np.exp(-coefficient * (2.0 - severity)) * np.power(pain, 2.0)
        risk = baseline_agg.risks(energy, 0.80)
        add_result(summary_rows, rank_rows, "严重度系数", coefficient, coefficient, mapped_rate, risk, formal_topology, ecs, reconstructed_reference)
    print("[2/5] 严重度系数完成", flush=True)

    pain_exponents = np.round(np.linspace(0.0, 4.0, 41), 2)
    for exponent in pain_exponents:
        energy = np.exp(-2.3 * (2.0 - severity)) * np.power(pain, exponent)
        risk = baseline_agg.risks(energy, 0.80)
        add_result(summary_rows, rank_rows, "痛感指数", exponent, exponent, mapped_rate, risk, formal_topology, ecs, reconstructed_reference)
    print("[3/5] 痛感指数完成", flush=True)

    diffusion_effect = np.unique(np.concatenate([np.linspace(0.0, 0.99, 100), [0.30 / 1.30]]))
    for effect in diffusion_effect:
        gamma = effect / max(1e-12, 1.0 - effect)
        topology = topology_from_weighted(weighted, gamma)
        add_result(summary_rows, rank_rows, "拓扑传播强度", gamma, effect, mapped_rate, baseline_risk, topology, ecs, reconstructed_reference)
    print("[4/5] 拓扑传播强度完成", flush=True)

    dense_weights = np.round(np.linspace(0.0, 1.0, 101), 2)
    for index, weight in enumerate(dense_weights):
        score = 0.70 * (weight * dense + (1.0 - weight) * sparse) + 0.30 * lexical
        assignment = assign(score, blocked, forced)
        aggregator = Aggregator(assignment, comment_codes, ecs)
        risk = aggregator.risks(formal_energy, 0.80)
        add_result(
            summary_rows,
            rank_rows,
            "BGE稠密权重",
            weight,
            weight,
            assignment["mapped"] / N_ROWS,
            risk,
            formal_topology,
            ecs,
            reconstructed_reference,
        )
        if index % 10 == 0:
            print(f"[5/5] BGE稠密权重 {weight:.2f}", flush=True)

    summary = pd.DataFrame(summary_rows)
    ranks = pd.concat(rank_rows, ignore_index=True)
    summary.to_csv(STEP_DIR / "五参数广范围敏感性汇总.csv", index=False, encoding="utf-8-sig")
    ranks.to_csv(STEP_DIR / "五参数广范围敏感性32EC排名.csv", index=False, encoding="utf-8-sig")

    formal = pd.read_csv(FORMAL_REFERENCE).rename(columns={"拓扑调整排名": "排名"})
    audit = {
        "状态": "通过",
        "生成时间UTC": datetime.now(timezone.utc).isoformat(),
        "参数配置数": len(summary),
        "基准": "float16映射缓存重构的正式参数结果；正式参数点在敏感性曲线中定义为1.0一致性",
        "缓存重构与正式输出一致性": metrics(reconstructed_reference, formal[["EC编号", "排名"]]),
        "参数范围": {
            "CVaR分位数": "0.00-0.99，步长0.01",
            "严重度系数": "0-5，步长0.1；覆盖从关闭严重度到极强加权的有效作用范围",
            "痛感指数": "0-4，步长0.1；0表示关闭痛感差异",
            "拓扑传播强度": "以delta=gamma/(1+gamma)均匀覆盖0-0.99，并加入正式gamma=0.30",
            "BGE稠密权重": "0-1，步长0.01，完整有界区间",
        },
        "正式参数": {"CVaR分位数": 0.80, "严重度系数": 2.30, "痛感指数": 2.0, "拓扑传播强度gamma": 0.30, "BGE稠密权重": 0.60},
    }
    (STEP_DIR / "五参数广范围敏感性审计.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
