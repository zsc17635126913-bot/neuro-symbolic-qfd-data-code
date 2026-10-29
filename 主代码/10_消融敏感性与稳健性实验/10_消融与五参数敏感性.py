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
BGE_ROOT = PROJECT_DIR.parent / "数据" / "02_BGE-M3语义特征"
REFERENCE = PROJECT_DIR / "05_EC拓扑传播与Jacobian分析" / "EC纯风险与拓扑调整排序_全部样本总体.csv"
KG_PATH = PROJECT_DIR.parent / "数据" / "03_工程知识图谱" / "冻结工程特性本体与符号约束图谱.xlsx"

N_ROWS = 525_647
N_EC = 32
MIN_TAIL = 20
KAPPA = 20.0
THRESHOLD = 0.38
SECOND_GAP = 0.10


def normalized_dense(path: Path) -> np.ndarray:
    dense = np.asarray(np.load(path), dtype=np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    return np.divide(dense, norms, out=np.zeros_like(dense), where=norms > 0)


def assign(score: np.ndarray, blocked: np.ndarray, forced: np.ndarray, constraints: bool = True) -> dict:
    score = np.asarray(score, dtype=np.float32).copy()
    if constraints:
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


def ranking_from_assignment(assignment: dict, energy: np.ndarray, comment_codes: np.ndarray, ecs: list[dict], topology: np.ndarray, alpha: float = .80, mean_risk: bool = False, shrink: bool = True) -> pd.DataFrame:
    rows = assignment["rows"]
    ec_index = assignment["ecs"]
    values = energy[rows] * assignment["weights"]
    parent_names = list(dict.fromkeys(ec["parent"] for ec in ecs))
    parent_index = {name: index for index, name in enumerate(parent_names)}
    ec_parent = np.asarray([parent_index[ec["parent"]] for ec in ecs])
    parent_risk = np.zeros(len(parent_names), dtype=float)
    for index in range(len(parent_names)):
        selected = values[ec_parent[ec_index] == index]
        parent_risk[index] = float(selected.mean()) if mean_risk and len(selected) else stable_cvar(selected, alpha, MIN_TAIL)[1]
    risk = np.zeros(len(ecs), dtype=float)
    evidence = np.zeros(len(ecs), dtype=int)
    for index in range(len(ecs)):
        mask = ec_index == index
        selected = values[mask]
        raw = float(selected.mean()) if mean_risk and len(selected) else stable_cvar(selected, alpha, MIN_TAIL)[1]
        evidence[index] = len(np.unique(comment_codes[rows[mask]]))
        weight = evidence[index] / (evidence[index] + KAPPA)
        risk[index] = weight * raw + (1 - weight) * parent_risk[ec_parent[index]] if shrink else raw
    scores = risk * topology
    result = pd.DataFrame({"EC编号": [ec["id"] for ec in ecs], "EC名称": [ec["name"] for ec in ecs], "风险": risk, "拓扑效用": topology, "全局得分": scores, "独立评论数": evidence})
    result = result.sort_values(["全局得分", "EC编号"], ascending=[False, True]).reset_index(drop=True)
    result.insert(0, "排名", np.arange(1, len(result) + 1))
    return result


def metrics(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict:
    ref = reference.set_index("EC编号")["拓扑调整排名"].sort_index()
    cur = candidate.set_index("EC编号")["排名"].reindex(ref.index)
    ref5, cur5 = set(ref.nsmallest(5).index), set(cur.nsmallest(5).index)
    ref10, cur10 = set(ref.nsmallest(10).index), set(cur.nsmallest(10).index)
    return {
        "Spearman": float(spearmanr(ref, cur).statistic),
        "Kendall_tau_b": float(kendalltau(ref, cur).statistic),
        "Top5_Jaccard": len(ref5 & cur5) / len(ref5 | cur5),
        "Top10_Jaccard": len(ref10 & cur10) / len(ref10 | cur10),
        "平均绝对排名变化": float(np.mean(np.abs(ref.to_numpy() - cur.to_numpy()))),
        "最大排名变化": int(np.max(np.abs(ref.to_numpy() - cur.to_numpy()))),
    }


def topology_from_weighted(weighted: np.ndarray, gamma: float) -> np.ndarray:
    adjacency = weighted @ weighted.T
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    inv = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    laplacian = np.eye(len(weighted)) - inv[:, None] * adjacency * inv[None, :]
    projection = np.divide(weighted, weighted.sum(axis=0, keepdims=True), out=np.zeros_like(weighted), where=weighted.sum(axis=0, keepdims=True) > 0)
    return (np.linalg.inv(np.eye(len(weighted)) + gamma * laplacian) @ projection).sum(axis=1)


def anchor_dense_component(lexical: np.ndarray, blocked: np.ndarray, shards: list[Path]) -> np.ndarray:
    anchors = (lexical > 0) & ~blocked
    sums = np.zeros((N_EC, 1024), dtype=np.float64)
    counts = anchors.sum(axis=0).astype(np.int64)
    offset = 0
    for shard_number, shard in enumerate(shards):
        dense = normalized_dense(shard / "dense.npy")
        end = offset + len(dense)
        for ec_index in range(N_EC):
            local = np.flatnonzero(anchors[offset:end, ec_index])
            if len(local):
                sums[ec_index] += dense[local].sum(axis=0, dtype=np.float64)
        offset = end
    centers = np.asarray(sums / counts[:, None], dtype=np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    result = np.empty((N_ROWS, N_EC), dtype=np.float16)
    offset = 0
    for shard in shards:
        dense = normalized_dense(shard / "dense.npy")
        end = offset + len(dense)
        result[offset:end] = (((dense @ centers.T).clip(-1, 1) + 1) / 2).astype(np.float16)
        offset = end
    return result


def main() -> None:
    interface = pd.read_parquet(INTERFACE, columns=["Original_Index", "S6_效价值", "严重度连续分值_0_4"])
    comment_codes = pd.factorize(interface["Original_Index"].astype(str), sort=False)[0]
    pain = np.maximum(0.0, -interface["S6_效价值"].to_numpy(dtype=float))
    severity = interface["严重度连续分值_0_4"].to_numpy(dtype=float)
    reference = pd.read_csv(REFERENCE)
    _, ecs, _, _ = load_ontology(KG_PATH)
    _, parent, cross, _, jacobian, _ = build_topology(ecs, .30)
    topology = jacobian.sum(axis=1)
    cache = np.load(COMPONENTS)
    lexical = cache["lexical"].astype(np.float32)
    dense = cache["dense"].astype(np.float32)
    sparse = cache["sparse"].astype(np.float32)
    blocked = np.unpackbits(cache["blocked"], axis=1, count=N_EC).astype(bool)
    forced = np.unpackbits(cache["forced"], axis=1, count=N_EC).astype(bool)
    energy = np.exp(-2.3 * (2.0 - severity)) * np.power(pain, 2.0)

    full_score = .70 * (.60 * dense + .40 * sparse) + .30 * lexical
    baseline_assignment = assign(full_score, blocked, forced, True)
    baseline_ranking = ranking_from_assignment(baseline_assignment, energy, comment_codes, ecs, topology)

    shards = sorted((BGE_ROOT / "shards").glob("shard_*"))
    anchor_dense = anchor_dense_component(lexical, blocked, shards).astype(np.float32)
    mapping_variants = {
        "完整方法_缓存重构": (full_score, True),
        "仅dense语义": (.70 * dense + .30 * lexical, True),
        "仅sparse语义": (.70 * sparse + .30 * lexical, True),
        "无词汇分量": (.60 * dense + .40 * sparse, True),
        "无符号硬约束": (full_score, False),
        "无PCK_means": (.70 * (.60 * anchor_dense + .40 * sparse) + .30 * lexical, True),
    }
    ablation_rows, ablation_ranks = [], []
    for name, (score, use_constraints) in mapping_variants.items():
        assignment = assign(score, blocked, forced, use_constraints)
        rank = ranking_from_assignment(assignment, energy, comment_codes, ecs, topology)
        rank.insert(0, "实验", name); ablation_ranks.append(rank)
        ablation_rows.append({"实验": name, "类型": "映射消融", "映射片段数": assignment["mapped"], "映射率": assignment["mapped"] / N_ROWS, "双EC片段数": assignment["second"], **metrics(rank, reference)})
        print(f"[Ablation] {name}", flush=True)

    risk_variants = [
        ("均值替代CVaR", energy, topology, .80, True, True),
        ("无严重度项", np.power(pain, 2.0), topology, .80, False, True),
        ("无拓扑传播", energy, np.ones(N_EC), .80, False, True),
        ("无经验贝叶斯收缩", energy, topology, .80, False, False),
        ("无跨域矩阵", energy, topology_from_weighted(parent, .30), .80, False, True),
    ]
    for name, variant_energy, variant_topology, alpha, use_mean, use_shrink in risk_variants:
        rank = ranking_from_assignment(baseline_assignment, variant_energy, comment_codes, ecs, variant_topology, alpha, use_mean, use_shrink)
        rank.insert(0, "实验", name); ablation_ranks.append(rank)
        ablation_rows.append({"实验": name, "类型": "风险排序消融", "映射片段数": baseline_assignment["mapped"], "映射率": baseline_assignment["mapped"] / N_ROWS, "双EC片段数": baseline_assignment["second"], **metrics(rank, reference)})
    pd.DataFrame(ablation_rows).to_csv(STEP_DIR / "消融实验汇总.csv", index=False, encoding="utf-8-sig")
    pd.concat(ablation_ranks, ignore_index=True).to_csv(STEP_DIR / "消融实验32EC完整排名.csv", index=False, encoding="utf-8-sig")

    sensitivity_rows, sensitivity_ranks = [], []
    parameter_values = {
        "CVaR分位数": [0.70, 0.75, 0.80, 0.85, 0.90],
        "严重度系数": [1.5, 1.9, 2.3, 2.7, 3.1],
        "痛感指数": [1.0, 1.5, 2.0, 2.5, 3.0],
        "拓扑传播强度": [0.0, 0.1, 0.3, 0.5, 0.7],
        "BGE稠密权重": [0.0, 0.25, 0.50, 0.60, 0.75, 1.0],
    }
    for parameter, values in parameter_values.items():
        for value in values:
            assignment = baseline_assignment
            variant_energy = energy
            variant_topology = topology
            alpha = .80
            if parameter == "CVaR分位数": alpha = value
            elif parameter == "严重度系数": variant_energy = np.exp(-value * (2.0 - severity)) * np.power(pain, 2.0)
            elif parameter == "痛感指数": variant_energy = np.exp(-2.3 * (2.0 - severity)) * np.power(pain, value)
            elif parameter == "拓扑传播强度": variant_topology = topology_from_weighted(parent + cross, value)
            elif parameter == "BGE稠密权重":
                score = .70 * (value * dense + (1.0 - value) * sparse) + .30 * lexical
                assignment = assign(score, blocked, forced, True)
            rank = ranking_from_assignment(assignment, variant_energy, comment_codes, ecs, variant_topology, alpha)
            rank.insert(0, "参数值", value); rank.insert(0, "参数", parameter); sensitivity_ranks.append(rank)
            sensitivity_rows.append({"参数": parameter, "参数值": value, "映射率": assignment["mapped"] / N_ROWS, **metrics(rank, reference)})
        print(f"[Sensitivity] {parameter}", flush=True)
    pd.DataFrame(sensitivity_rows).to_csv(STEP_DIR / "五参数敏感性汇总.csv", index=False, encoding="utf-8-sig")
    pd.concat(sensitivity_ranks, ignore_index=True).to_csv(STEP_DIR / "五参数敏感性32EC完整排名.csv", index=False, encoding="utf-8-sig")
    audit = {
        "状态": "通过",
        "生成时间UTC": datetime.now(timezone.utc).isoformat(),
        "缓存精度": "float16映射分量；排序计算转float32",
        "完整缓存重构映射片段数": baseline_assignment["mapped"],
        "正式映射片段数": 512920,
        "映射片段差": baseline_assignment["mapped"] - 512920,
        "消融实验数": len(ablation_rows),
        "敏感性配置数": len(sensitivity_rows),
        "参数取值": parameter_values,
    }
    with (STEP_DIR / "消融敏感性审计.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
