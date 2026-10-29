from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


STEP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STEP_DIR.parent
MAPPING = PROJECT_DIR / "03_神经符号EC映射" / "逐片段_约束混合EC映射.parquet"
RANKING = PROJECT_DIR / "05_EC拓扑传播与Jacobian分析" / "EC纯风险与拓扑调整排序_总体与品牌.csv"
REPETITIONS = 200
SEED = 20260831
ALPHA = 0.80
MIN_TAIL = 20


def weighted_cvar(sorted_values: np.ndarray, weights: np.ndarray) -> float:
    total = int(weights.sum())
    if total <= 0:
        return 0.0
    target = min(total, max(math.ceil((1.0 - ALPHA) * total), MIN_TAIL))
    prior = np.cumsum(weights) - weights
    take = np.clip(target - prior, 0, weights)
    return float(np.dot(sorted_values, take) / take.sum()) if take.sum() else 0.0


def bootstrap_cvar(values: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(REPETITIONS)
    codes, uniques = pd.factorize(groups, sort=False)
    order = np.argsort(values)[::-1]
    values = np.asarray(values, dtype=float)[order]
    codes = codes[order]
    rng = np.random.default_rng(seed)
    result = np.empty(REPETITIONS)
    for repeat in range(REPETITIONS):
        sampled = rng.integers(0, len(uniques), size=len(uniques))
        group_weights = np.bincount(sampled, minlength=len(uniques))
        result[repeat] = weighted_cvar(values, group_weights[codes])
    return result


def rank_comparison(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> dict:
    a = left.set_index("EC编号")["拓扑调整排名"].sort_index()
    b = right.set_index("EC编号")["拓扑调整排名"].reindex(a.index)
    top5a, top5b = set(a.nsmallest(5).index), set(b.nsmallest(5).index)
    top10a, top10b = set(a.nsmallest(10).index), set(b.nsmallest(10).index)
    return {
        "范围A": left_name,
        "范围B": right_name,
        "Spearman": float(spearmanr(a, b).statistic),
        "Kendall_tau_b": float(kendalltau(a, b).statistic),
        "Top5交集": len(top5a & top5b),
        "Top5_Jaccard": len(top5a & top5b) / len(top5a | top5b),
        "Top10交集": len(top10a & top10b),
        "Top10_Jaccard": len(top10a & top10b) / len(top10a | top10b),
        "平均绝对排名差": float(np.mean(np.abs(a.to_numpy() - b.to_numpy()))),
        "最大排名差": int(np.max(np.abs(a.to_numpy() - b.to_numpy()))),
    }


def main() -> None:
    mapped = pd.read_parquet(MAPPING, columns=["EC编号", "EC名称", "品牌归一", "Original_Index", "分摊初始势能"])
    ranking = pd.read_csv(RANKING)
    scope_filters = {"全部样本总体": None, "蔚来品牌": "蔚来", "比亚迪品牌": "比亚迪"}
    stability_rows = []
    for scope_index, (scope, brand) in enumerate(scope_filters.items()):
        scope_data = mapped if brand is None else mapped[mapped["品牌归一"] == brand]
        point = ranking[ranking["分析范围"] == scope].set_index("EC编号")
        ec_ids = point.sort_index().index.tolist()
        distributions = np.zeros((len(ec_ids), REPETITIONS), dtype=float)
        for ec_index, ec_id in enumerate(ec_ids):
            subset = scope_data[scope_data["EC编号"] == ec_id]
            distributions[ec_index] = bootstrap_cvar(
                subset["分摊初始势能"].to_numpy(dtype=float),
                subset["Original_Index"].astype(str).to_numpy(),
                SEED + scope_index * 1000 + ec_index,
            )
        topology = point.loc[ec_ids, "拓扑边际效用"].to_numpy(dtype=float)[:, None]
        scores = distributions * topology
        order = np.argsort(-scores, axis=0, kind="stable")
        ranks = np.empty_like(order)
        ranks[order, np.arange(REPETITIONS)] = np.arange(1, len(ec_ids) + 1)[:, None]
        for ec_index, ec_id in enumerate(ec_ids):
            stability_rows.append(
                {
                    "分析范围": scope,
                    "EC编号": ec_id,
                    "EC名称": point.loc[ec_id, "EC名称"],
                    "正式拓扑排名": int(point.loc[ec_id, "拓扑调整排名"]),
                    "Bootstrap排名中位数": float(np.median(ranks[ec_index])),
                    "Bootstrap排名95CI下限": float(np.quantile(ranks[ec_index], 0.025)),
                    "Bootstrap排名95CI上限": float(np.quantile(ranks[ec_index], 0.975)),
                    "Top5进入率": float(np.mean(ranks[ec_index] <= 5)),
                    "Top10进入率": float(np.mean(ranks[ec_index] <= 10)),
                    "CVaR80_Bootstrap均值": float(distributions[ec_index].mean()),
                    "CVaR80_95CI下限": float(np.quantile(distributions[ec_index], 0.025)),
                    "CVaR80_95CI上限": float(np.quantile(distributions[ec_index], 0.975)),
                }
            )
        print(f"[Bootstrap ranking] {scope} complete", flush=True)
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(STEP_DIR / "EC优先级Bootstrap稳定性.csv", index=False, encoding="utf-8-sig")
    comparisons = []
    scope_names = list(scope_filters)
    for i in range(len(scope_names)):
        for j in range(i + 1, len(scope_names)):
            comparisons.append(
                rank_comparison(
                    ranking[ranking["分析范围"] == scope_names[i]],
                    ranking[ranking["分析范围"] == scope_names[j]],
                    scope_names[i],
                    scope_names[j],
                )
            )
    pd.DataFrame(comparisons).to_csv(STEP_DIR / "总体与品牌排名一致性.csv", index=False, encoding="utf-8-sig")
    stable_priority = stability[(stability["正式拓扑排名"] <= 10) | (stability["Top10进入率"] >= 0.50)].copy()
    stable_priority["稳定性判定"] = np.select(
        [stable_priority["Top5进入率"] >= 0.80, stable_priority["Top10进入率"] >= 0.80],
        ["高稳定Top5", "高稳定Top10"],
        default="条件稳定",
    )
    stable_priority.sort_values(["分析范围", "正式拓扑排名"]).to_csv(
        STEP_DIR / "稳定EC优先级清单.csv", index=False, encoding="utf-8-sig"
    )
    audit = {
        "状态": "通过",
        "生成时间UTC": datetime.now(timezone.utc).isoformat(),
        "bootstrap": {"原始评论组重采样": True, "重复次数": REPETITIONS, "种子": SEED},
        "范围数": stability["分析范围"].nunique(),
        "结果行数": len(stability),
        "范围间比较数": len(comparisons),
        "稳定优先清单行数": len(stable_priority),
    }
    if len(stability) != 96 or len(comparisons) != 3:
        audit["状态"] = "失败"
    with (STEP_DIR / "优先级稳定性审计.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    if audit["状态"] != "通过":
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
