from __future__ import annotations

import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STEP_DIR.parent
CORE_DIR = PROJECT_DIR / "03_神经符号EC映射"
sys.path.insert(0, str(CORE_DIR))

from qfd_core import load_ontology, stable_cvar  # noqa: E402


MAPPING = CORE_DIR / "逐片段_约束混合EC映射.parquet"
INTERFACE = PROJECT_DIR / "02_QFD接口转换" / "全量QFD统一接口.parquet"
RISK = PROJECT_DIR / "04_风险势能与CVaR聚合" / "EC_CVaR80风险_总体与品牌.csv"
RANKING = PROJECT_DIR / "05_EC拓扑传播与Jacobian分析" / "EC纯风险与拓扑调整排序_总体与品牌.csv"
KG_PATH = PROJECT_DIR.parent / "数据" / "03_工程知识图谱" / "冻结工程特性本体与符号约束图谱.xlsx"

BOOTSTRAP_REPETITIONS = 300
BOOTSTRAP_SEED = 20260831
CVaR_ALPHA = 0.80
MIN_TAIL = 20
SHRINKAGE_KAPPA = 20.0
EPSILON = 1e-12


def weighted_cvar(sorted_values: np.ndarray, sorted_weights: np.ndarray) -> float:
    total = int(sorted_weights.sum())
    if total <= 0:
        return 0.0
    target = min(total, max(math.ceil((1.0 - CVaR_ALPHA) * total), MIN_TAIL))
    prior = np.cumsum(sorted_weights) - sorted_weights
    take = np.clip(target - prior, 0, sorted_weights)
    denominator = float(take.sum())
    return float(np.dot(sorted_values, take) / denominator) if denominator else 0.0


def cluster_bootstrap_cvar(values: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(BOOTSTRAP_REPETITIONS, dtype=float)
    codes, uniques = pd.factorize(groups, sort=False)
    order = np.argsort(values)[::-1]
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_codes = codes[order]
    n_groups = len(uniques)
    rng = np.random.default_rng(seed)
    result = np.empty(BOOTSTRAP_REPETITIONS, dtype=float)
    for repeat in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.integers(0, n_groups, size=n_groups)
        group_weights = np.bincount(sampled, minlength=n_groups)
        result[repeat] = weighted_cvar(sorted_values, group_weights[sorted_codes])
    return result


def aggregate_model_risk(mapped: pd.DataFrame, ecs: list[dict]) -> pd.DataFrame:
    rows = []
    for model, model_data in mapped.groupby("车型", sort=False):
        parent_values = {}
        for parent in list(dict.fromkeys(ec["parent"] for ec in ecs)):
            values = model_data.loc[model_data["正式一级归属"] == parent, "分摊初始势能"].to_numpy(dtype=float)
            parent_values[parent] = stable_cvar(values, CVaR_ALPHA, MIN_TAIL)[1]
        for ec in ecs:
            subset = model_data[model_data["EC编号"] == ec["id"]]
            values = subset["分摊初始势能"].to_numpy(dtype=float)
            strict, stable, strict_k, stable_k = stable_cvar(values, CVaR_ALPHA, MIN_TAIL)
            n_comments = int(subset["Original_Index"].nunique())
            weight = n_comments / (n_comments + SHRINKAGE_KAPPA)
            shrunk = weight * stable + (1.0 - weight) * parent_values[ec["parent"]]
            rows.append(
                {
                    "车型": model,
                    "品牌归一": "蔚来" if model == "蔚来总体" else "比亚迪",
                    "EC编号": ec["id"],
                    "EC名称": ec["name"],
                    "正式一级归属": ec["parent"],
                    "映射行数": len(subset),
                    "独立评论数": n_comments,
                    "CVaR80_严格": strict,
                    "CVaR80_稳定": stable,
                    "严格尾部数": strict_k,
                    "稳定尾部数": stable_k,
                    "收缩后稳定风险": shrunk,
                    "证据等级": "充分" if n_comments >= 50 else "一般" if n_comments >= 20 else "不足",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    mapped = pd.read_parquet(MAPPING)
    interface = pd.read_parquet(INTERFACE)
    risk = pd.read_csv(RISK)
    ranking = pd.read_csv(RANKING)
    _, ecs, _, _ = load_ontology(KG_PATH)

    mapped_ids = mapped[["品牌归一", "片段行号"]].drop_duplicates()
    input_summary = (
        interface.groupby("品牌归一", observed=True)
        .agg(
            输入片段数=("片段行号", "size"),
            独立评论数=("Original_Index", "nunique"),
            负向片段率=("P_情感_负向", "mean"),
            平均效价=("S6_效价值", "mean"),
            平均严重度概率期望=("严重度连续分值_0_4", "mean"),
        )
        .reset_index()
    )
    mapped_summary = mapped.groupby("品牌归一", observed=True).agg(映射行数=("片段行号", "size"), 独立映射片段数=("片段行号", "nunique"), 分摊势能总和=("分摊初始势能", "sum")).reset_index()
    input_summary = input_summary.merge(mapped_summary, on="品牌归一", how="left")
    input_summary["映射率"] = input_summary["独立映射片段数"] / input_summary["输入片段数"]
    input_summary.to_csv(STEP_DIR / "品牌输入与风险证据摘要.csv", index=False, encoding="utf-8-sig")

    nio = risk[risk["分析范围"] == "蔚来品牌"].set_index("EC编号")
    byd = risk[risk["分析范围"] == "比亚迪品牌"].set_index("EC编号")
    nio_rank = ranking[ranking["分析范围"] == "蔚来品牌"].set_index("EC编号")
    byd_rank = ranking[ranking["分析范围"] == "比亚迪品牌"].set_index("EC编号")
    brand_rows = []
    for ec_index, ec in enumerate(ecs):
        ec_data = mapped[mapped["EC编号"] == ec["id"]]
        distributions = {}
        for brand_index, brand in enumerate(("蔚来", "比亚迪")):
            subset = ec_data[ec_data["品牌归一"] == brand]
            distributions[brand] = cluster_bootstrap_cvar(
                subset["分摊初始势能"].to_numpy(dtype=float),
                subset["Original_Index"].astype(str).to_numpy(),
                BOOTSTRAP_SEED + ec_index * 10 + brand_index,
            )
        difference_distribution = distributions["比亚迪"] - distributions["蔚来"]
        point_nio = float(nio.loc[ec["id"], "收缩后稳定风险"])
        point_byd = float(byd.loc[ec["id"], "收缩后稳定风险"])
        brand_rows.append(
            {
                "EC编号": ec["id"],
                "EC名称": ec["name"],
                "正式一级归属": ec["parent"],
                "蔚来收缩风险": point_nio,
                "比亚迪收缩风险": point_byd,
                "风险差_比亚迪减蔚来": point_byd - point_nio,
                "风险比_比亚迪除蔚来": point_byd / max(point_nio, EPSILON),
                "log2风险比": math.log2((point_byd + EPSILON) / (point_nio + EPSILON)),
                "蔚来拓扑排名": int(nio_rank.loc[ec["id"], "拓扑调整排名"]),
                "比亚迪拓扑排名": int(byd_rank.loc[ec["id"], "拓扑调整排名"]),
                "排名差_比亚迪减蔚来": int(byd_rank.loc[ec["id"], "拓扑调整排名"] - nio_rank.loc[ec["id"], "拓扑调整排名"]),
                "原始CVaR差Bootstrap均值": float(difference_distribution.mean()),
                "原始CVaR差95CI下限": float(np.quantile(difference_distribution, 0.025)),
                "原始CVaR差95CI上限": float(np.quantile(difference_distribution, 0.975)),
                "双侧方向概率P": float(min(1.0, 2 * min(np.mean(difference_distribution <= 0), np.mean(difference_distribution >= 0)))),
                "差异方向": "比亚迪更高" if point_byd > point_nio else "蔚来更高" if point_nio > point_byd else "相同",
            }
        )
        if (ec_index + 1) % 8 == 0:
            print(f"[Brand bootstrap] {ec_index + 1}/32", flush=True)
    brand_difference = pd.DataFrame(brand_rows).sort_values("风险差_比亚迪减蔚来", key=np.abs, ascending=False)
    brand_difference.to_csv(STEP_DIR / "同一EC品牌风险差异.csv", index=False, encoding="utf-8-sig")

    model_risk = aggregate_model_risk(mapped, ecs)
    model_risk.to_csv(STEP_DIR / "同一EC车型风险长表.csv", index=False, encoding="utf-8-sig")
    pair_rows = []
    for ec in ecs:
        subset = model_risk[model_risk["EC编号"] == ec["id"]].set_index("车型")
        for left, right in itertools.combinations(subset.index.tolist(), 2):
            left_value = float(subset.loc[left, "收缩后稳定风险"])
            right_value = float(subset.loc[right, "收缩后稳定风险"])
            pair_rows.append(
                {
                    "EC编号": ec["id"],
                    "EC名称": ec["name"],
                    "车型A": left,
                    "车型B": right,
                    "风险A": left_value,
                    "风险B": right_value,
                    "风险差_A减B": left_value - right_value,
                    "绝对风险差": abs(left_value - right_value),
                    "log2风险比_A除B": math.log2((left_value + EPSILON) / (right_value + EPSILON)),
                }
            )
    pd.DataFrame(pair_rows).sort_values("绝对风险差", ascending=False).to_csv(
        STEP_DIR / "同一EC车型两两风险差异.csv", index=False, encoding="utf-8-sig"
    )
    audit = {
        "状态": "通过",
        "生成时间UTC": datetime.now(timezone.utc).isoformat(),
        "品牌范围": ["蔚来", "比亚迪"],
        "车型范围": model_risk["车型"].drop_duplicates().tolist(),
        "车型说明": "蔚来冻结输入无车型字段，使用蔚来总体；比亚迪保留四车型。",
        "bootstrap": {"单位": "原始评论组", "重复次数": BOOTSTRAP_REPETITIONS, "种子": BOOTSTRAP_SEED, "估计量": "未收缩CVaR80品牌差"},
        "品牌差异EC数": len(brand_difference),
        "车型风险行数": len(model_risk),
        "车型两两比较行数": len(pair_rows),
        "未使用变量": int(len(mapped_ids) - len(mapped_ids.drop_duplicates())),
    }
    if len(brand_difference) != 32 or len(model_risk) != 160 or len(pair_rows) != 320:
        audit["状态"] = "失败"
    with (STEP_DIR / "品牌车型差异分析审计.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    if audit["状态"] != "通过":
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
