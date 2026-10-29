from __future__ import annotations

import json
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
KG_PATH = PROJECT_DIR.parent / "数据" / "03_工程知识图谱" / "冻结工程特性本体与符号约束图谱.xlsx"
CVaR_ALPHA = 0.80
MIN_STABLE_TAIL = 20
SHRINKAGE_KAPPA = 20.0
SCOPES = {"全部样本总体": None, "蔚来品牌": "蔚来", "比亚迪品牌": "比亚迪"}


def aggregate_scope(mapped: pd.DataFrame, ecs: list[dict], scope: str, brand: str | None):
    subset_scope = mapped if brand is None else mapped[mapped["品牌归一"] == brand]
    parent_risk = {}
    for domain in list(dict.fromkeys(ec["parent"] for ec in ecs)):
        values = subset_scope.loc[subset_scope["正式一级归属"] == domain, "分摊初始势能"].to_numpy(dtype=float)
        _, stable, _, stable_k = stable_cvar(values, CVaR_ALPHA, MIN_STABLE_TAIL)
        parent_risk[domain] = {"risk": stable, "tail": stable_k, "rows": len(values)}
    rows = []
    for ec in ecs:
        subset = subset_scope[subset_scope["EC编号"] == ec["id"]]
        values = subset["分摊初始势能"].to_numpy(dtype=float)
        strict, stable, strict_k, stable_k = stable_cvar(values, CVaR_ALPHA, MIN_STABLE_TAIL)
        unique_comments = int(subset["Original_Index"].nunique())
        weight = unique_comments / (unique_comments + SHRINKAGE_KAPPA)
        parent_value = parent_risk[ec["parent"]]["risk"]
        shrunk = weight * stable + (1.0 - weight) * parent_value
        rows.append(
            {
                "分析范围": scope,
                "品牌": brand or "全部",
                "EC编号": ec["id"],
                "EC名称": ec["name"],
                "EC节点": ec["node"],
                "正式一级归属": ec["parent"],
                "映射行数": len(subset),
                "独立片段数": int(subset["片段行号"].nunique()),
                "独立原始评论数": unique_comments,
                "正势能行数": int((values > 0).sum()),
                "势能总和": float(values.sum()) if len(values) else 0.0,
                "势能均值": float(values.mean()) if len(values) else 0.0,
                "CVaR80_严格": strict,
                "CVaR80_稳定": stable,
                "严格尾部数": strict_k,
                "稳定尾部数": stable_k,
                "收缩权重": weight,
                "一级域稳定风险": parent_value,
                "收缩后稳定风险": shrunk,
                "证据等级": "充分" if unique_comments >= 50 else "一般" if unique_comments >= 20 else "不足",
            }
        )
    ec_frame = pd.DataFrame(rows)
    l1_frame = pd.DataFrame(
        [
            {
                "分析范围": scope,
                "品牌": brand or "全部",
                "一级技术域": domain,
                "稳定风险": values["risk"],
                "映射行数": values["rows"],
                "稳定尾部数": values["tail"],
            }
            for domain, values in parent_risk.items()
        ]
    )
    return ec_frame, l1_frame


def main() -> None:
    mapped = pd.read_parquet(MAPPING)
    _, ecs, _, _ = load_ontology(KG_PATH)
    ec_frames, l1_frames = [], []
    for scope, brand in SCOPES.items():
        ec, l1 = aggregate_scope(mapped, ecs, scope, brand)
        ec_frames.append(ec)
        l1_frames.append(l1)
    ec_risk = pd.concat(ec_frames, ignore_index=True)
    l1_risk = pd.concat(l1_frames, ignore_index=True)
    ec_risk.to_csv(STEP_DIR / "EC_CVaR80风险_总体与品牌.csv", index=False, encoding="utf-8-sig")
    l1_risk.to_csv(STEP_DIR / "一级技术域风险_总体与品牌.csv", index=False, encoding="utf-8-sig")
    for scope in SCOPES:
        safe = scope.replace("品牌", "")
        ec_risk[ec_risk["分析范围"] == scope].to_csv(STEP_DIR / f"EC_CVaR80风险_{safe}.csv", index=False, encoding="utf-8-sig")
    checks = {
        "范围数": ec_risk["分析范围"].nunique(),
        "每范围EC数": ec_risk.groupby("分析范围")["EC编号"].nunique().to_dict(),
        "风险非负": bool((ec_risk["收缩后稳定风险"] >= 0).all()),
        "有限值": bool(np.isfinite(ec_risk.select_dtypes(include=[np.number]).to_numpy()).all()),
    }
    if checks["范围数"] != 3 or set(checks["每范围EC数"].values()) != {32} or not checks["风险非负"] or not checks["有限值"]:
        raise RuntimeError(json.dumps(checks, ensure_ascii=False, indent=2))
    audit = {
        "状态": "通过",
        "生成时间UTC": datetime.now(timezone.utc).isoformat(),
        "参数": {"CVaR_alpha": CVaR_ALPHA, "最小稳定尾部": MIN_STABLE_TAIL, "收缩kappa": SHRINKAGE_KAPPA},
        "范围": list(SCOPES),
        "检查": checks,
    }
    with (STEP_DIR / "CVaR聚合审计.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
