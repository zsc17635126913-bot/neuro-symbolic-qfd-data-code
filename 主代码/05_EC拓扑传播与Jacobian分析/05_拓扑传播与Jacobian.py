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

from qfd_core import build_topology, load_ontology  # noqa: E402


RISK_INPUT = PROJECT_DIR / "04_风险势能与CVaR聚合" / "EC_CVaR80风险_总体与品牌.csv"
KG_PATH = PROJECT_DIR.parent / "数据" / "03_工程知识图谱" / "冻结工程特性本体与符号约束图谱.xlsx"
DIFFUSION_GAMMA = 0.30


def main() -> None:
    risk = pd.read_csv(RISK_INPUT)
    _, ecs, _, _ = load_ontology(KG_PATH)
    l1_order, parent, cross, laplacian, jacobian, cross_audit = build_topology(ecs, DIFFUSION_GAMMA)
    nodes = [ec["node"] for ec in ecs]
    ec_ids = [ec["id"] for ec in ecs]
    phi = np.linalg.inv(np.eye(len(ecs)) + DIFFUSION_GAMMA * laplacian)
    topology_utility = jacobian.sum(axis=1)

    pd.DataFrame(parent, index=nodes, columns=l1_order).reset_index(names="EC节点").to_csv(
        STEP_DIR / "P_parent_唯一正式归属.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(cross, index=nodes, columns=l1_order).reset_index(names="EC节点").to_csv(
        STEP_DIR / "C_cross_工程跨域影响.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(laplacian, index=nodes, columns=nodes).reset_index(names="EC节点").to_csv(
        STEP_DIR / "L_sym_归一化拉普拉斯.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(jacobian, index=nodes, columns=l1_order).reset_index(names="EC节点").to_csv(
        STEP_DIR / "Jacobian敏感矩阵_32x7.csv", index=False, encoding="utf-8-sig"
    )
    cross_audit.to_csv(STEP_DIR / "跨域关系审计.csv", index=False, encoding="utf-8-sig")

    rankings = []
    for scope, group in risk.groupby("分析范围", sort=False):
        ordered = group.set_index("EC编号").loc[ec_ids]
        pure = ordered["收缩后稳定风险"].to_numpy(dtype=float)
        propagated = phi @ pure
        global_raw = pure * topology_utility
        result = pd.DataFrame(
            {
                "分析范围": scope,
                "品牌": ordered["品牌"].to_numpy(),
                "EC编号": ec_ids,
                "EC名称": [ec["name"] for ec in ecs],
                "EC节点": nodes,
                "正式一级归属": [ec["parent"] for ec in ecs],
                "纯风险值": pure,
                "拓扑传播风险": propagated,
                "拓扑边际效用": topology_utility,
                "全局边际效用_原值": global_raw,
                "全局边际效用": global_raw / global_raw.max() if global_raw.max() > 0 else 0.0,
            }
        )
        result["纯风险排名"] = result["纯风险值"].rank(method="min", ascending=False).astype(int)
        result["传播风险排名"] = result["拓扑传播风险"].rank(method="min", ascending=False).astype(int)
        result = result.sort_values(["全局边际效用", "EC编号"], ascending=[False, True]).reset_index(drop=True)
        result.insert(0, "拓扑调整排名", np.arange(1, len(result) + 1))
        rankings.append(result)
        safe = scope.replace("品牌", "")
        result.to_csv(STEP_DIR / f"EC纯风险与拓扑调整排序_{safe}.csv", index=False, encoding="utf-8-sig")
    combined = pd.concat(rankings, ignore_index=True)
    combined.to_csv(STEP_DIR / "EC纯风险与拓扑调整排序_总体与品牌.csv", index=False, encoding="utf-8-sig")
    audit = {
        "状态": "通过",
        "生成时间UTC": datetime.now(timezone.utc).isoformat(),
        "参数": {"拓扑传播强度": DIFFUSION_GAMMA},
        "EC数": len(ecs),
        "一级技术域数": len(l1_order),
        "父归属非零数": int(np.count_nonzero(parent)),
        "跨域关系非零数": int(np.count_nonzero(cross)),
        "Jacobian有限": bool(np.isfinite(jacobian).all()),
        "范围排名数": combined["分析范围"].nunique(),
        "每范围EC数": combined.groupby("分析范围")["EC编号"].nunique().to_dict(),
    }
    if not audit["Jacobian有限"] or audit["每范围EC数"] != {"全部样本总体": 32, "蔚来品牌": 32, "比亚迪品牌": 32}:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    with (STEP_DIR / "拓扑与Jacobian审计.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
