from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "主代码"
OUT = PROJECT / "11_外部验证"
INTERNAL = PROJECT / "05_EC拓扑传播与Jacobian分析" / "EC纯风险与拓扑调整排序_总体与品牌.csv"
VEHICLE_QUALITY = ROOT / "数据" / "05_外部验证" / "车质网_蔚来EC投诉频次与冻结排序.csv"
DONGCHEDI = ROOT / "数据" / "05_外部验证" / "懂车帝_蔚来EC纯风险与拓扑调整排序.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ec_from_node(value: str) -> str:
    match = re.match(r"(EC\d{2})", str(value))
    if not match:
        raise ValueError(f"无法从 EC 节点解析编号: {value!r}")
    return match.group(1)


def top_overlap(frame: pd.DataFrame, external_rank: str, k: int) -> dict[str, float | int | str]:
    internal_set = set(frame.nsmallest(k, "内部拓扑调整排名")["EC编号"])
    external_set = set(frame.nsmallest(k, external_rank)["EC编号"])
    intersection = sorted(internal_set & external_set)
    union = internal_set | external_set
    return {
        "TopK": k,
        "交集数量": len(intersection),
        "重叠率_交集除以K": len(intersection) / k,
        "Jaccard": len(intersection) / len(union),
        "交集EC": ";".join(intersection),
    }


def evaluate(source_name: str, external: pd.DataFrame, external_rank: str) -> tuple[pd.DataFrame, dict, list[dict]]:
    internal = pd.read_csv(INTERNAL)
    internal = internal.loc[
        internal["分析范围"].eq("蔚来品牌"),
        ["EC编号", "EC名称", "拓扑调整排名", "全局边际效用"],
    ].rename(
        columns={
            "拓扑调整排名": "内部拓扑调整排名",
            "全局边际效用": "内部归一化全局边际效用",
        }
    )
    merged = internal.merge(external, on="EC编号", how="inner", validate="one_to_one")
    if len(merged) != 32:
        raise AssertionError(f"{source_name}: 预期 32 个共同 EC，实际 {len(merged)}")

    rho = spearmanr(merged["内部拓扑调整排名"], merged[external_rank])
    tau = kendalltau(merged["内部拓扑调整排名"], merged[external_rank], variant="b")
    merged["排名差_内部减外部"] = merged["内部拓扑调整排名"] - merged[external_rank]
    merged["排名差绝对值"] = merged["排名差_内部减外部"].abs()
    merged.insert(0, "外部来源", source_name)
    merged = merged.sort_values("内部拓扑调整排名").reset_index(drop=True)

    overlaps = [top_overlap(merged, external_rank, k) for k in (5, 10)]
    summary = {
        "外部来源": source_name,
        "共同EC数": len(merged),
        "Spearman_rho": float(rho.statistic),
        "Spearman_p": float(rho.pvalue),
        "Kendall_tau_b": float(tau.statistic),
        "Kendall_p": float(tau.pvalue),
        "平均排名绝对差": float(merged["排名差绝对值"].mean()),
        "中位排名绝对差": float(merged["排名差绝对值"].median()),
    }
    for item in overlaps:
        k = item["TopK"]
        summary[f"Top{k}_交集数量"] = item["交集数量"]
        summary[f"Top{k}_重叠率"] = item["重叠率_交集除以K"]
        summary[f"Top{k}_Jaccard"] = item["Jaccard"]
        summary[f"Top{k}_交集EC"] = item["交集EC"]
        item["外部来源"] = source_name
    return merged, summary, overlaps


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    quality = pd.read_csv(VEHICLE_QUALITY)[
        ["EC编号", "EC节点", "外部投诉频次", "外部投诉占全部投诉比例", "外部频次排名"]
    ].rename(columns={"EC节点": "外部EC节点"})
    quality_detail, quality_summary, quality_overlap = evaluate(
        "车质网_蔚来全部投诉", quality, "外部频次排名"
    )

    dcd = pd.read_csv(DONGCHEDI)
    dcd = dcd.loc[dcd["范围"].eq("总体")].copy()
    dcd["EC编号"] = dcd["EC节点"].map(ec_from_node)
    dcd = dcd[
        ["EC编号", "EC节点", "纯风险值", "拓扑调整排名", "全局边际效用"]
    ].rename(
        columns={
            "EC节点": "外部EC节点",
            "纯风险值": "外部纯风险值",
            "拓扑调整排名": "外部拓扑调整排名",
            "全局边际效用": "外部归一化全局边际效用",
        }
    )
    dcd_detail, dcd_summary, dcd_overlap = evaluate(
        "懂车帝_蔚来冻结样本", dcd, "外部拓扑调整排名"
    )

    quality_detail.to_csv(OUT / "蔚来内部与车质网逐EC排名比较.csv", index=False, encoding="utf-8-sig")
    dcd_detail.to_csv(OUT / "蔚来内部与懂车帝逐EC排名比较.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([quality_summary, dcd_summary])
    summary.to_csv(OUT / "蔚来外部验证统计汇总.csv", index=False, encoding="utf-8-sig")
    overlaps = pd.DataFrame(quality_overlap + dcd_overlap)
    overlaps = overlaps[["外部来源", "TopK", "交集数量", "重叠率_交集除以K", "Jaccard", "交集EC"]]
    overlaps.to_csv(OUT / "蔚来外部验证TopK重叠.csv", index=False, encoding="utf-8-sig")

    audit = {
        "analysis_scope": "NIO only",
        "scope_boundary": "冻结资料中没有独立的比亚迪外部平台数据，因此本步骤不能验证比亚迪结果。",
        "interpretation_boundary": "本步骤评估跨来源 EC 排名一致性，不是逐评论预测精度或因果验证。",
        "internal_ranking": "蔚来品牌拓扑调整排名",
        "sources": [
            {"role": "internal", "path": str(INTERNAL), "sha256": sha256(INTERNAL)},
            {"role": "external_vehicle_quality", "path": str(VEHICLE_QUALITY), "sha256": sha256(VEHICLE_QUALITY)},
            {"role": "external_dongchedi", "path": str(DONGCHEDI), "sha256": sha256(DONGCHEDI)},
        ],
        "checks": {
            "nio_internal_ec_count": 32,
            "vehicle_quality_common_ec_count": int(quality_summary["共同EC数"]),
            "dongchedi_common_ec_count": int(dcd_summary["共同EC数"]),
            "all_statistics_finite": bool(np.isfinite(summary.select_dtypes("number").to_numpy()).all()),
        },
        "summary": [quality_summary, dcd_summary],
    }
    (OUT / "外部验证审计.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
