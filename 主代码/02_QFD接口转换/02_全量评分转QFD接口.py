from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STEP_DIR.parent
SCORE_CSV = PROJECT_DIR.parent / "数据" / "01_评论片段与冻结评分" / "双品牌评论片段_全量情感效价与工程严重度冻结评分.csv"
OUTPUT_PARQUET = STEP_DIR / "全量QFD统一接口.parquet"
EXPECTED_ROWS = 525_647


def sequence_sha256(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def main() -> None:
    if not SCORE_CSV.is_file():
        raise FileNotFoundError(SCORE_CSV)
    frame = pd.read_csv(SCORE_CSV, encoding="utf-8-sig", low_memory=False)
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"行数锁失败：{len(frame)} != {EXPECTED_ROWS}")
    required = {
        "打分原始行号", "品牌", "record_id", "原始评论分组ID", "source_file", "field_name",
        "Filtered_SubSeq", "Entity", "Aspect", "预测情感类别", "预测情感效价",
        "预测严重度_0到4", "P_严重度0", "P_严重度1", "P_严重度2", "P_严重度3", "P_严重度4",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"缺少字段：{sorted(missing)}")

    result = pd.DataFrame()
    result["片段行号"] = frame["打分原始行号"].astype(np.int64)
    result["Original_Index"] = frame["原始评论分组ID"].astype(str)
    result["record_id"] = frame["record_id"].astype(str)
    result["品牌原字段"] = frame["品牌"].astype(str)
    result["品牌归一"] = np.where(frame["品牌"].astype(str).eq("蔚来"), "蔚来", "比亚迪")
    result["车型"] = np.where(frame["品牌"].astype(str).eq("蔚来"), "蔚来总体", frame["品牌"].astype(str))
    result["source_file"] = frame["source_file"].astype(str)
    result["field_name"] = frame["field_name"].astype(str)
    result["Filtered_SubSeq"] = frame["Filtered_SubSeq"].fillna("").astype(str).str.strip()
    result["Entity"] = frame["Entity"].fillna("").astype(str)
    result["Aspect"] = frame["Aspect"].fillna("").astype(str)
    result["Modifier"] = ""
    result["S6_情感类别"] = frame["预测情感类别"].astype(str)
    result["S6_效价值"] = frame["预测情感效价"].astype(np.float32)
    result["严重度等级_最终"] = frame["预测严重度_0到4"].astype(np.int8)
    probabilities = frame[[f"P_严重度{i}" for i in range(5)]].to_numpy(dtype=np.float64)
    expected_severity = probabilities @ np.arange(5, dtype=np.float64)
    result["严重度连续分值_0_4"] = expected_severity.astype(np.float32)
    for column in ("P_情感_负向", "P_情感_中性", "P_情感_正向"):
        result[column] = frame[column].astype(np.float32)
    for i in range(5):
        result[f"P_严重度{i}"] = frame[f"P_严重度{i}"].astype(np.float32)

    expected_ids = np.arange(EXPECTED_ROWS, dtype=np.int64)
    checks = {
        "行数": len(result),
        "片段行号唯一": bool(result["片段行号"].is_unique),
        "片段行号连续": bool(np.array_equal(result["片段行号"].to_numpy(), expected_ids)),
        "空文本数": int(result["Filtered_SubSeq"].eq("").sum()),
        "品牌分布": {str(k): int(v) for k, v in result["品牌归一"].value_counts().items()},
        "车型分布": {str(k): int(v) for k, v in result["车型"].value_counts().items()},
        "严重度期望最小值": float(result["严重度连续分值_0_4"].min()),
        "严重度期望最大值": float(result["严重度连续分值_0_4"].max()),
        "有序文本SHA256": sequence_sha256(result["Filtered_SubSeq"]),
        "有序主IDSHA256": sequence_sha256(result["片段行号"].astype(str)),
    }
    if not checks["片段行号唯一"] or not checks["片段行号连续"] or checks["空文本数"]:
        raise ValueError(json.dumps(checks, ensure_ascii=False, indent=2))
    if not result["严重度连续分值_0_4"].between(0, 4).all():
        raise ValueError("严重度概率期望越界")

    result.to_parquet(OUTPUT_PARQUET, index=False, compression="zstd")
    audit = {
        "状态": "通过",
        "生成时间UTC": datetime.now(timezone.utc).isoformat(),
        "输入": str(SCORE_CSV.relative_to(PROJECT_DIR)),
        "输出": OUTPUT_PARQUET.name,
        "输出字节": OUTPUT_PARQUET.stat().st_size,
        "字段": result.columns.tolist(),
        "检查": checks,
        "品牌口径": "原字段为蔚来时归为蔚来，其余四个车型值归为比亚迪。",
        "车型口径": "比亚迪保留四车型；冻结输入未提供蔚来车型字段，因此统一标记为蔚来总体。",
        "严重度连续分值": "sum(k * P_严重度k), k=0..4",
    }
    with (STEP_DIR / "QFD接口转换审计.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    pd.DataFrame(
        [
            {"目标字段": "片段行号", "来源/计算": "打分原始行号"},
            {"目标字段": "Original_Index", "来源/计算": "原始评论分组ID"},
            {"目标字段": "品牌归一", "来源/计算": "蔚来/比亚迪归一规则"},
            {"目标字段": "车型", "来源/计算": "比亚迪车型原值；蔚来总体"},
            {"目标字段": "S6_情感类别", "来源/计算": "预测情感类别"},
            {"目标字段": "S6_效价值", "来源/计算": "预测情感效价"},
            {"目标字段": "严重度等级_最终", "来源/计算": "预测严重度_0到4"},
            {"目标字段": "严重度连续分值_0_4", "来源/计算": "五类概率期望"},
        ]
    ).to_csv(STEP_DIR / "QFD接口字段映射.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
