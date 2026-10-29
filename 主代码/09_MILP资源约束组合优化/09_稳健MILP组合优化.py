from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


STEP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STEP_DIR.parent
CANDIDATES = PROJECT_DIR / "08_TRIZ工程方案生成" / "当前Top10_TRIZ工程候选.csv"
LIMITS = {
    "bom_cost": 1000.0,
    "development_validation_cost": 2_000_000.0,
    "person_months": 20.0,
    "delta_mass": 3.5,
    "delta_volume": 2.0,
    "delta_power": 100.0,
}
SAFETY_ECS = {"EC06", "EC25", "EC26"}


def parse_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value) or str(value).strip() == "":
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def constraints(frame: pd.DataFrame, scenario: str) -> LinearConstraint:
    n = len(frame)
    rows, lower, upper = [], [], []
    for field, limit in LIMITS.items():
        rows.append(frame[field].to_numpy(dtype=float))
        lower.append(-np.inf)
        upper.append(limit)
    for ec_id in frame["ec_id"].unique():
        rows.append((frame["ec_id"] == ec_id).astype(float).to_numpy())
        lower.append(0.0)
        upper.append(1.0)
    id_to_index = {solution_id: index for index, solution_id in enumerate(frame["solution_id"])}
    for index, row in frame.iterrows():
        for dependency in row["dependencies_parsed"]:
            if dependency in id_to_index:
                vector = np.zeros(n); vector[index] = 1; vector[id_to_index[dependency]] = -1
                rows.append(vector); lower.append(-np.inf); upper.append(0.0)
        for conflict in row["conflicts_parsed"]:
            if conflict in id_to_index and index < id_to_index[conflict]:
                vector = np.zeros(n); vector[index] = 1; vector[id_to_index[conflict]] = 1
                rows.append(vector); lower.append(-np.inf); upper.append(1.0)
    if scenario == "balanced":
        rows.append(np.ones(n)); lower.append(2.0); upper.append(np.inf)
    if scenario == "safety_first":
        rows.append(frame["ec_id"].isin(SAFETY_ECS).astype(float).to_numpy())
        lower.append(1.0); upper.append(np.inf)
    matrix = np.vstack(rows)
    return LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))


def solve(frame: pd.DataFrame, scenario: str) -> tuple[dict, pd.DataFrame]:
    normalized_cost = (
        frame["bom_cost"] / LIMITS["bom_cost"]
        + frame["development_validation_cost"] / LIMITS["development_validation_cost"]
        + frame["person_months"] / LIMITS["person_months"]
        + frame["delta_mass"] / LIMITS["delta_mass"]
        + frame["delta_power"] / LIMITS["delta_power"]
    )
    if scenario == "maximum_robust_benefit":
        objective = -frame["robust_benefit"].to_numpy(dtype=float)
    elif scenario == "balanced":
        objective = -(frame["robust_benefit"] - 0.08 * normalized_cost).to_numpy(dtype=float)
    elif scenario == "safety_first":
        safety_bonus = frame["ec_id"].isin(SAFETY_ECS).astype(float) * 0.05
        objective = -(frame["robust_benefit"] + safety_bonus - 0.03 * normalized_cost).to_numpy(dtype=float)
    else:
        raise ValueError(scenario)
    result = milp(
        c=objective,
        integrality=np.ones(len(frame)),
        bounds=Bounds(np.zeros(len(frame)), np.ones(len(frame))),
        constraints=constraints(frame, scenario),
        options={"time_limit": 60.0},
    )
    selected = frame.iloc[np.flatnonzero(result.x > 0.5)].copy() if result.x is not None else frame.iloc[[]].copy()
    summary = {
        "情景": scenario,
        "成功": bool(result.success),
        "状态": int(result.status),
        "消息": str(result.message),
        "目标值": float(result.fun) if result.fun is not None else None,
        "入选方案": selected["solution_id"].tolist(),
        "覆盖EC": selected["ec_id"].tolist(),
        "稳健效益总和": float(selected["robust_benefit"].sum()),
        "资源": {field: float(selected[field].sum()) for field in LIMITS},
        "最长工期月": float(selected["duration_months"].max()) if len(selected) else 0.0,
    }
    selected.insert(0, "优化情景", scenario)
    return summary, selected


def main() -> None:
    frame = pd.read_csv(CANDIDATES)
    frame["dependencies_parsed"] = frame["dependencies"].map(parse_list)
    frame["conflicts_parsed"] = frame["conflicts"].map(parse_list)
    numeric = [*LIMITS, "duration_months", "robust_benefit", "reduction_lower", "feasibility_score", "combined_confidence"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    eligible = frame[
        frame["review_status"].isin(["machine_passed_engineering_pending", "provisional_engineering_review_required"])
        & ~frame["safety_block"].astype(bool)
        & (frame["relevance_score"] >= 0.80)
        & (frame["mechanism_score"] >= 0.70)
        & (frame["feasibility_score"] >= 0.60)
        & (frame["resource_plausibility_score"] >= 0.60)
    ].reset_index(drop=True)
    eligible.drop(columns=["dependencies_parsed", "conflicts_parsed"]).to_csv(
        STEP_DIR / "MILP可用工程候选.csv", index=False, encoding="utf-8-sig"
    )
    summaries, selected_frames = [], []
    for scenario in ("maximum_robust_benefit", "balanced", "safety_first"):
        summary, selected = solve(eligible, scenario)
        summaries.append(summary); selected_frames.append(selected)
    selected = pd.concat(selected_frames, ignore_index=True)
    selected.drop(columns=["dependencies_parsed", "conflicts_parsed"]).to_csv(
        STEP_DIR / "MILP三情景入选方案.csv", index=False, encoding="utf-8-sig"
    )
    with (STEP_DIR / "稳健MILP三情景组合.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
    audit = {
        "状态": "通过" if all(item["成功"] for item in summaries) else "失败",
        "生成时间UTC": datetime.now(timezone.utc).isoformat(),
        "候选数": len(frame),
        "可用候选数": len(eligible),
        "资源上限": LIMITS,
        "资源口径": "可复现实验情景，不代表企业已批准预算。",
        "情景数": len(summaries),
        "成功情景数": sum(item["成功"] for item in summaries),
        "工程专家验证仍必需": True,
    }
    with (STEP_DIR / "MILP优化审计.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    if audit["状态"] != "通过":
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({"审计": audit, "结果": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
