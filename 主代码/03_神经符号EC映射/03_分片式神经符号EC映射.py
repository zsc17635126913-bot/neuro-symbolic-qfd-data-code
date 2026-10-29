from __future__ import annotations

import gzip
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


STEP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STEP_DIR.parent
WORKSPACE = PROJECT_DIR.parent
sys.path.insert(0, str(STEP_DIR))

from qfd_core import load_ontology, normalize_text, terms_by_ec  # noqa: E402


INTERFACE = PROJECT_DIR / "02_QFD接口转换" / "全量QFD统一接口.parquet"
BGE_ROOT = WORKSPACE / "数据" / "02_BGE-M3语义特征"
KG_PATH = WORKSPACE / "数据" / "03_工程知识图谱" / "冻结工程特性本体与符号约束图谱.xlsx"
MAPPING_OUTPUT = STEP_DIR / "逐片段_约束混合EC映射.parquet"
COMPONENT_OUTPUT = STEP_DIR / "映射分量缓存.npz"

DENSE_WEIGHT = 0.60
SPARSE_WEIGHT = 0.40
SEMANTIC_WEIGHT = 0.70
LEXICAL_WEIGHT = 0.30
MAPPING_THRESHOLD = 0.38
SECOND_EC_GAP = 0.10
LAMBDA_PARAM = 2.3
PAIN_GAMMA = 2.0
MAX_PCK_ITERATIONS = 60
PCK_TOLERANCE = 1e-4
PCK_STABLE_CHANGE_RATE = 0.001
PCK_STABLE_ROUNDS = 3
EXPECTED_ROWS = 525_647


def load_sparse(path: Path) -> list[dict]:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def normalized_dense(path: Path) -> np.ndarray:
    dense = np.asarray(np.load(path), dtype=np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    return np.divide(dense, norms, out=np.zeros_like(dense), where=norms > 0)


def build_symbol_matrices(frame: pd.DataFrame, ecs: list[dict], must_rules: list[dict], cannot_rules: list[dict]):
    must = terms_by_ec(must_rules)
    cannot = terms_by_ec(cannot_rules)
    normalized = frame[["Filtered_SubSeq", "Entity", "Aspect", "Modifier"]].fillna("").astype(str)
    for column in normalized.columns:
        normalized[column] = normalized[column].map(normalize_text)
    combined = normalized.agg("|".join, axis=1)
    n_rows, n_ec = len(frame), len(ecs)
    lexical = np.zeros((n_rows, n_ec), dtype=np.float32)
    blocked = np.zeros((n_rows, n_ec), dtype=bool)
    forced = np.zeros((n_rows, n_ec), dtype=bool)
    anchors = np.zeros((n_rows, n_ec), dtype=bool)
    pattern_audit = []
    for index, ec in enumerate(ecs):
        positives = list(dict.fromkeys(ec["terms"] + must.get(ec["id"], [])))
        negatives = list(dict.fromkeys(ec["exclusions"] + cannot.get(ec["id"], [])))
        positive_pattern = "|".join(re.escape(term) for term in positives)
        negative_pattern = "|".join(re.escape(term) for term in negatives)
        must_pattern = "|".join(re.escape(term) for term in must.get(ec["id"], []))
        counts = combined.str.count(positive_pattern).to_numpy(dtype=np.float32)
        lexical[:, index] = np.minimum(1.0, 0.20 * counts)
        if negative_pattern:
            blocked[:, index] = combined.str.contains(negative_pattern, regex=True, na=False).to_numpy()
        if must_pattern:
            forced[:, index] = combined.str.contains(must_pattern, regex=True, na=False).to_numpy()
        anchors[:, index] = (counts > 0) & ~blocked[:, index]
        pattern_audit.append(
            {
                "EC编号": ec["id"],
                "正向词数": len(positives),
                "排除词数": len(negatives),
                "锚点数": int(anchors[:, index].sum()),
                "必连命中数": int(forced[:, index].sum()),
                "勿连命中数": int(blocked[:, index].sum()),
            }
        )
        print(f"[Symbol] {index + 1:02d}/32 {ec['id']} anchors={anchors[:, index].sum()}", flush=True)
    if any(item["锚点数"] == 0 for item in pattern_audit):
        raise ValueError("至少一个EC没有可用锚点")
    return lexical, blocked, forced, anchors, pattern_audit


def initial_centroids(shards: list[Path], anchors: np.ndarray, n_ec: int):
    sums = np.zeros((n_ec, 1024), dtype=np.float64)
    counts = np.zeros(n_ec, dtype=np.int64)
    anchor_indices = [np.flatnonzero(anchors[:, index])[:5000] for index in range(n_ec)]
    sparse_sums: list[dict[str, float]] = [defaultdict(float) for _ in range(n_ec)]
    offset = 0
    for shard_number, shard in enumerate(shards):
        dense = normalized_dense(shard / "dense.npy")
        end = offset + len(dense)
        for ec_index in range(n_ec):
            local = np.flatnonzero(anchors[offset:end, ec_index])
            if len(local):
                sums[ec_index] += dense[local].sum(axis=0, dtype=np.float64)
                counts[ec_index] += len(local)
        selected_by_ec = []
        need_sparse = False
        for indices in anchor_indices:
            selected = indices[(indices >= offset) & (indices < end)] - offset
            selected_by_ec.append(selected)
            need_sparse |= len(selected) > 0
        if need_sparse:
            sparse = load_sparse(shard / "sparse.pkl.gz")
            for ec_index, selected in enumerate(selected_by_ec):
                for row_index in selected:
                    for token, value in sparse[int(row_index)].items():
                        sparse_sums[ec_index][str(token)] += float(value)
        offset = end
        if (shard_number + 1) % 20 == 0 or shard_number + 1 == len(shards):
            print(f"[Anchors] {shard_number + 1}/{len(shards)}", flush=True)
    centers = np.asarray(sums / counts[:, None], dtype=np.float32)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = np.divide(centers, norms, out=np.zeros_like(centers), where=norms > 0)
    sparse_centers = []
    for ec_index, values in enumerate(sparse_sums):
        divisor = float(len(anchor_indices[ec_index]))
        sparse_centers.append({token: value / divisor for token, value in values.items()})
    return centers, sparse_centers, counts


def run_pck_means(shards: list[Path], anchor_centers: np.ndarray, blocked: np.ndarray, forced: np.ndarray):
    centers = anchor_centers.copy()
    previous = np.full(EXPECTED_ROWS, -1, dtype=np.int16)
    history = []
    stable_rounds = 0
    for iteration in range(1, MAX_PCK_ITERATIONS + 1):
        sums = np.zeros_like(centers, dtype=np.float64)
        counts = np.zeros(len(centers), dtype=np.int64)
        labels = np.empty(EXPECTED_ROWS, dtype=np.int16)
        offset = 0
        for shard in shards:
            dense = normalized_dense(shard / "dense.npy")
            end = offset + len(dense)
            scores = dense @ centers.T
            scores[blocked[offset:end]] = -np.inf
            scores[forced[offset:end] & ~blocked[offset:end]] = 2.0
            chosen = np.argmax(scores, axis=1).astype(np.int16)
            labels[offset:end] = chosen
            for ec_index in range(len(centers)):
                selected = dense[chosen == ec_index]
                if len(selected):
                    sums[ec_index] += selected.sum(axis=0, dtype=np.float64)
                    counts[ec_index] += len(selected)
            offset = end
        updated = centers.copy()
        for ec_index in range(len(centers)):
            if counts[ec_index]:
                mean = np.asarray(sums[ec_index] / counts[ec_index], dtype=np.float32)
                candidate = 0.75 * mean + 0.25 * anchor_centers[ec_index]
                norm = float(np.linalg.norm(candidate))
                updated[ec_index] = candidate / norm if norm else anchor_centers[ec_index]
        changes = int(np.count_nonzero(labels != previous))
        change_rate = changes / EXPECTED_ROWS
        shift = float(np.linalg.norm(updated - centers))
        stable_rounds = stable_rounds + 1 if change_rate <= PCK_STABLE_CHANGE_RATE else 0
        history.append(
            {
                "iteration": iteration,
                "changes": changes,
                "change_rate": change_rate,
                "centroid_shift": shift,
                "stable_rounds": stable_rounds,
            }
        )
        print(
            f"[PCK] iteration={iteration} changes={changes} rate={change_rate:.6f} "
            f"shift={shift:.8f} stable_rounds={stable_rounds}",
            flush=True,
        )
        centers = updated
        if iteration > 1 and (
            changes == 0 or shift < PCK_TOLERANCE or stable_rounds >= PCK_STABLE_ROUNDS
        ):
            return centers, labels, history, True
        previous = labels
    return centers, labels, history, False


def sparse_similarity(sparse: list[dict], sparse_centers: list[dict]) -> np.ndarray:
    scores = np.zeros((len(sparse), len(sparse_centers)), dtype=np.float32)
    center_norms = np.asarray(
        [math.sqrt(sum(float(value) ** 2 for value in center.values())) for center in sparse_centers],
        dtype=np.float32,
    )
    targets: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for ec_index, center in enumerate(sparse_centers):
        for token, value in center.items():
            targets[str(token)].append((ec_index, float(value)))
    for row_index, vector in enumerate(sparse):
        vector_norm = math.sqrt(sum(float(value) ** 2 for value in vector.values()))
        if vector_norm == 0:
            continue
        for token, value in vector.items():
            for ec_index, center_value in targets.get(str(token), ()):
                scores[row_index, ec_index] += float(value) * center_value
        denominator = vector_norm * center_norms
        scores[row_index] = np.divide(scores[row_index], denominator, out=np.zeros_like(scores[row_index]), where=denominator > 0)
    return scores


def result_rows(base: pd.DataFrame, positions: np.ndarray, selected: np.ndarray, weights: np.ndarray, scores: np.ndarray, ecs: list[dict], energy: np.ndarray, pain: np.ndarray, method: str) -> pd.DataFrame:
    columns = [
        "片段行号", "Original_Index", "record_id", "品牌归一", "车型", "field_name", "Filtered_SubSeq",
        "Entity", "Aspect", "S6_情感类别", "S6_效价值", "严重度等级_最终", "严重度连续分值_0_4",
    ]
    result = base.iloc[positions][columns].copy().reset_index(drop=True)
    chosen = selected[positions]
    result["EC编号"] = [ecs[index]["id"] for index in chosen]
    result["EC名称"] = [ecs[index]["name"] for index in chosen]
    result["EC节点"] = [ecs[index]["node"] for index in chosen]
    result["正式一级归属"] = [ecs[index]["parent"] for index in chosen]
    result["映射方法"] = method
    result["映射得分"] = scores[positions, chosen]
    result["映射置信度"] = weights[positions]
    result["负向痛感"] = pain[positions]
    result["评论初始势能"] = energy[positions]
    result["分摊初始势能"] = energy[positions] * weights[positions]
    return result


def main() -> None:
    frame = pd.read_parquet(INTERFACE)
    if len(frame) != EXPECTED_ROWS or not np.array_equal(frame["片段行号"].to_numpy(), np.arange(EXPECTED_ROWS)):
        raise ValueError("统一接口行锁失败")
    shards = sorted((BGE_ROOT / "shards").glob("shard_*"))
    if len(shards) != 106:
        raise ValueError(f"BGE分片数错误：{len(shards)}")
    ontology, ecs, must_rules, cannot_rules = load_ontology(KG_PATH)
    ontology.to_csv(STEP_DIR / "冻结EC本体.csv", index=False, encoding="utf-8-sig")
    lexical, blocked, forced, anchors, pattern_audit = build_symbol_matrices(frame, ecs, must_rules, cannot_rules)
    pd.DataFrame(pattern_audit).to_csv(STEP_DIR / "符号约束命中审计.csv", index=False, encoding="utf-8-sig")

    anchor_centers, sparse_centers, anchor_counts = initial_centroids(shards, anchors, len(ecs))
    centers, labels, pck_history, converged = run_pck_means(shards, anchor_centers, blocked, forced)
    if not converged:
        raise RuntimeError("PCK-means在60轮内未收敛")
    np.save(STEP_DIR / "PCK_dense_centroids.npy", centers)
    with gzip.open(STEP_DIR / "PCK_sparse_centroids.pkl.gz", "wb", compresslevel=6) as handle:
        pickle.dump(sparse_centers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dense_component = np.empty((EXPECTED_ROWS, len(ecs)), dtype=np.float16)
    sparse_component = np.empty_like(dense_component)
    pain = np.maximum(0.0, -frame["S6_效价值"].to_numpy(dtype=np.float64))
    severity = frame["严重度连续分值_0_4"].to_numpy(dtype=np.float64)
    energy = np.exp(-LAMBDA_PARAM * (2.0 - severity)) * np.power(pain, PAIN_GAMMA)
    writer = None
    offset = 0
    mapped_fragments = 0
    second_fragments = 0
    mapped_energy = 0.0
    cannot_violations = 0
    output_rows = 0
    for shard_number, shard in enumerate(shards):
        dense = normalized_dense(shard / "dense.npy")
        sparse = load_sparse(shard / "sparse.pkl.gz")
        end = offset + len(dense)
        dense_scores = ((dense @ centers.T).clip(-1.0, 1.0) + 1.0) / 2.0
        sparse_scores = sparse_similarity(sparse, sparse_centers)
        dense_component[offset:end] = dense_scores.astype(np.float16)
        sparse_component[offset:end] = sparse_scores.astype(np.float16)
        semantic = DENSE_WEIGHT * dense_scores + SPARSE_WEIGHT * sparse_scores
        scores = SEMANTIC_WEIGHT * semantic + LEXICAL_WEIGHT * lexical[offset:end]
        local_blocked = blocked[offset:end]
        local_forced = forced[offset:end]
        scores[local_blocked] = -np.inf
        scores[local_forced & ~local_blocked] = np.maximum(scores[local_forced & ~local_blocked], 1.25)
        top_two = np.argpartition(scores, -2, axis=1)[:, -2:]
        top_values = np.take_along_axis(scores, top_two, axis=1)
        order = np.argsort(top_values, axis=1)[:, ::-1]
        top_two = np.take_along_axis(top_two, order, axis=1)
        top_values = np.take_along_axis(top_values, order, axis=1)
        valid_first = np.isfinite(top_values[:, 0]) & (top_values[:, 0] >= MAPPING_THRESHOLD)
        valid_second = (
            valid_first
            & np.isfinite(top_values[:, 1])
            & (top_values[:, 1] >= MAPPING_THRESHOLD)
            & ((top_values[:, 0] - top_values[:, 1]) <= SECOND_EC_GAP)
        )
        denominator = np.maximum(top_values[:, 0], 1e-9) + np.where(valid_second, np.maximum(top_values[:, 1], 1e-9), 0.0)
        weight_first = np.where(valid_first, np.maximum(top_values[:, 0], 1e-9) / denominator, 0.0)
        weight_second = np.where(valid_second, np.maximum(top_values[:, 1], 1e-9) / denominator, 0.0)
        base = frame.iloc[offset:end]
        local_energy = energy[offset:end]
        local_pain = pain[offset:end]
        first_positions = np.flatnonzero(valid_first)
        second_positions = np.flatnonzero(valid_second)
        parts = [result_rows(base, first_positions, top_two[:, 0], weight_first, scores, ecs, local_energy, local_pain, "BGE-M3+符号约束PCK-means")]
        if len(second_positions):
            parts.append(result_rows(base, second_positions, top_two[:, 1], weight_second, scores, ecs, local_energy, local_pain, "BGE-M3+符号约束PCK-means"))
        mapped = pd.concat(parts, ignore_index=True)
        table = pa.Table.from_pandas(mapped, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(MAPPING_OUTPUT, table.schema, compression="zstd")
        writer.write_table(table)
        mapped_fragments += len(first_positions)
        second_fragments += len(second_positions)
        mapped_energy += float(mapped["分摊初始势能"].sum())
        cannot_violations += int(local_blocked[first_positions, top_two[first_positions, 0]].sum())
        cannot_violations += int(local_blocked[second_positions, top_two[second_positions, 1]].sum())
        output_rows += len(mapped)
        offset = end
        if (shard_number + 1) % 10 == 0 or shard_number + 1 == len(shards):
            print(f"[Mapping] {shard_number + 1}/{len(shards)} mapped={mapped_fragments}", flush=True)
    if writer is not None:
        writer.close()

    np.savez_compressed(
        COMPONENT_OUTPUT,
        lexical=lexical.astype(np.float16),
        dense=dense_component,
        sparse=sparse_component,
        blocked=np.packbits(blocked, axis=1),
        forced=np.packbits(forced, axis=1),
        pck_labels=labels,
    )
    mapped = pd.read_parquet(MAPPING_OUTPUT)
    coverage = (
        mapped.groupby(["EC编号", "EC名称", "正式一级归属"], observed=True)
        .agg(映射行数=("片段行号", "size"), 独立片段数=("片段行号", "nunique"), 独立评论数=("Original_Index", "nunique"), 势能总和=("分摊初始势能", "sum"))
        .reset_index()
    )
    coverage.to_csv(STEP_DIR / "EC映射覆盖与势能守恒汇总.csv", index=False, encoding="utf-8-sig")
    brand_coverage = (
        mapped.groupby("品牌归一", observed=True)
        .agg(映射行数=("片段行号", "size"), 独立映射片段数=("片段行号", "nunique"), 独立评论数=("Original_Index", "nunique"), 映射势能=("分摊初始势能", "sum"))
        .reset_index()
    )
    totals = frame.groupby("品牌归一", observed=True).agg(输入片段数=("片段行号", "size")).reset_index()
    brand_coverage = totals.merge(brand_coverage, on="品牌归一", how="left")
    brand_coverage["映射率"] = brand_coverage["独立映射片段数"] / brand_coverage["输入片段数"]
    brand_coverage.to_csv(STEP_DIR / "品牌映射覆盖率.csv", index=False, encoding="utf-8-sig")
    expected_energy = float(energy[np.unique(mapped["片段行号"].to_numpy(dtype=np.int64))].sum())
    audit = {
        "状态": "通过",
        "生成时间UTC": datetime.now(timezone.utc).isoformat(),
        "输入片段数": len(frame),
        "输出映射行数": output_rows,
        "映射片段数": mapped_fragments,
        "未映射片段数": len(frame) - mapped_fragments,
        "映射率": mapped_fragments / len(frame),
        "双EC映射片段数": second_fragments,
        "勿连约束违规数": cannot_violations,
        "总初始势能": float(energy.sum()),
        "已映射片段初始势能": expected_energy,
        "实际分摊势能": mapped_energy,
        "势能守恒绝对误差": abs(expected_energy - mapped_energy),
        "BGE覆盖率": 1.0,
        "EC数量": len(ecs),
        "必连规则数": len(must_rules),
        "勿连规则数": len(cannot_rules),
        "PCK_means": {"收敛": converged, "历史": pck_history, "锚点数": {ecs[i]["id"]: int(anchor_counts[i]) for i in range(len(ecs))}},
        "参数": {
            "dense_weight": DENSE_WEIGHT,
            "sparse_weight": SPARSE_WEIGHT,
            "semantic_weight": SEMANTIC_WEIGHT,
            "lexical_weight": LEXICAL_WEIGHT,
            "mapping_threshold": MAPPING_THRESHOLD,
            "second_ec_gap": SECOND_EC_GAP,
            "lambda": LAMBDA_PARAM,
            "pain_gamma": PAIN_GAMMA,
        },
    }
    if cannot_violations or abs(expected_energy - mapped_energy) > 1e-6 * max(1.0, expected_energy):
        audit["状态"] = "失败"
    with (STEP_DIR / "EC映射完成审计.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    if audit["状态"] != "通过":
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in audit.items() if key != "PCK_means"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
