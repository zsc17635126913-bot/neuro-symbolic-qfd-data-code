from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


EC_ID_COL = "二级指标编号"
EC_NAME_COL = "二级指标名称"
L1_COL = "一级技术域"
SYNONYM_COL = "同义词/对象词"
POSITIVE_COL = "正向关键词"
NEGATIVE_COL = "负向关键词"
EXCLUDE_COL = "排除词"


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    return re.sub(r"[^0-9a-z_\u4e00-\u9fff]+", "", text)


def split_terms(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    terms = []
    for item in re.split(r"[；;|、,，\n]+", str(value)):
        term = normalize_text(item)
        if len(term) >= 2:
            terms.append(term)
    return list(dict.fromkeys(terms))


def load_ontology(kg_path: Path) -> tuple[pd.DataFrame, list[dict], list[dict], list[dict]]:
    kg = pd.ExcelFile(kg_path)
    ontology_sheet = next(name for name in kg.sheet_names if name.startswith("02_"))
    ownership_sheet = next(name for name in kg.sheet_names if name.startswith("03_"))
    must_sheet = next(name for name in kg.sheet_names if name.startswith("04_"))
    cannot_sheet = next(name for name in kg.sheet_names if name.startswith("05_"))
    ontology = pd.read_excel(kg, sheet_name=ontology_sheet)
    ownership = pd.read_excel(kg, sheet_name=ownership_sheet)
    must_raw = pd.read_excel(kg, sheet_name=must_sheet)
    cannot_raw = pd.read_excel(kg, sheet_name=cannot_sheet)
    ontology = ontology[ontology[EC_ID_COL].astype(str).str.fullmatch(r"EC\d{2}", na=False)].copy()
    ontology = ontology[ontology[EC_ID_COL] != "EC33"].drop_duplicates(EC_ID_COL)
    ownership = ownership[ownership[EC_ID_COL].isin(ontology[EC_ID_COL])].drop_duplicates(EC_ID_COL)
    if len(ontology) != 32 or len(ownership) != 32:
        raise ValueError("冻结本体必须包含32个EC及唯一一级归属")

    valid_ids = set(ontology[EC_ID_COL])

    def rows_to_rules(frame: pd.DataFrame, kind: str) -> list[dict]:
        id_column = next(column for column in frame.columns if "EC编号" in str(column))
        trigger_column = next(
            column for column in frame.columns if "触发词" in str(column) or "排斥触发词" in str(column)
        )
        rules = []
        for row in frame.to_dict("records"):
            ec_id = str(row.get(id_column, "")).strip()
            terms = split_terms(row.get(trigger_column, ""))
            if ec_id in valid_ids and terms:
                rules.append({"kind": kind, "ec_id": ec_id, "terms": terms})
        return rules

    owner_by_id = ownership.set_index(EC_ID_COL)[L1_COL].to_dict()
    ecs = []
    for row in ontology.sort_values(EC_ID_COL).to_dict("records"):
        ec_id = row[EC_ID_COL]
        name = str(row[EC_NAME_COL])
        terms = []
        for column in (SYNONYM_COL, POSITIVE_COL, NEGATIVE_COL):
            terms.extend(split_terms(row.get(column, "")))
        terms.extend([normalize_text(name), normalize_text(f"{ec_id}_{name}")])
        ecs.append(
            {
                "id": ec_id,
                "name": name,
                "node": f"{ec_id}_{name}",
                "parent": owner_by_id[ec_id],
                "terms": sorted(set(terms), key=lambda value: (-len(value), value)),
                "exclusions": sorted(set(split_terms(row.get(EXCLUDE_COL, ""))), key=lambda value: (-len(value), value)),
            }
        )
    return ontology, ecs, rows_to_rules(must_raw, "must"), rows_to_rules(cannot_raw, "cannot")


def terms_by_ec(rules: list[dict]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for rule in rules:
        result[rule["ec_id"]].extend(rule["terms"])
    return {key: list(dict.fromkeys(values)) for key, values in result.items()}


def top_k_mean(values: np.ndarray, k: int) -> float:
    if len(values) == 0:
        return 0.0
    actual = min(len(values), max(1, int(k)))
    return float(np.partition(values, len(values) - actual)[-actual:].mean())


def stable_cvar(values: np.ndarray, alpha: float = 0.80, minimum_tail: int = 20) -> tuple[float, float, int, int]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0, 0, 0
    strict_k = max(1, math.ceil((1.0 - alpha) * len(values)))
    stable_k = min(len(values), max(strict_k, minimum_tail))
    return top_k_mean(values, strict_k), top_k_mean(values, stable_k), strict_k, stable_k


def build_topology(ecs: list[dict], diffusion_gamma: float = 0.3):
    l1_order = list(dict.fromkeys(ec["parent"] for ec in ecs))
    if len(l1_order) != 7:
        raise ValueError(f"预期7个一级技术域，实际{len(l1_order)}")
    l1_index = {name: index for index, name in enumerate(l1_order)}
    parent = np.zeros((len(ecs), len(l1_order)), dtype=float)
    for index, ec in enumerate(ecs):
        parent[index, l1_index[ec["parent"]]] = 1.0
    cross = np.zeros_like(parent)
    cross_links = {
        "EC01": [("辅助驾驶与安全", 0.50), ("空调与热管理", 0.50)],
        "EC02": [("空调与热管理", 0.25)],
        "EC04": [("辅助驾驶与安全", 0.25)],
        "EC05": [("车辆服务与保障", 0.50)],
        "EC06": [("辅助驾驶与安全", 0.75)],
        "EC07": [("电子电气与智能网联", 0.25)],
        "EC16": [("辅助驾驶与安全", 0.50)],
        "EC19": [("辅助驾驶与安全", 0.50), ("车辆服务与保障", 0.50)],
        "EC22": [("动力与高压系统", 0.75), ("辅助驾驶与安全", 0.25)],
        "EC24": [("底盘与悬架系统", 0.50)],
        "EC25": [("底盘与悬架系统", 0.25)],
        "EC26": [("底盘与悬架系统", 0.50)],
        "EC27": [("底盘与悬架系统", 0.25)],
        "EC29": [("电子电气与智能网联", 0.50), ("车身与座舱结构", 0.25)],
        "EC31": [("动力与高压系统", 0.50)],
    }
    audit = []
    for row, ec in enumerate(ecs):
        for suffix, weight in cross_links.get(ec["id"], []):
            target = next((name for name in l1_order if name.endswith(suffix)), None)
            if target is None or target == ec["parent"]:
                continue
            cross[row, l1_index[target]] = weight
            audit.append({"EC编号": ec["id"], "正式归属": ec["parent"], "跨域影响": target, "权重": weight})
    weighted = parent + cross
    adjacency = weighted @ weighted.T
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    laplacian = np.eye(len(ecs)) - inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    projection = np.divide(
        weighted,
        weighted.sum(axis=0, keepdims=True),
        out=np.zeros_like(weighted),
        where=weighted.sum(axis=0, keepdims=True) > 0,
    )
    phi = np.linalg.inv(np.eye(len(ecs)) + diffusion_gamma * laplacian)
    jacobian = phi @ projection
    return l1_order, parent, cross, laplacian, jacobian, pd.DataFrame(audit)
