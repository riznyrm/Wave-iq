"""Explicit, reversible repairs and order-family separation of the handover CSV."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .features import BASE_NUM, RAW_FEATURES

TARGET = "Rejection"
GROUP = "Order_Group"
SOURCE_COLUMNS = [
    "ID", "Month", "Construction", "Req_Finish_Fabrics", "Fabric_Allowance",
    "Rec_Beam_length(yds)", "Shrink_allow", "Previous_pdn", "Req_grey_fabric",
    "Req_beam_length(yds)", "Total_Pdn(yds)", TARGET, "warp_count", "weft_count",
    "epi", "ppi",
]


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family_tokens(identifier: str) -> list[str]:
    """Conservative grouping rule, not a claim that two records are one event.

    Share every five-digit family across suborders, suffixes and SF/ED prefixes.
    SF-13026+27 links families 13026 and 13027. 13035-1+2 stays family 13035.
    An unresolved short numeric ID such as 94-5 is kept as a separate RAW key.
    """
    normalized = re.sub(r"\s+", "", str(identifier).upper())
    tokens = set(re.findall(r"(?<!\d)\d{5}(?!\d)", normalized))
    combined = re.fullmatch(r"(?:SF|ED)-?(\d{5})((?:\+\d{1,5})+)", normalized)
    if combined:
        first = combined.group(1)
        for short in combined.group(2).split("+")[1:]:
            tokens.add(first[:5 - len(short)] + short)
    return sorted(tokens) if tokens else [f"RAW:{normalized}"]


def build_family_map(identifiers) -> pd.DataFrame:
    """Union connected families, including the links implied by combined IDs."""
    identifiers = sorted(set(str(value).strip() for value in identifiers))
    parent: dict[str, str] = {}

    def find(token):
        parent.setdefault(token, token)
        if parent[token] != token:
            parent[token] = find(parent[token])
        return parent[token]

    for identifier in identifiers:
        tokens = family_tokens(identifier)
        for token in tokens:
            find(token)
        for token in tokens[1:]:
            a, b = sorted([find(tokens[0]), find(token)])
            parent[b] = a
    return pd.DataFrame([
        {"ID": identifier, "Family_Tokens": ";".join(family_tokens(identifier)),
         GROUP: "FAMILY:" + find(family_tokens(identifier)[0])}
        for identifier in identifiers
    ])


@dataclass
class PreparedData:
    source: pd.DataFrame
    clean: pd.DataFrame
    development: pd.DataFrame
    test: pd.DataFrame
    purged: pd.DataFrame
    excluded: pd.DataFrame
    split_manifest: pd.DataFrame
    family_map: pd.DataFrame
    audit: dict


def prepare_data(
    source_path: str | Path,
    *,
    seed: int = 42,
    test_size: float = 0.20,
    expected_sha256: str | None = None,
) -> PreparedData:
    source_path = Path(source_path)
    digest = sha256_file(source_path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Input SHA256 differs from the reviewed handover. Review the new data and update config explicitly.")
    source = pd.read_csv(source_path, dtype={"ID": str}, low_memory=False)
    missing = sorted(set(SOURCE_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"Missing handover columns: {missing}. Use the 16-column EDA handover, not the raw detail CSV.")
    source = source[SOURCE_COLUMNS].copy()
    if source["ID"].isna().any():
        raise ValueError("Missing order ID. Repair upstream before splitting.")
    source["ID"] = source["ID"].str.strip()
    if source["ID"].eq("").any():
        raise ValueError("Blank order ID. Repair upstream before splitting.")
    source[TARGET] = pd.to_numeric(source[TARGET], errors="raise")
    if not np.isfinite(source[TARGET]).all() or source[TARGET].lt(0).any():
        raise ValueError("Target values must be finite, observed and nonnegative. Targets are never imputed.")
    # Reproduce Member 02's split BEFORE removing IDs or duplicates. Changing the
    # group universe first would silently move the original test orders.
    dev_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed,
    ).split(source, groups=source["ID"]))
    original_test_ids = set(source.iloc[test_idx]["ID"])
    original_dev_ids = set(source.iloc[dev_idx]["ID"])
    assert original_dev_ids.isdisjoint(original_test_ids)

    mapping = build_family_map(source["ID"])
    group_lookup = mapping.set_index("ID")[GROUP]
    original_test_families = set(group_lookup.loc[list(original_test_ids)])

    duplicate_mask = source.duplicated(subset=SOURCE_COLUMNS, keep="first")
    placeholder_mask = ~source["ID"].str.contains(r"\d", regex=True)
    exclusion_reason = pd.Series("", index=source.index, dtype=object)
    exclusion_reason.loc[duplicate_mask] = "exact_duplicate"
    exclusion_reason.loc[~duplicate_mask & placeholder_mask] = "nonnumeric_order_placeholder"
    excluded = source.loc[exclusion_reason.ne("")].copy()
    excluded.insert(0, "Source_Row", excluded.index)
    excluded["Exclusion_Reason"] = exclusion_reason.loc[excluded.index]

    clean = source.loc[~duplicate_mask & ~placeholder_mask].copy()
    clean.insert(0, "Source_Row", clean.index)
    # The immutable row key follows the supplied file, not a newly invented event ID.
    clean["Record_ID"] = [f"{digest[:12]}:{int(i)}" for i in clean.index]
    clean[GROUP] = clean["ID"].map(group_lookup)
    clean["Split"] = np.where(clean["ID"].isin(original_test_ids), "Test", "Development")
    alias_to_test = clean[GROUP].isin(original_test_families) & clean["Split"].eq("Development")
    clean.loc[alias_to_test, "Split"] = "Purged"
    clean.reset_index(drop=True, inplace=True)
    dev = clean.loc[clean["Split"].eq("Development")].copy().reset_index(drop=True)
    test = clean.loc[clean["Split"].eq("Test")].copy().reset_index(drop=True)
    purged = clean.loc[clean["Split"].eq("Purged")].copy().reset_index(drop=True)
    if dev[GROUP].nunique() < 5 or test[GROUP].nunique() < 2:
        raise ValueError("Insufficient distinct order families for 5-fold CV and a separate test set.")
    assert set(dev[GROUP]).isdisjoint(test[GROUP])
    assert set(dev["ID"]).isdisjoint(test["ID"])
    assert clean["Record_ID"].is_unique

    manifest = clean[["Record_ID", "Source_Row", "ID", "Month", GROUP, "Split"]].copy()
    numeric = source[BASE_NUM].apply(pd.to_numeric, errors="coerce")
    # Nonfinite/negative input values are not silently accepted as physical measurements.
    if numeric.lt(0).any().any():
        raise ValueError("Negative planning inputs require upstream review.")
    audit = {
        "source_name": source_path.name, "source_sha256": digest,
        "source_rows": len(source), "source_columns": len(SOURCE_COLUMNS),
        "source_missing_cells": int(source.isna().sum().sum()),
        "exact_duplicate_rows_removed": int(duplicate_mask.sum()),
        "placeholder_rows_in_source": int(placeholder_mask.sum()),
        "unique_placeholder_rows_removed": int((~duplicate_mask & placeholder_mask).sum()),
        "clean_rows_including_purged": len(clean),
        "original_exact_id_development_rows": len(dev_idx),
        "original_exact_id_test_rows": len(test_idx),
        "original_exact_id_development_groups": len(original_dev_ids),
        "original_exact_id_test_groups": len(original_test_ids),
        "development_rows": len(dev), "test_rows": len(test),
        "purged_alias_rows": len(purged),
        "development_families": int(dev[GROUP].nunique()),
        "test_families": int(test[GROUP].nunique()),
        "purged_raw_ids": int(purged["ID"].nunique()),
        "family_overlap_development_test": len(set(dev[GROUP]) & set(test[GROUP])),
        "exact_id_overlap_development_test": len(set(dev["ID"]) & set(test["ID"])),
        "split_policy": "preserve_member02_exact_id_test_then_purge_related_development_families",
        "prediction_target": TARGET, "target_unit": "unverified_recorded_amount",
        "observation": "one_distinct_supplied_handover_row_not_verified_unique_event",
        "duplicate_assumption": "identical export rows count once; source event keys are unavailable",
    }
    return PreparedData(source, clean, dev, test, purged, excluded, manifest, mapping, audit)


def export_prepared(prepared: PreparedData, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("development", prepared.development), ("test", prepared.test),
        ("split_manifest", prepared.split_manifest),
    ]:
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "data_audit.json").write_text(json.dumps(prepared.audit, indent=2), encoding="utf-8")
