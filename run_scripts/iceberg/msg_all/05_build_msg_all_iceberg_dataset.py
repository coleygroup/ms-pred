#!/usr/bin/env python3
"""Create data/spec_datasets/msg_all_iceberg from ICEBERG CE imputations.

The output labels.tsv starts from full MassSpecGym labels, fills missing
collision energies in place, and rewrites retrieval candidate tables so test
queries use the ICEBERG-imputed energies.
"""

from __future__ import annotations

import ast
import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


# Edit this block when rebuilding the imputed full MassSpecGym dataset.
SOURCE_DATASET_DIR = Path("data/spec_datasets/msg_all")
OUT_DATASET_DIR = Path("data/spec_datasets/msg_all_iceberg")
SOURCE_LABELS = SOURCE_DATASET_DIR / "labels.tsv"
IMPUTED_TABLE = Path("results/iceberg_msg_known_ce/iceberg_ce_imputation/imputed_collision_energies.tsv")
CANDIDATE_TABLE = Path("results/iceberg_msg_known_ce/iceberg_ce_imputation/missing_ce_candidates.tsv")

# Operational controls.
ALLOW_MISSING_IMPUTATIONS = False
COPY_FILES = False
OVERWRITE_LABELS = True
DROP_LABEL_COLUMNS = {"collision_imputed", "confidence"}

def get_config() -> SimpleNamespace:
    return SimpleNamespace(
        source_dataset_dir=SOURCE_DATASET_DIR,
        out_dataset_dir=OUT_DATASET_DIR,
        source_labels=SOURCE_LABELS,
        imputed_table=IMPUTED_TABLE,
        candidate_table=CANDIDATE_TABLE,
        allow_missing_imputations=ALLOW_MISSING_IMPUTATIONS,
        copy_files=COPY_FILES,
        overwrite_labels=OVERWRITE_LABELS,
    )


def parse_collision_list(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            return []
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = [text]
    if not isinstance(value, (list, tuple, np.ndarray)):
        value = [value]
    out = []
    for item in value:
        if item is None:
            out.append(float("nan"))
            continue
        if isinstance(item, str):
            item = item.strip()
            if item == "" or item.lower() in {"nan", "none", "null"}:
                out.append(float("nan"))
                continue
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def has_missing_collision_energy(value: Any) -> bool:
    vals = parse_collision_list(value)
    return len(vals) == 0 or any(math.isnan(v) for v in vals)


def clean_label_columns(labels: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        col
        for col in labels.columns
        if col.startswith("Unnamed:") or col in DROP_LABEL_COLUMNS
    ]
    if drop_cols:
        labels = labels.drop(columns=drop_cols)
    return labels


def normalize_instruments(labels: pd.DataFrame) -> pd.DataFrame:
    if "instrument" not in labels.columns:
        labels["instrument"] = "Orbitrap"
        return labels
    valid = labels["instrument"].isin(["Orbitrap", "QTOF", "IT-FT", "Unknown"])
    labels.loc[~valid, "instrument"] = "Orbitrap"
    return labels


def format_collision_list(ev: Any) -> str:
    return str([f"{float(ev):.0f}"])


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if not src.exists():
        return
    if dst.is_symlink():
        dst.unlink()
    if dst.exists():
        return
    if copy_files:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        rel_src = os.path.relpath(src.absolute(), dst.parent.absolute())
        os.symlink(rel_src, dst)


def build_imputed_labels(
    labels: pd.DataFrame,
    candidates: pd.DataFrame,
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    spec_to_ev = dict(scored[["spec", "best_collision_energy"]].values)
    label_specs = set(labels["spec"])
    scored_not_in_labels = sorted(set(spec_to_ev) - label_specs)
    if scored_not_in_labels:
        preview = ", ".join(scored_not_in_labels[:10])
        raise ValueError(f"{len(scored_not_in_labels)} scored specs are absent from labels. Examples: {preview}")

    candidate_specs = set(candidates["spec"])
    scored_not_in_candidates = sorted(set(spec_to_ev) - candidate_specs)
    if scored_not_in_candidates:
        preview = ", ".join(scored_not_in_candidates[:10])
        raise ValueError(f"{len(scored_not_in_candidates)} scored specs are absent from candidate table. Examples: {preview}")

    out_labels = labels.copy()

    missing_mask = out_labels["collision_energies"].map(has_missing_collision_energy)
    fill_mask = missing_mask & out_labels["spec"].isin(spec_to_ev)
    out_labels.loc[fill_mask, "collision_energies"] = out_labels.loc[fill_mask, "spec"].map(
        lambda spec: format_collision_list(spec_to_ev[spec])
    )
    return out_labels, int(fill_mask.sum())


def rewrite_retrieval_collision_energies(retrieval_path: Path, labels: pd.DataFrame) -> dict[str, int]:
    label_map = dict(labels[["spec", "collision_energies"]].astype({"spec": str}).values)
    cands = pd.read_csv(retrieval_path, sep="\t", dtype={"spec": str}, keep_default_na=False)
    missing_specs = sorted(set(cands["spec"]) - set(label_map))
    if missing_specs:
        preview = ", ".join(missing_specs[:10])
        raise ValueError(f"{len(missing_specs)} retrieval specs missing from labels. Examples: {preview}")
    cands["collision_energies"] = cands["spec"].map(label_map)
    cands.to_csv(retrieval_path, sep="\t", index=False)
    missing_ce_specs = int(cands.loc[cands["collision_energies"].map(has_missing_collision_energy), "spec"].nunique())
    return {"rows": int(len(cands)), "unique_specs": int(cands["spec"].nunique()), "missing_ce_specs": missing_ce_specs}


def copy_and_rewrite_retrieval(source_dir: Path, out_dir: Path, labels: pd.DataFrame) -> dict[str, dict[str, int]]:
    src_retrieval = source_dir / "retrieval"
    out_retrieval = out_dir / "retrieval"
    if out_retrieval.is_symlink():
        out_retrieval.unlink()
    out_retrieval.mkdir(exist_ok=True)

    summary = {}
    for name in ["cands_df_test_formula_256.tsv", "cands_df_test_mass_256.tsv"]:
        src = src_retrieval / name
        if not src.exists():
            raise FileNotFoundError(f"Missing retrieval candidate file: {src}")
        dst = out_retrieval / name
        shutil.copy2(src, dst)
        summary[name] = rewrite_retrieval_collision_energies(dst, labels)
    return summary


def main() -> None:
    args = get_config()
    source_dir = Path(args.source_dataset_dir)
    out_dir = Path(args.out_dataset_dir)
    labels_path = Path(args.source_labels)
    imputed_path = Path(args.imputed_table)
    candidate_path = Path(args.candidate_table)

    labels = normalize_instruments(clean_label_columns(pd.read_csv(labels_path, sep="\t", keep_default_na=False)))
    imputed = pd.read_csv(imputed_path, sep="\t")
    scored = imputed[imputed["status"] == "scored"].copy()
    candidates = pd.read_csv(candidate_path, sep="\t")

    known_count = int((~labels["collision_energies"].map(has_missing_collision_energy)).sum())
    missing_before = int(labels["collision_energies"].map(has_missing_collision_energy).sum())

    missing_imputation_count = int((imputed["status"] != "scored").sum())
    if missing_imputation_count and not args.allow_missing_imputations:
        failed = imputed[imputed["status"] != "scored"]
        preview = ", ".join(failed["spec"].head(10).astype(str))
        raise ValueError(f"{missing_imputation_count} imputation rows are not scored. Examples: {preview}")

    out_labels, fill_count = build_imputed_labels(labels, candidates, scored)
    duplicated_specs = out_labels["spec"].duplicated()
    if duplicated_specs.any():
        preview = ", ".join(out_labels.loc[duplicated_specs, "spec"].head(10).astype(str))
        raise ValueError(f"{int(duplicated_specs.sum())} duplicate specs in output labels. Examples: {preview}")
    remaining_missing = int(out_labels["collision_energies"].map(has_missing_collision_energy).sum())
    if remaining_missing and not args.allow_missing_imputations:
        raise ValueError(f"{remaining_missing} rows still have missing collision energies")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_labels_path = out_dir / "labels.tsv"
    if out_labels_path.exists() and not args.overwrite_labels:
        raise FileExistsError(f"{out_labels_path} already exists. Pass --overwrite-labels to replace it.")
    out_labels.to_csv(out_labels_path, sep="\t", index=False)

    for name in ["spec_files.hdf5", "splits", "magma_outputs"]:
        link_or_copy(source_dir / name, out_dir / name, args.copy_files)

    subformulae_dir = out_dir / "subformulae"
    if subformulae_dir.is_symlink():
        subformulae_dir.unlink()
    subformulae_dir.mkdir(exist_ok=True)

    retrieval_summary = copy_and_rewrite_retrieval(source_dir, out_dir, out_labels)

    summary = {
        "source_labels": str(labels_path),
        "imputed_table": str(imputed_path),
        "candidate_table": str(candidate_path),
        "rows": int(len(out_labels)),
        "known_collision_energy_rows": int(known_count),
        "missing_collision_energy_rows_before_imputation": int(missing_before),
        "filled_collision_energies": int(fill_count),
        "remaining_missing_collision_energies": int(remaining_missing),
        "retrieval": retrieval_summary,
    }
    (out_dir / "msg_all_iceberg_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
