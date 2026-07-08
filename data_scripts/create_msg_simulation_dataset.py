#!/usr/bin/env python3
"""Create or audit a MassSpecGym simulation-challenge dataset.

The source of truth is MassSpecGym1.5.tsv. Rows are retained only when
simulation_challenge is true and collision energy is known. The output mirrors
the repository's spec_datasets layout and creates tiny debug labels/splits for
quick MARASON checks.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import h5py
import pandas as pd


DEFAULT_SOURCE_TABLE = (
    "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/"
    "resolve/main/data/MassSpecGym1.5.tsv"
)
DEFAULT_SOURCE_DATASET = Path("data/spec_datasets/msg")
DEFAULT_OUTPUT_DATASET = Path("data/spec_datasets/msg_simulation")
DEFAULT_CANDIDATES = (
    "retrieval/cands_df_test_formula_256.tsv",
    "retrieval/cands_df_test_mass_256.tsv",
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--output-dataset", type=Path, default=DEFAULT_OUTPUT_DATASET)
    parser.add_argument("--force-output", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--debug-train", type=int, default=64)
    parser.add_argument("--debug-val", type=int, default=16)
    parser.add_argument("--debug-test", type=int, default=8)
    parser.add_argument(
        "--candidate-file",
        action="append",
        default=None,
        help="Candidate table path relative to the source dataset. May be repeated.",
    )
    parser.add_argument("--audit-out", type=Path, default=None)
    parser.add_argument(
        "--ensure-debug-hdf5",
        action="store_true",
        help="Only create spec_files_debug.hdf5 from existing labels_debug.tsv.",
    )
    return parser.parse_args()


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def format_collision_energy(value) -> str | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(val):
        return None
    if val.is_integer():
        label = str(int(val))
    else:
        label = f"{val:g}"
    if "[imputed]" in label:
        return None
    return f"['{label}']"


def normalize_fold(value) -> str:
    val = str(value).strip().lower()
    if val == "train":
        return "train"
    if val in {"val", "valid", "validation"}:
        return "val"
    return "test"


def read_simulation_source(source_table: str, chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_chunks = []
    split_chunks = []
    source_rows = 0
    sim_rows = 0

    usecols = [
        "identifier",
        "smiles",
        "inchikey",
        "formula",
        "precursor_mz",
        "adduct",
        "instrument_type",
        "collision_energy",
        "fold",
        "simulation_challenge",
    ]
    for chunk in pd.read_csv(source_table, sep="\t", usecols=usecols, chunksize=chunksize):
        source_rows += len(chunk)
        chunk = chunk[truthy(chunk["simulation_challenge"])].copy()
        sim_rows += len(chunk)
        chunk["collision_energies"] = chunk["collision_energy"].map(format_collision_energy)
        chunk = chunk.dropna(subset=["collision_energies"])
        chunk = chunk[~chunk["collision_energies"].astype(str).str.contains(r"\[imputed\]", regex=True)]

        label_chunks.append(
            pd.DataFrame(
                {
                    "dataset": "MassSpecGym",
                    "spec": chunk["identifier"].astype(str),
                    "ionization": chunk["adduct"],
                    "formula": chunk["formula"],
                    "smiles": chunk["smiles"],
                    "inchikey": chunk["inchikey"],
                    "instrument": chunk["instrument_type"],
                    "collision_energies": chunk["collision_energies"],
                    "precursor": chunk["precursor_mz"],
                    "collision_imputed": False,
                }
            )
        )
        split_chunks.append(
            pd.DataFrame(
                {
                    "name": chunk["identifier"].astype(str),
                    "split": chunk["fold"].map(normalize_fold),
                }
            )
        )

    labels = pd.concat(label_chunks, ignore_index=True)
    splits = pd.concat(split_chunks, ignore_index=True)
    labels.attrs["source_rows"] = source_rows
    labels.attrs["simulation_rows_before_known_ce_filter"] = sim_rows
    return labels, splits


def current_msg_matches(labels: pd.DataFrame, source_dataset: Path) -> dict:
    current_labels_path = source_dataset / "labels.tsv"
    out = {
        "current_labels": str(current_labels_path),
        "current_exists": current_labels_path.exists(),
        "same_spec_set": False,
        "current_rows": 0,
        "current_unique_specs": 0,
        "current_collision_imputed_true": None,
        "current_literal_imputed_count": None,
    }
    if not current_labels_path.exists():
        return out

    current = pd.read_csv(current_labels_path, sep="\t")
    out["current_rows"] = int(len(current))
    out["current_unique_specs"] = int(current["spec"].nunique()) if "spec" in current else 0
    if "collision_imputed" in current:
        out["current_collision_imputed_true"] = int(current["collision_imputed"].fillna(False).astype(bool).sum())
    out["current_literal_imputed_count"] = int(
        current.astype(str).apply(lambda col: col.str.contains(r"\[imputed\]", regex=True)).sum().sum()
    )
    out["same_spec_set"] = set(current["spec"].astype(str)) == set(labels["spec"].astype(str))
    return out


def replace_path(path: Path, overwrite: bool):
    if not path.exists() and not path.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(f"{path} exists. Re-run with --overwrite to replace it.")
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def symlink_or_copy(src: Path, dst: Path, overwrite: bool):
    if not src.exists():
        return False
    replace_path(dst, overwrite=overwrite)
    rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
    os.symlink(rel_src, dst, target_is_directory=src.is_dir())
    return True


def write_tsv(df: pd.DataFrame, path: Path, overwrite: bool):
    replace_path(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def collision_key(value) -> str | None:
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return None


def iter_label_collision_pairs(labels: pd.DataFrame) -> set[tuple[str, str]]:
    pairs = set()
    for spec, collision_energies in zip(labels["spec"].astype(str), labels["collision_energies"].astype(str)):
        try:
            parsed = ast.literal_eval(collision_energies)
        except (SyntaxError, ValueError):
            parsed = [collision_energies]
        if not isinstance(parsed, list):
            parsed = [parsed]
        for collision_energy in parsed:
            ce_key = collision_key(str(collision_energy).split()[0])
            if ce_key is not None:
                pairs.add((spec, ce_key))
    return pairs


def parse_subformula_key(key: str) -> tuple[str, str] | None:
    match = re.match(r"(.+)_collision\s+([0-9]+\.?[0-9]*|nan)\.json$", key)
    if match is None:
        return None
    spec, collision_energy = match.groups()
    ce_key = collision_key(collision_energy)
    if ce_key is None:
        return None
    return spec, ce_key


def filter_subformulae(source_dataset: Path, output_dataset: Path, labels: pd.DataFrame, overwrite: bool) -> dict:
    source_root = (source_dataset / "labels.tsv").resolve().parent
    source_h5 = source_root / "subformulae/no_subform.hdf5"
    if not source_h5.exists():
        source_h5 = source_dataset / "subformulae/no_subform.hdf5"
    if not source_h5.exists():
        raise FileNotFoundError(source_h5)

    valid_pairs = iter_label_collision_pairs(labels)
    subformula_dir = output_dataset / "subformulae"
    replace_path(subformula_dir, overwrite=overwrite)
    subformula_dir.mkdir(parents=True, exist_ok=True)
    out_h5 = subformula_dir / "no_subform.hdf5"

    source_entries = 0
    written = 0
    skipped_unparsed = 0
    skipped_not_in_labels = 0
    with h5py.File(source_h5, "r") as src, h5py.File(out_h5, "w") as dst:
        for key in src.keys():
            source_entries += 1
            pair = parse_subformula_key(key)
            if pair is None:
                skipped_unparsed += 1
                continue
            if pair not in valid_pairs:
                skipped_not_in_labels += 1
                continue
            src.copy(key, dst, name=key)
            written += 1

    return {
        "source_subformulae": str(source_h5),
        "output_subformulae": str(out_h5),
        "label_collision_pairs": int(len(valid_pairs)),
        "source_entries": int(source_entries),
        "written_entries": int(written),
        "skipped_unparsed": int(skipped_unparsed),
        "skipped_not_in_labels": int(skipped_not_in_labels),
        "missing_label_pairs": int(len(valid_pairs) - written),
    }


def copy_debug_spec_files(source_dataset: Path, output_dataset: Path, overwrite: bool) -> dict:
    labels_path = output_dataset / "labels_debug.tsv"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    source_h5 = (source_dataset / "labels.tsv").resolve().parent / "spec_files.hdf5"
    if not source_h5.exists():
        source_h5 = source_dataset / "spec_files.hdf5"
    if not source_h5.exists():
        raise FileNotFoundError(source_h5)

    out_h5 = output_dataset / "spec_files_debug.hdf5"
    replace_path(out_h5, overwrite=overwrite)
    specs = pd.read_csv(labels_path, sep="\t")["spec"].astype(str).tolist()
    missing = []
    written = 0
    with h5py.File(source_h5, "r") as src, h5py.File(out_h5, "w") as dst:
        for spec in specs:
            source_key = None
            for key in (spec, f"{spec}.ms"):
                if key in src:
                    source_key = key
                    break
            if source_key is None:
                missing.append(spec)
                continue
            src.copy(source_key, dst, name=source_key)
            written += 1
    return {
        "source_spec_files": str(source_h5),
        "debug_spec_files": str(out_h5),
        "debug_specs_requested": int(len(specs)),
        "debug_specs_written": int(written),
        "debug_specs_missing": missing[:20],
        "debug_specs_missing_count": int(len(missing)),
    }


def filter_candidate_table(path: Path, out_path: Path, spec_set: set[str], overwrite: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    chunks = []
    for chunk in pd.read_csv(path, sep="\t", chunksize=100000):
        chunk = chunk[chunk["spec"].astype(str).isin(spec_set)].copy()
        if "collision_energies" in chunk:
            chunk = chunk[~chunk["collision_energies"].astype(str).str.contains(r"\[imputed\]", regex=True)]
        chunks.append(chunk)
    out_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    write_tsv(out_df, out_path, overwrite=overwrite)
    return out_df


def first_n_specs(splits: pd.DataFrame, split_name: str, n: int, allowed: set[str] | None = None) -> list[str]:
    df = splits[splits["split"] == split_name]
    if allowed is not None:
        df = df[df["name"].astype(str).isin(allowed)]
    return df["name"].astype(str).drop_duplicates().head(n).tolist()


def write_debug_files(
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    output_dataset: Path,
    formula_candidates: pd.DataFrame,
    debug_train: int,
    debug_val: int,
    debug_test: int,
    overwrite: bool,
):
    candidate_specs = set(formula_candidates["spec"].astype(str)) if len(formula_candidates) else None
    test_specs = first_n_specs(splits, "test", debug_test, allowed=candidate_specs)
    if len(test_specs) < debug_test:
        test_specs = first_n_specs(splits, "test", debug_test)
    train_specs = first_n_specs(splits, "train", debug_train)
    val_specs = first_n_specs(splits, "val", debug_val)

    chosen = train_specs + val_specs + test_specs
    debug_labels = labels[labels["spec"].astype(str).isin(chosen)].copy()
    debug_labels["_order"] = debug_labels["spec"].map({spec: ind for ind, spec in enumerate(chosen)})
    debug_labels = debug_labels.sort_values("_order").drop(columns=["_order"])
    debug_splits = pd.DataFrame(
        {
            "name": train_specs + val_specs + test_specs,
            "split": ["train"] * len(train_specs) + ["val"] * len(val_specs) + ["test"] * len(test_specs),
        }
    )
    debug_candidates = formula_candidates[formula_candidates["spec"].astype(str).isin(test_specs)].copy()

    write_tsv(debug_labels, output_dataset / "labels_debug.tsv", overwrite=overwrite)
    write_tsv(debug_splits, output_dataset / "splits/split_debug.tsv", overwrite=overwrite)
    write_tsv(
        debug_candidates,
        output_dataset / "retrieval/cands_df_test_formula_debug.tsv",
        overwrite=overwrite,
    )
    return {
        "debug_labels": int(len(debug_labels)),
        "debug_train_specs": int(len(train_specs)),
        "debug_val_specs": int(len(val_specs)),
        "debug_test_specs": int(len(test_specs)),
        "debug_candidate_rows": int(len(debug_candidates)),
    }


def assert_no_imputed(labels: pd.DataFrame, candidate_frames: Iterable[pd.DataFrame]):
    literal_count = int(
        labels.astype(str).apply(lambda col: col.str.contains(r"\[imputed\]", regex=True)).sum().sum()
    )
    collision_imputed_true = (
        int(labels["collision_imputed"].fillna(False).astype(bool).sum())
        if "collision_imputed" in labels
        else 0
    )
    for candidates in candidate_frames:
        if len(candidates) == 0:
            continue
        literal_count += int(
            candidates.astype(str).apply(lambda col: col.str.contains(r"\[imputed\]", regex=True)).sum().sum()
        )
    if literal_count or collision_imputed_true:
        raise ValueError(
            f"Imputed labels found: literal_count={literal_count}, "
            f"collision_imputed_true={collision_imputed_true}"
        )
    return {
        "literal_imputed_count": literal_count,
        "collision_imputed_true": collision_imputed_true,
    }


def main():
    args = get_args()
    if args.ensure_debug_hdf5:
        audit = {
            "debug_spec_files": copy_debug_spec_files(
                source_dataset=args.source_dataset,
                output_dataset=args.output_dataset,
                overwrite=args.overwrite,
            )
        }
        print(json.dumps(audit, indent=2))
        return

    candidate_files = args.candidate_file or list(DEFAULT_CANDIDATES)
    labels, splits = read_simulation_source(args.source_table, chunksize=args.chunksize)
    labels = labels.drop_duplicates(subset=["spec"], keep="first")
    splits = splits.drop_duplicates(subset=["name"], keep="first")

    audit = {
        "source_table": str(args.source_table),
        "source_rows": int(labels.attrs["source_rows"]),
        "simulation_rows_before_known_ce_filter": int(
            labels.attrs["simulation_rows_before_known_ce_filter"]
        ),
        "output_rows": int(len(labels)),
        "output_unique_specs": int(labels["spec"].nunique()),
    }
    audit["current_msg"] = current_msg_matches(labels, args.source_dataset)

    should_write = args.force_output or not audit["current_msg"]["same_spec_set"]
    audit["wrote_output_dataset"] = bool(should_write)
    if should_write:
        args.output_dataset.mkdir(parents=True, exist_ok=True)
        write_tsv(labels, args.output_dataset / "labels.tsv", overwrite=args.overwrite)
        write_tsv(splits, args.output_dataset / "splits/split.tsv", overwrite=args.overwrite)

        resource_dir = (args.source_dataset / "labels.tsv").resolve().parent
        linked = {}
        for name in ["magma_outputs", "spec_files.hdf5", "spec_files", "spec_files_w_eV"]:
            linked[name] = symlink_or_copy(
                resource_dir / name,
                args.output_dataset / name,
                overwrite=args.overwrite,
            )
        audit["linked_resources"] = linked
        audit["subformulae"] = filter_subformulae(
            source_dataset=args.source_dataset,
            output_dataset=args.output_dataset,
            labels=labels,
            overwrite=args.overwrite,
        )

        candidate_frames = {}
        spec_set = set(labels["spec"].astype(str))
        for candidate_file in candidate_files:
            source_path = resource_dir / candidate_file
            if not source_path.exists():
                source_path = args.source_dataset / candidate_file
            out_path = args.output_dataset / candidate_file
            candidate_frames[candidate_file] = filter_candidate_table(
                source_path,
                out_path,
                spec_set=spec_set,
                overwrite=args.overwrite,
            )

        formula_candidates = candidate_frames.get("retrieval/cands_df_test_formula_256.tsv")
        if formula_candidates is None:
            formula_candidates = next(iter(candidate_frames.values()))
        audit["debug"] = write_debug_files(
            labels=labels,
            splits=splits,
            output_dataset=args.output_dataset,
            formula_candidates=formula_candidates,
            debug_train=args.debug_train,
            debug_val=args.debug_val,
            debug_test=args.debug_test,
            overwrite=args.overwrite,
        )
        audit["debug_spec_files"] = copy_debug_spec_files(
            source_dataset=args.source_dataset,
            output_dataset=args.output_dataset,
            overwrite=args.overwrite,
        )
        audit["imputed_check"] = assert_no_imputed(labels, candidate_frames.values())

    if args.audit_out is not None:
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        args.audit_out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
