#!/usr/bin/env python3
"""Create or audit the all, known-CE, or simulation MassSpecGym dataset.

The source of truth is MassSpecGym1.5.tsv. Without --source-dataset, spectra
are converted directly from its mzs/intensities columns. An existing processed
dataset can instead supply spectra and optional annotations/candidate tables.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import h5py
import pandas as pd


DEFAULT_SOURCE_TABLE = (
    "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/"
    "resolve/main/data/MassSpecGym1.5.tsv"
)
DEFAULT_CANDIDATES = (
    "retrieval/cands_df_test_formula_256.tsv",
    "retrieval/cands_df_test_mass_256.tsv",
)
STRATEGIES = ("msg_all", "msg_known_ce", "msg_simulation")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=STRATEGIES, default="msg_simulation")
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument(
        "--source-dataset", type=Path,
        help="Reuse processed spectra and optional resources instead of converting TSV peaks.",
    )
    parser.add_argument("--output-dataset", type=Path, help="Defaults to data/spec_datasets/<strategy>.")
    parser.add_argument("--force-output", action="store_true", help="Accepted for compatibility; output is always written.")
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
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    if args.output_dataset is None:
        args.output_dataset = Path("data/spec_datasets") / args.strategy
    if args.source_dataset is not None and args.source_dataset.resolve() == args.output_dataset.resolve():
        parser.error("--source-dataset and --output-dataset must be different directories")
    if args.ensure_debug_hdf5 and args.source_dataset is None:
        args.source_dataset = args.output_dataset
    if args.candidate_file and args.source_dataset is None:
        parser.error("--candidate-file requires --source-dataset")
    for name in args.candidate_file or []:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            parser.error("--candidate-file must be a path within the source dataset")
    return args


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def format_collision_energy(value) -> str | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    if val.is_integer():
        label = str(int(val))
    else:
        label = str(val)
    return f"['{label}']"


def normalize_fold(value) -> str:
    val = str(value).strip().lower()
    if val == "train":
        return "train"
    if val in {"val", "valid", "validation"}:
        return "val"
    if val == "test":
        return "test"
    raise ValueError(f"Unrecognized source fold: {value!r}")


def spectrum_text(row) -> str:
    """Preserve source peaks and metadata in the repository's .ms text format."""
    mzs = [float(v) for v in str(row.mzs).split(",")]
    intensities = [float(v) for v in str(row.intensities).split(",")]
    if len(mzs) != len(intensities) or not mzs:
        raise ValueError(f"Invalid peak array lengths for {row.identifier}")
    if not all(math.isfinite(v) and v >= 0 for v in mzs + intensities):
        raise ValueError(f"Nonfinite or negative peaks for {row.identifier}")
    ce = format_collision_energy(row.collision_energy)
    ce_label = ast.literal_eval(ce)[0] if ce is not None else "nan"
    metadata = {
        "compound": row.identifier, "formula": row.formula,
        "parentmass": row.precursor_mz, "ionization": row.adduct,
        "InChiKey": row.inchikey, "smiles": row.smiles,
        "instrumentation": row.instrument_type,
    }
    header = "\n".join(f"#{key} {value}" for key, value in metadata.items())
    peaks = "\n".join(f"{mz} {intensity}" for mz, intensity in zip(mzs, intensities))
    return f"{header}\n\n>collision {ce_label} eV\n{peaks}\n"


def read_source(
    source_table: str, chunksize: int, strategy: str = "msg_simulation",
    spectra_h5: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_chunks = []
    split_chunks = []
    source_rows = 0
    sim_rows = 0
    selected_rows = 0
    seen = set()

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
    ]
    if strategy == "msg_simulation":
        usecols.append("simulation_challenge")
    if spectra_h5 is not None:
        usecols.extend(["mzs", "intensities"])
    with contextlib.ExitStack() as stack:
        h5 = stack.enter_context(h5py.File(spectra_h5, "w", libver="latest")) if spectra_h5 else None
        reader = stack.enter_context(pd.read_csv(source_table, sep="\t", usecols=usecols, chunksize=chunksize))
        for chunk in reader:
            source_rows += len(chunk)
            if strategy == "msg_simulation":
                chunk = chunk[truthy(chunk["simulation_challenge"])].copy()
            sim_rows += len(chunk)
            chunk["collision_energies"] = chunk["collision_energy"].map(format_collision_energy)
            if strategy != "msg_all":
                chunk = chunk.dropna(subset=["collision_energies"])
            chunk["collision_energies"] = chunk["collision_energies"].fillna("['nan']")
            selected_rows += len(chunk)
            if chunk["identifier"].isna().any():
                raise ValueError("Missing spectrum identifier")
            chunk = chunk[~chunk["identifier"].astype(str).isin(seen)].drop_duplicates("identifier").copy()
            identifiers = chunk["identifier"].astype(str)
            if identifiers.str.contains(r"[/\x00\n\r]", regex=True).any():
                raise ValueError("Invalid spectrum identifier")
            seen.update(identifiers)
            if h5 is not None:
                for row in chunk.itertuples(index=False):
                    ds = h5.create_dataset(f"{row.identifier}.ms", (1,), dtype=h5py.string_dtype("utf-8"))
                    ds[0] = spectrum_text(row)
            labels, splits = source_chunk_frames(chunk)
            label_chunks.append(labels)
            split_chunks.append(splits)

    if not label_chunks:
        raise ValueError("Source table contains no rows")
    labels = pd.concat(label_chunks, ignore_index=True)
    splits = pd.concat(split_chunks, ignore_index=True)
    labels.attrs["source_rows"] = source_rows
    labels.attrs["simulation_rows_before_known_ce_filter"] = sim_rows if strategy == "msg_simulation" else None
    labels.attrs["duplicate_selected_rows"] = selected_rows - len(labels)
    return labels, splits


def source_chunk_frames(chunk):
    return (
        pd.DataFrame({
            "dataset": "MassSpecGym", "spec": chunk["identifier"].astype(str),
            "ionization": chunk["adduct"], "formula": chunk["formula"],
            "smiles": chunk["smiles"], "inchikey": chunk["inchikey"],
            "instrument": chunk["instrument_type"],
            "collision_energies": chunk["collision_energies"],
            "precursor": chunk["precursor_mz"], "collision_imputed": False,
        }),
        pd.DataFrame({"name": chunk["identifier"].astype(str), "split": chunk["fold"].map(normalize_fold)}),
    )


def read_simulation_source(source_table: str, chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible entry point for existing callers."""
    return read_source(source_table, chunksize)


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


def build_dataset(args):
    output = args.output_dataset
    source = args.source_dataset
    if source is not None:
        resource_dir = (source / "labels.tsv").resolve().parent
        if resource_dir.resolve() == output.resolve():
            raise ValueError("Resolved source resources and output must be different directories")
        if not (resource_dir / "spec_files.hdf5").exists():
            raise FileNotFoundError(resource_dir / "spec_files.hdf5")
        for name in args.candidate_file or []:
            if not (resource_dir / name).is_file():
                raise FileNotFoundError(resource_dir / name)
    else:
        resource_dir = None

    # Refuse existing outputs before reading/downloading a potentially large TSV.
    for name in [
        "labels.tsv", "splits/split.tsv", "spec_files.hdf5", "labels_debug.tsv",
        "splits/split_debug.tsv", "spec_files_debug.hdf5",
        "retrieval/cands_df_test_formula_debug.tsv",
        *(args.candidate_file or DEFAULT_CANDIDATES), "subformulae/no_subform.hdf5",
    ]:
        target = output / name
        if (target.exists() or target.is_symlink()) and not args.overwrite:
            raise FileExistsError(f"{target} exists. Re-run with --overwrite to replace it.")
    for name in ["subformulae/no_subform.hdf5", *(args.candidate_file or DEFAULT_CANDIDATES), "magma_outputs"]:
        target = output / name
        if (target.exists() or target.is_symlink()) and (resource_dir is None or not (resource_dir / name).exists()):
            raise ValueError(f"Cannot retain stale resource {target}; use a clean output directory or supply --source-dataset.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".msg-build-", dir=output.parent) as staging:
        spectra_h5 = Path(staging) / "spec_files.hdf5" if source is None else None
        labels, splits = read_source(args.source_table, args.chunksize, args.strategy, spectra_h5)
        if labels.empty:
            raise ValueError(f"No spectra selected for {args.strategy}")
        audit = {
            "strategy": args.strategy, "source_table": str(args.source_table),
            "source_rows": int(labels.attrs["source_rows"]),
            "simulation_rows_before_known_ce_filter": labels.attrs["simulation_rows_before_known_ce_filter"],
            "duplicate_selected_rows": int(labels.attrs["duplicate_selected_rows"]),
            "output_rows": int(len(labels)), "output_unique_specs": int(labels["spec"].nunique()),
            "unknown_ce_rows": int((labels["collision_energies"] == "['nan']").sum()),
            "split_counts": {str(k): int(v) for k, v in splits["split"].value_counts().items()},
            "spectra_origin": "source_dataset" if source else "source_table",
            "wrote_output_dataset": True,
        }
        audit["imputed_check"] = assert_no_imputed(labels, [])
        if source is not None:
            audit["current_msg"] = current_msg_matches(labels, source)
            with h5py.File(resource_dir / "spec_files.hdf5", "r") as h5:
                missing = [spec for spec in labels["spec"] if spec not in h5 and f"{spec}.ms" not in h5]
            if missing:
                raise ValueError(f"Source HDF5 is missing {len(missing)} selected spectra: {missing[:5]}")

        output.mkdir(parents=True, exist_ok=True)
        write_tsv(labels, output / "labels.tsv", overwrite=args.overwrite)
        write_tsv(splits, output / "splits/split.tsv", overwrite=args.overwrite)
        if spectra_h5 is not None:
            replace_path(output / "spec_files.hdf5", overwrite=args.overwrite)
            os.replace(spectra_h5, output / "spec_files.hdf5")
        else:
            symlink_or_copy(resource_dir / "spec_files.hdf5", output / "spec_files.hdf5", args.overwrite)

        audit["linked_resources"] = {}
        audit["pending_resources"] = []
        for name in ["magma_outputs", "spec_files", "spec_files_w_eV"]:
            if resource_dir is not None:
                audit["linked_resources"][name] = symlink_or_copy(resource_dir / name, output / name, args.overwrite)
        if resource_dir is not None and (resource_dir / "subformulae/no_subform.hdf5").is_file():
            audit["subformulae"] = filter_subformulae(source, output, labels, args.overwrite)
        else:
            audit["pending_resources"].append("subformulae/no_subform.hdf5")

        candidate_frames = {}
        for name in args.candidate_file or DEFAULT_CANDIDATES:
            if resource_dir is not None and (resource_dir / name).is_file():
                candidate_frames[name] = filter_candidate_table(
                    resource_dir / name, output / name, set(labels["spec"]), args.overwrite,
                )
            else:
                audit["pending_resources"].append(name)
        formula_candidates = candidate_frames.get("retrieval/cands_df_test_formula_256.tsv")
        if formula_candidates is None:
            formula_candidates = next(iter(candidate_frames.values()), pd.DataFrame(columns=["spec"]))
        audit["debug"] = write_debug_files(
            labels, splits, output, formula_candidates, args.debug_train, args.debug_val,
            args.debug_test, args.overwrite,
        )
        audit["debug_spec_files"] = copy_debug_spec_files(output, output, args.overwrite)
        audit["imputed_check"] = assert_no_imputed(labels, candidate_frames.values())
    return audit


def main():
    args = get_args()
    if args.ensure_debug_hdf5:
        audit = {"debug_spec_files": copy_debug_spec_files(
            args.source_dataset, args.output_dataset, args.overwrite,
        )}
    else:
        audit = build_dataset(args)
    if args.audit_out is not None:
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        args.audit_out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
