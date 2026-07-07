#!/usr/bin/env python3
"""Impute missing MassSpecGym collision energies with a trained ICEBERG model.

This script:
  1. Selects MassSpecGym rows whose collision_energies are missing.
  2. Builds a candidate label TSV with a 5..100 NCE grid, step 5.
  3. Runs src/ms_pred/iceberg/predict_smis.py for those candidates.
  4. Scores each predicted collision-energy spectrum against the experimental
     spectrum by entropy similarity and writes the best pseudo label.

Run from the repository root. The default checkpoints match the msg_known_ce
training pipeline in run_scripts/iceberg/msg_all.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

import ms_pred.common as common


PRED_DB_CACHE: dict[tuple[str, int, int], common.PredSpecDB] = {}


# Edit this block when rerunning the imputation job.
DATASET_DIR = Path("data/spec_datasets/msg_all")
LABELS_FILE = DATASET_DIR / "labels.tsv"
SPEC_FILE = DATASET_DIR / "spec_files.hdf5"
SAVE_DIR = Path("results/iceberg_msg_known_ce/iceberg_ce_imputation")

GEN_CHECKPOINT = Path("results/iceberg_msg_known_ce/split_rnd1/ckpt/gen/best.ckpt")
INTEN_CHECKPOINT = Path("results/iceberg_msg_known_ce/split_rnd1/ckpt/inten/best.ckpt")

PYTHON_BIN = sys.executable
# None means inherit Slurm's GPU assignment. For local runs, this falls back to
# DEFAULT_LOCAL_CUDA_DEVICES below.
CUDA_DEVICES = None
DEFAULT_LOCAL_CUDA_DEVICES = "0,1"

# ICEBERG prediction/scoring settings. These should rarely need edits.
NCE_GRID = range(5, 151, 5)
INSTRUMENT_FALLBACK = "Orbitrap"
SKIP_UNSUPPORTED_INSTRUMENT = False
BATCH_SIZE = 64
NUM_GPU_WORKERS = 4
# None means use SLURM_CPUS_PER_TASK on Slurm, otherwise
# DEFAULT_LOCAL_NUM_CPU_WORKERS.
NUM_CPU_WORKERS = None
DEFAULT_LOCAL_NUM_CPU_WORKERS = 64
NUM_H5_CHUNKS = 1
SPARSE_K = 100
MAX_NODES = 100
THRESHOLD = 0.0
NUM_BINS = 150000
UPPER_LIMIT = 1500
MAX_REAL_PEAKS = 20
REAL_INTENSITY_THRESHOLD = 0.05

def get_config() -> SimpleNamespace:
    return SimpleNamespace(
        dataset_dir=DATASET_DIR,
        labels_file=LABELS_FILE,
        spec_file=SPEC_FILE,
        save_dir=SAVE_DIR,
        gen_checkpoint=GEN_CHECKPOINT,
        inten_checkpoint=INTEN_CHECKPOINT,
        python_bin=PYTHON_BIN,
        cuda_devices=CUDA_DEVICES,
        default_local_cuda_devices=DEFAULT_LOCAL_CUDA_DEVICES,
        instrument_fallback=INSTRUMENT_FALLBACK,
        skip_unsupported_instrument=SKIP_UNSUPPORTED_INSTRUMENT,
        batch_size=BATCH_SIZE,
        num_gpu_workers=NUM_GPU_WORKERS,
        num_cpu_workers=NUM_CPU_WORKERS,
        default_local_num_cpu_workers=DEFAULT_LOCAL_NUM_CPU_WORKERS,
        num_h5_chunks=NUM_H5_CHUNKS,
        sparse_k=SPARSE_K,
        max_nodes=MAX_NODES,
        threshold=THRESHOLD,
        num_bins=NUM_BINS,
        upper_limit=UPPER_LIMIT,
        max_real_peaks=MAX_REAL_PEAKS,
        real_intensity_threshold=REAL_INTENSITY_THRESHOLD,
        nce_grid=list(NCE_GRID),
    )


def setup_logging(save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(save_dir / "impute_missing_collision_energies.log"),
        ],
        force=True,
    )


def resolve_dataset_dir(dataset_dir: str) -> Path:
    return Path(dataset_dir)


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


def precursor_for_row(row: pd.Series) -> float:
    if "precursor" in row and not pd.isna(row["precursor"]):
        return float(row["precursor"])
    return common.mass_from_smi(row["smiles"]) + common.ion2mass[row["ionization"]]


def nce_grid_to_unique_evs(nces: list[int], precursor_mz: float) -> tuple[list[int], dict[int, int]]:
    ev_to_nce: dict[int, int] = {}
    for nce in nces:
        ev = int(common.nce_to_ev(int(nce), precursor_mz))
        ev_to_nce.setdefault(ev, int(nce))
    evs = sorted(ev_to_nce)
    return evs, ev_to_nce


def build_candidate_table(args: SimpleNamespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    labels_file = Path(args.labels_file) if args.labels_file else dataset_dir / "labels.tsv"
    labels = pd.read_csv(labels_file, sep="\t")

    missing_mask = labels["collision_energies"].map(has_missing_collision_energy)
    missing = labels[missing_mask].copy()

    nces = list(args.nce_grid)
    candidate_rows = []
    skipped_rows = []
    for _, row in missing.iterrows():
        spec = row["spec"]
        ion = row["ionization"]
        instrument = row["instrument"] if "instrument" in row else "Orbitrap"
        try:
            if ion not in common.ion2mass:
                raise ValueError(f"unsupported ionization {ion}")
            if pd.isna(instrument) or instrument not in common.instrument2onehot_pos:
                if args.skip_unsupported_instrument:
                    raise ValueError(f"unsupported instrument {instrument}")
                instrument = args.instrument_fallback
                if instrument not in common.instrument2onehot_pos:
                    raise ValueError(f"unsupported fallback instrument {instrument}")
            precursor_mz = precursor_for_row(row)
            evs, ev_to_nce = nce_grid_to_unique_evs(nces, precursor_mz)
            if len(evs) == 0:
                raise ValueError("empty NCE grid after conversion to eV")
            candidate_rows.append(
                {
                    "dataset": row.get("dataset", "msg"),
                    "spec": spec,
                    "ionization": ion,
                    "formula": row.get("formula", ""),
                    "smiles": row["smiles"],
                    "inchikey": row.get("inchikey", ""),
                    "instrument": instrument,
                    "source_instrument": row["instrument"] if "instrument" in row else "",
                    "collision_energies": evs,
                    "precursor": precursor_mz,
                    "candidate_nces": nces,
                    "ev_to_nce": json.dumps(ev_to_nce, sort_keys=True),
                }
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "spec": spec,
                    "status": "skipped_prepare",
                    "error": str(exc),
                }
            )

    skipped_df = pd.DataFrame(skipped_rows, columns=["spec", "status", "error"])
    return pd.DataFrame(candidate_rows), skipped_df


def prediction_done(pred_path: Path, num_h5_chunks: int) -> bool:
    if num_h5_chunks <= 1:
        return pred_path.exists()
    return all(
        (pred_path.parent / f"{pred_path.stem}_chunk_{idx}{pred_path.suffix}").exists()
        for idx in range(num_h5_chunks)
    )


def get_cuda_devices(args: SimpleNamespace) -> str | None:
    if args.cuda_devices is not None:
        return args.cuda_devices
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    if "SLURM_JOB_ID" in os.environ:
        return None
    return args.default_local_cuda_devices


def get_num_cpu_workers(args: SimpleNamespace) -> int:
    if args.num_cpu_workers is not None:
        return int(args.num_cpu_workers)
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return int(args.default_local_num_cpu_workers)


def run_iceberg_predictions(args: SimpleNamespace, candidate_path: Path, pred_path: Path) -> None:
    if prediction_done(pred_path, args.num_h5_chunks):
        logging.info("Prediction output already exists, skipping ICEBERG run: %s", pred_path)
        return

    cmd = f"""{args.python_bin} src/ms_pred/iceberg/predict_smis.py \\
    --batch-size {args.batch_size} \\
    --num-gpu-workers {args.num_gpu_workers} \\
    --num-cpu-workers {args.num_cpu_workers} \\
    --dataset-labels {candidate_path} \\
    --sparse-out \\
    --sparse-k {args.sparse_k} \\
    --max-nodes {args.max_nodes} \\
    --threshold {args.threshold} \\
    --gen-checkpoint {args.gen_checkpoint} \\
    --inten-checkpoint {args.inten_checkpoint} \\
    --save-dir {pred_path.parent} \\
    --out-name {pred_path.name} \\
    --num-h5-chunks {args.num_h5_chunks} \\
    --adduct-shift \\
    --gpu
    """
    env = os.environ.copy()
    cuda_devices = get_cuda_devices(args)
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    logging.info("Running ICEBERG prediction:\n%s", cmd)
    subprocess.run(cmd, check=True, env=env, shell=True)


def read_experimental_spectrum(spec_source: Path, spec: str) -> common.CompositeMassSpec:
    if spec_source.is_file() and spec_source.suffix in {".hdf5", ".h5"}:
        h5_obj = common.HDF5Dataset(spec_source)
        try:
            key = spec if spec in h5_obj.h5_obj else f"{spec}.ms"
            lines = h5_obj.read_str(key).splitlines()
        finally:
            h5_obj.close()
        _, composite = common.parse_spectra(lines)
        return composite

    if spec_source.is_dir():
        candidates = [spec_source / spec, spec_source / f"{spec}.ms"]
        for candidate in candidates:
            if candidate.exists():
                _, composite = common.parse_spectra(candidate)
                return composite
    raise FileNotFoundError(f"Could not find experimental spectrum for {spec} in {spec_source}")


def flatten_pred_dict(pred_dict: dict[str, Any], has_remark: bool) -> dict[str, Any]:
    if not has_remark:
        return pred_dict

    flat_pred_dict = {}
    for remark, remark_pred_dict in pred_dict.items():
        for collision_key, pred_spec in remark_pred_dict.items():
            if collision_key in flat_pred_dict:
                raise ValueError(f"duplicate predicted collision energy {collision_key} across remarks")
            flat_pred_dict[collision_key] = pred_spec
    return flat_pred_dict


def score_one_row(
    row: dict[str, Any] | pd.Series,
    pred_db: common.PredSpecDB,
    spec_source: Path,
    args: SimpleNamespace,
) -> dict[str, Any]:
    spec = row["spec"]
    pred_name = f"pred_{spec}"
    precursor_mz = float(row["precursor"])
    try:
        pred_dict, has_remark = pred_db.read_from_name(pred_name)
        pred_dict = flatten_pred_dict(pred_dict, has_remark)
        if len(pred_dict) == 0:
            raise KeyError(f"missing predicted spectrum {pred_name}")

        pred_comp = common.CompositeMassSpec(pred_dict)
        pred_comp.ensure_binned_spectrum(
            mass_upper_limit=args.upper_limit,
            num_bins=args.num_bins,
            pool_fn="add",
        )

        real_comp = read_experimental_spectrum(spec_source, spec)
        real_comp.process_spec_file(
            parentmass=precursor_mz,
            denoise=False,
            max_num_inten=args.max_real_peaks,
            inten_thresh=args.real_intensity_threshold,
        )
        real_comp.ensure_binned_spectrum(
            mass_upper_limit=args.upper_limit,
            num_bins=args.num_bins,
            pool_fn="max",
        )
        if len(real_comp) == 1:
            real_ms = next(iter(real_comp.values()))
        else:
            real_ms = real_comp.merge_spectra(merge_method="sum")
            real_ms.ensure_binned_spectrum(
                mass_upper_limit=args.upper_limit,
                num_bins=args.num_bins,
                pool_fn="max",
            )

        sim, best_ev = pred_comp.similarity(
            real_ms,
            ignore_mass=precursor_mz - 1,
            merge_method="unknown",
            metric="entropy",
            return_ce=True,
        )
        ev_to_nce = {int(k): int(v) for k, v in json.loads(row["ev_to_nce"]).items()}
        best_ev_int = int(round(float(best_ev)))
        return {
            "spec": spec,
            "best_collision_energy": best_ev_int,
            "best_nce": ev_to_nce.get(best_ev_int, np.nan),
            "entropy_similarity": float(sim),
            "entropy_distance": float(1 - sim),
            "num_candidate_collision_energies": len(parse_collision_list(row["collision_energies"])),
            "candidate_collision_energies": row["collision_energies"],
            "candidate_nces": row["candidate_nces"],
            "status": "scored",
            "error": "",
        }
    except Exception as exc:
        return {
            "spec": spec,
            "best_collision_energy": np.nan,
            "best_nce": np.nan,
            "entropy_similarity": np.nan,
            "entropy_distance": np.nan,
            "num_candidate_collision_energies": len(parse_collision_list(row["collision_energies"])),
            "candidate_collision_energies": row["collision_energies"],
            "candidate_nces": row["candidate_nces"],
            "status": "failed_score",
            "error": str(exc),
        }


def score_one_row_from_paths(
    row: dict[str, Any],
    pred_path: str,
    num_h5_chunks: int,
    spec_source: str,
    args: SimpleNamespace,
) -> dict[str, Any]:
    cache_key = (pred_path, num_h5_chunks, os.getpid())
    pred_db = PRED_DB_CACHE.get(cache_key)
    if pred_db is None:
        pred_db = common.PredSpecDB(Path(pred_path), num_h5s=num_h5_chunks)
        PRED_DB_CACHE[cache_key] = pred_db
    return score_one_row(row, pred_db, Path(spec_source), args)


def score_predictions(args: SimpleNamespace, candidate_df: pd.DataFrame, pred_path: Path) -> pd.DataFrame:
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    spec_source = Path(args.spec_file) if args.spec_file else dataset_dir / "spec_files.hdf5"
    records = candidate_df.to_dict("records")
    if len(records) == 0:
        return pd.DataFrame()

    score_fn = partial(
        score_one_row_from_paths,
        pred_path=str(pred_path),
        num_h5_chunks=args.num_h5_chunks,
        spec_source=str(spec_source),
        args=args,
    )
    rows = common.chunked_parallel(
        records,
        score_fn,
        chunks=1000,
        max_cpu=args.num_cpu_workers,
        task_name="Scoring imputed CEs",
    )
    return pd.DataFrame(rows)


def read_optional_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    args = get_config()
    args.num_cpu_workers = get_num_cpu_workers(args)
    save_dir = Path(args.save_dir)
    setup_logging(save_dir)

    candidate_path = save_dir / "missing_ce_candidates.tsv"
    skipped_path = save_dir / "missing_ce_skipped.tsv"
    pred_path = save_dir / "preds.hdf5"
    output_path = save_dir / "imputed_collision_energies.tsv"

    candidate_df, skipped_df = build_candidate_table(args)
    candidate_df.to_csv(candidate_path, sep="\t", index=False)
    skipped_df.to_csv(skipped_path, sep="\t", index=False)
    logging.info("Wrote %d candidate rows to %s", len(candidate_df), candidate_path)
    logging.info("Wrote %d skipped rows to %s", len(skipped_df), skipped_path)

    run_iceberg_predictions(args, candidate_path, pred_path)

    scored = score_predictions(args, candidate_df, pred_path)
    skipped = read_optional_tsv(skipped_path)
    if len(skipped) > 0:
        scored = pd.concat([scored, skipped], ignore_index=True, sort=False)
    scored.to_csv(output_path, sep="\t", index=False)
    logging.info("Wrote imputation table to %s", output_path)
    logging.info("Status counts: %s", scored["status"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
