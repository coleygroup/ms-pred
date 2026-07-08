#!/usr/bin/env python3
"""Run MassSpecGym simulation-challenge retrieval evaluation for ICEBERG."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


# Edit this block when rerunning retrieval.
DATASET = "msg_simulation"
GEN_CHECKPOINT = Path("results/iceberg_msg_simulation/split_rnd1/ckpt/gen/best.ckpt")
INTEN_CHECKPOINT = Path("results/iceberg_msg_simulation/split_rnd1/ckpt/inten_contr/best.ckpt")
SAVE_ROOT = Path("results/iceberg_msg_simulation/split_rnd1")

PYTHON_BIN = "python"
# None means inherit Slurm's GPU assignment. For local runs, this falls back to
# DEFAULT_LOCAL_CUDA_DEVICES below. Set to "0,1" here to force local devices.
CUDA_DEVICES = None
DEFAULT_LOCAL_CUDA_DEVICES = "0,1"
NUM_GPU_WORKERS = 4
# None means use SLURM_CPUS_PER_TASK on Slurm, otherwise
# DEFAULT_LOCAL_NUM_CPU_WORKERS. Set to an int here to force a fixed value.
NUM_CPU_WORKERS = None
DEFAULT_LOCAL_NUM_CPU_WORKERS = 128
MCES_NUM_WORKERS = None
DEFAULT_MCES_NUM_WORKERS = 32
BATCH_SIZE = 64
NUM_BINS = 150000
UPPER_LIMIT = 1500
MAX_NODES = 100

RETRIEVAL_ENTRIES = [
    {
        "split": "test_formula",
        "labels": Path(f"data/spec_datasets/{DATASET}/retrieval/cands_df_test_formula_256.tsv"),
        "strict_sanitize": False,
    },
    {
        "split": "test_mass",
        "labels": Path(f"data/spec_datasets/{DATASET}/retrieval/cands_df_test_mass_256.tsv"),
        "strict_sanitize": True,
    },
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["test_formula", "test_mass", "all"],
        default="all",
        help="Run one retrieval split or both splits.",
    )
    return parser.parse_args()


def run(cmd: str) -> None:
    print(cmd + "\n")
    subprocess.run(cmd, shell=True, check=True)


def get_cuda_devices() -> str | None:
    if CUDA_DEVICES is not None:
        return CUDA_DEVICES
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    if "SLURM_JOB_ID" in os.environ:
        return None
    return DEFAULT_LOCAL_CUDA_DEVICES


def get_num_cpu_workers() -> int:
    if NUM_CPU_WORKERS is not None:
        return NUM_CPU_WORKERS
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return DEFAULT_LOCAL_NUM_CPU_WORKERS


def get_mces_num_workers(num_cpu_workers: int) -> int:
    if MCES_NUM_WORKERS is not None:
        return MCES_NUM_WORKERS
    return min(num_cpu_workers, DEFAULT_MCES_NUM_WORKERS)


args = get_args()
pred_file = "src/ms_pred/iceberg/predict_smis.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_benchmark.py"
mces_file = "src/ms_pred/retrieval/add_top1_mces.py"
bootstrap_file = "analysis/msg_bootstrap.py"
cuda_devices = get_cuda_devices()
num_cpu_workers = get_num_cpu_workers()
mces_num_workers = get_mces_num_workers(num_cpu_workers)
device_str = f"CUDA_VISIBLE_DEVICES={cuda_devices} " if cuda_devices is not None else ""
print(f"CUDA_VISIBLE_DEVICES={cuda_devices if cuda_devices is not None else '<slurm default>'}")
print(f"NUM_CPU_WORKERS={num_cpu_workers}")
print(f"MCES_NUM_WORKERS={mces_num_workers}")

entries = [
    entry
    for entry in RETRIEVAL_ENTRIES
    if args.split == "all" or entry["split"] == args.split
]

for entry in entries:
    labels = Path(entry["labels"])
    if not labels.exists():
        raise FileNotFoundError(f"Missing retrieval candidate file: {labels}")
    if not GEN_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing generator checkpoint: {GEN_CHECKPOINT}")
    if not INTEN_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing intensity checkpoint: {INTEN_CHECKPOINT}")

    save_dir = SAVE_ROOT / f"retrieval_{DATASET}_{entry['split']}_256"
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_path = save_dir / "preds.hdf5"

    predict_cmd = f"""{device_str}{PYTHON_BIN} {pred_file} \\
    --batch-size {BATCH_SIZE} \\
    --dataset-name {DATASET} \\
    --sparse-out \\
    --sparse-k 100 \\
    --max-nodes {MAX_NODES} \\
    --split-name {entry['split']}.tsv \\
    --gen-checkpoint {GEN_CHECKPOINT} \\
    --inten-checkpoint {INTEN_CHECKPOINT} \\
    --save-dir {save_dir} \\
    --dataset-labels {labels} \\
    --num-cpu-workers {num_cpu_workers} \\
    --num-gpu-workers {NUM_GPU_WORKERS} \\
    --num-bins {NUM_BINS} \\
    --upper-limit {UPPER_LIMIT} \\
    --gpu \\
    --adduct-shift \\
    --binned-out"""
    if entry["strict_sanitize"]:
        predict_cmd += " \\\n    --strict-sanitize"
    run(predict_cmd)

    retrieve_cmd = f"""{PYTHON_BIN} {retrieve_file} \\
    --dataset {DATASET} \\
    --full-labels {labels} \\
    --formula-dir-name no_subform.hdf5 \\
    --pred-file {pred_path} \\
    --num-bins {NUM_BINS} \\
    --upper-limit {UPPER_LIMIT} \\
    --dist-fn entropy \\
    --num-cpu-workers {num_cpu_workers}"""
    run(retrieve_cmd)

    mces_cmd = f"""{PYTHON_BIN} {mces_file} \\
    --path {save_dir / "rerank_eval_entropy.yaml"} \\
    --num-workers {mces_num_workers}"""
    run(mces_cmd)

    bootstrap_cmd = f"""{PYTHON_BIN} {bootstrap_file} \\
    --path {save_dir / "rerank_eval_entropy.yaml"} \\
    --outfile {save_dir / "rerank_eval_entropy_bootstrap.yaml"} \\
    --dist-fn entropy"""
    run(bootstrap_cmd)
