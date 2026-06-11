import yaml
from pathlib import Path
import subprocess
import json
import math

pred_file = "src/ms_pred/GLACIER/predict_smis_joint.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_benchmark.py"
subform_name = "no_subform"
devices = [0, 1]
vis_devices = ",".join([str(_) for _ in devices])
total_num_cpu_workers = 64
batch_size = 64
binned_out = False
num_h5_chunks = 8
local_write_dir = "/tmp/ms_pred_h5"


def clear_prediction_outputs(save_dir: Path, pred_filename: str):
    stem = Path(pred_filename).stem
    suffix = Path(pred_filename).suffix
    patterns = [
        f"{stem}{suffix}",
        f"{stem}_shard*{suffix}",
        f"{stem}_chunk_*{suffix}",
        f"{stem}_shard*_chunk_*{suffix}",
    ]
    for pattern in patterns:
        for path in save_dir.glob(pattern):
            if path.is_file():
                path.unlink()

test_entries = [
    {"dataset": "nist20",
     "train_split": "split_1_rnd1",
     "test_split": "split_1",
     "max_k": 50, 
     "num_bins": 15000},

    {"dataset": "nist20",
     "train_split": "split_1_rnd2",
     "test_split": "split_1",
     "max_k": 50, 
     "num_bins": 15000},
    
    {"dataset": "nist20",
     "train_split": "split_1_rnd3",
     "test_split": "split_1",
     "max_k": 50, 
     "num_bins": 15000},
    
    {"dataset": "nist20",
     "train_split": "scaffold_1_rnd1",
     "test_split": "scaffold_1",
     "max_k": 50,
     "num_bins": 15000},
    
    {"dataset": "nist20",
     "train_split": "scaffold_1_rnd2",
     "test_split": "scaffold_1",
     "max_k": 50,
     "num_bins": 15000},
    
    {"dataset": "nist20",
     "train_split": "scaffold_1_rnd3",
     "test_split": "scaffold_1",
     "max_k": 50, 
     "num_bins": 15000},

    {"dataset": "msg",
     "train_split": "split_rnd1",
     "test_split": "test_mass",
     "max_k": 256, 
     "num_bins": 150000,},

    {"dataset": "msg",
     "train_split": "split_rnd1",
     "test_split": "test_formula",
     "max_k": 256, 
     "num_bins": 150000,},
]

pred_filename = "binned_preds.hdf5" if binned_out else "preds.hdf5"
for test_entry in test_entries:
    dataset = test_entry['dataset']
    train_split = test_entry['train_split']
    split = test_entry['test_split']
    maxk = test_entry['max_k']
    model_dir = Path(f"results/joint_train_{dataset}")
    num_bins = test_entry['num_bins']
    joint_model = model_dir/train_split/"version_1/best.ckpt"
    if not joint_model.exists():
        print(f"Could not find model {joint_model}; skipping\n: {json.dumps(test_entry, indent=1)}")
        continue
    
    labels = f"data/spec_datasets/{dataset}/retrieval/cands_df_{split}_{maxk}.tsv"

    save_dir = joint_model.parent.parent / f"retrieval_{dataset}_{split}_{maxk}"
    save_dir.mkdir(exist_ok=True)
    clear_prediction_outputs(save_dir, pred_filename)
    num_gpu_workers = len(devices)
    num_cpu_workers = max(1, math.ceil(total_num_cpu_workers / max(1, num_gpu_workers)))
    cmd = f"""python {pred_file} \\
    --batch-size {batch_size} \\
    --dataset-name {dataset} \\
    --sparse-out \\
    --sparse-k 100 \\
    --num-h5-chunks {num_h5_chunks} \
    --local-write-dir {local_write_dir} \
    --split-name {split}.tsv \\
    --checkpoint {joint_model} \\
    --save-dir {save_dir} \\
    --dataset-labels {labels} \\
    --num-cpu-workers {num_cpu_workers} \\
    --num-gpu-workers {num_gpu_workers} \\
    --gpu \\
    """
    if binned_out:
        cmd += "--binned-out"
    device_str = f"CUDA_VISIBLE_DEVICES={vis_devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    # # Run retrieval
    cmd = f"""python {retrieve_file} \\
    --dataset {dataset} \\
    --formula-dir-name {subform_name}.hdf5 \\
    --pred-file {save_dir / pred_filename} \\
    --dist-fn entropy \\
    --full-labels {labels} \\
    --num-cpu-workers {num_cpu_workers} \\
    --num-bins {num_bins} \\

    """
    if binned_out:
        cmd += "--binned-pred"

    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    cmd = f"""python {retrieve_file} \\
    --dataset {dataset} \\
    --formula-dir-name {subform_name}.hdf5 \\
    --pred-file {save_dir / pred_filename} \\
    --dist-fn cos \\
    --full-labels {labels} \\
    --num-cpu-workers {num_cpu_workers} \\
    --num-bins {num_bins} \\
    """
    if binned_out:
        cmd += "--binned-pred"

    print(cmd + "\n")
    subprocess.run(cmd, shell=True)
