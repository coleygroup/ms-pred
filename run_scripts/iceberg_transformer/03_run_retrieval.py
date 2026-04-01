import yaml
from pathlib import Path
import subprocess
import json

pred_file = "src/ms_pred/iceberg_transformer/predict_smis_joint.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_benchmark.py"
subform_name = "no_subform"
devices = [0, 1]
vis_devices = ",".join([str(_) for _ in devices])
num_cpu_workers = 16
batch_size = 16
binned_out = False
pool_fn = "max"

test_entries = [
    # {"dataset": "nist20",
    #  "train_split": "split_1_rnd1",
    #  "test_split": "split_1",
    #  "max_k": 50},

    # {"dataset": "nist20",
    #  "train_split": "split_1_rnd2",
    #  "test_split": "split_1",
    #  "max_k": 50},
    
    # {"dataset": "nist20",
    #  "train_split": "split_1_rnd3",
    #  "test_split": "split_1",
    #  "max_k": 50},
    
    
    # {"dataset": "nist20",
    #  "train_split": "scaffold_1_rnd1",
    #  "test_split": "scaffold_1",
    #  "max_k": 50},
    
    # {"dataset": "nist20",
    #  "train_split": "scaffold_1_rnd2",
    #  "test_split": "scaffold_1",
    #  "max_k": 50},
    
    # {"dataset": "nist20",
    #  "train_split": "scaffold_1_rnd3",
    #  "test_split": "scaffold_1",
    #  "max_k": 50},

    {"dataset": "msg",
     "train_split": "split_rnd1",
     "test_split": "test_mass",
     "max_k": 256},

    # {"dataset": "msg",
    #  "train_split": "split_rnd1",
    #  "test_split": "test_formula",
    #  "max_k": 256},
]

pred_filename = "binned_preds.hdf5" if binned_out else "preds.hdf5"
for test_entry in test_entries:
    dataset = test_entry['dataset']
    train_split = test_entry['train_split']
    split = test_entry['test_split']
    maxk = test_entry['max_k']
    model_dir = Path(f"results/joint_train_{dataset}")
    joint_model = model_dir/train_split/"version_80/best.ckpt"
    if not joint_model.exists():
        print(f"Could not find model {joint_model}; skipping\n: {json.dumps(test_entry, indent=1)}")
        continue
    
    labels = f"data/spec_datasets/{dataset}/retrieval/cands_df_{split}_{maxk}.tsv"

    save_dir = joint_model.parent.parent / f"retrieval_{dataset}_{split}_{maxk}_decoy_4_threshold_0.5"
    save_dir.mkdir(exist_ok=True)
    num_gpu_workers = len(devices) * 7 if split == "test_mass" else len(devices) * 9

    cmd = f"""python {pred_file} \\
    --batch-size {batch_size} \\
    --dataset-name {dataset} \\
    --sparse-out \\
    --sparse-k 1000 \\
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

    # subprocess.run(cmd, shell=True)

    # # Run retrieval
    cmd = f"""python {retrieve_file} \\
    --dataset {dataset} \\
    --formula-dir-name {subform_name}.hdf5 \\
    --pred-file {save_dir / pred_filename} \\
    --dist-fn entropy \\
    --pool-fn {pool_fn} \\
    --full-labels {labels} \\
    --num-cpu-workers {num_cpu_workers} \\
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
    --pool-fn {pool_fn} \\
    --full-labels {labels} \\
    --num-cpu-workers {num_cpu_workers} \\
    """
    if binned_out:
        cmd += "--binned-pred"

    print(cmd + "\n")
    subprocess.run(cmd, shell=True)
