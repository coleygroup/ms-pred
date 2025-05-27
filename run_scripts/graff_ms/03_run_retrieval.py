from pathlib import Path
import subprocess
import json

pred_file = "src/ms_pred/graff_ms/predict.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_benchmark.py"
subform_name = "no_subform"
devices = ",".join(["0"])
dist = "cos"
num_workers = 32

test_entries = [
    {"dataset": "nist20",
     "test_dataset": "nist20",
     "train_split": "split_1_rnd1",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "nist20",
     "train_split": "split_1_rnd2",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "nist20",
     "train_split": "split_1_rnd3",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "nist20",
     "train_split": "scaffold_1_rnd1",
     "test_split": "scaffold_1",
     "max_k": 50},

    {"dataset": "nist20",
     "train_split": "scaffold_1_rnd2",
     "test_split": "scaffold_1",
     "max_k": 50},

    {"dataset": "nist20",
     "train_split": "scaffold_1_rnd3",
     "test_split": "scaffold_1",
     "max_k": 50},

    {"dataset": "canopus_train_public",
     "train_split": "split_1_rnd1",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "canopus_train_public",
     "train_split": "split_1_rnd2",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "canopus_train_public",
     "train_split": "split_1_rnd3",
     "test_split": "split_1",
     "max_k": 50},
]


for test_entry in test_entries:
    dataset = test_entry['dataset']
    test_dataset = test_entry['test_dataset']
    train_split =  test_entry['train_split']
    split = test_entry['test_split']
    maxk = test_entry['max_k']

    res_folder = Path(f"results/graff_ms_baseline_{dataset}/")
    model =  res_folder / train_split / "version_0/best.ckpt"
    if not model.exists(): 
        print(f"Could not find model {model}; skipping\n: {json.dumps(test_entry, indent=1)}")
        continue

    save_dir = model.parent.parent
    if test_dataset != dataset:
        save_dir = save_dir / "cross_dataset" / test_dataset
 
    save_dir = save_dir / f"retrieval_{dataset}_{split}_{maxk}"
    save_dir.mkdir(exist_ok=True, parents=True)

    labels = f"retrieval/cands_df_{split}_{maxk}.tsv"
    cmd = f"""python {pred_file} \\
    --batch-size 32  \\
    --dataset-name {test_dataset} \\
    --sparse-out \\
    --sparse-k 100 \\
    --num-workers {num_workers} \\
    --split-name {split}.tsv   \\
    --checkpoint {model} \\
    --save-dir {save_dir} \\
    --dataset-labels {labels} \\
    --gpu"""
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    # Run retrieval
    cmd = f"""python {retrieve_file} \\
    --dataset {dataset} \\
    --formula-dir-name {subform_name}.hdf5 \\
    --pred-file {save_dir / 'binned_preds.hdf5'} \\
    --dist-fn cos \\
    --binned-pred \\
        """

    print(cmd + "\n")
    subprocess.run(cmd, shell=True)
