from pathlib import Path
import subprocess
import json


pred_file = "src/ms_pred/massformer_pred/predict.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_benchmark.py"
subform_name = "no_subform"
devices = ",".join(["0"])
dist = "cos"
num_workers = 32

test_entries = [
    {
     "test_dataset": "nist20",
     "dataset": "nist20",
     "train_split": "split_1_rnd1",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "nist20",
     "test_dataset": "nist20",
     "train_split": "split_1_rnd2",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "nist20",
     "test_dataset": "nist20",
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
     "test_dataset": "canopus_train_public",
     "train_split": "split_1_rnd3",
     "test_split": "split_1",
     "max_k": 50},

    #{"dataset": "canopus_train_public",
    # "test_dataset": "casmi22",
    # "train_split": "split_1_rnd1",
    # "test_split": "all_split",
    # "max_k": None},

]


for test_entry in test_entries:
    dataset = test_entry['dataset']
    test_dataset = test_entry['test_dataset']
    train_split = test_entry['train_split']
    split = test_entry['test_split']
    maxk = test_entry['max_k']

    res_folder = Path(f"results/massformer_baseline_{dataset}/")
    model =  res_folder / train_split  / "version_0/best.ckpt"
    if not model.exists(): 
        print(f"Could not find model {model}; skipping\n: {json.dumps(test_entry, indent=1)}")
        continue

    labels = f"retrieval/cands_df_{split}_{maxk}.tsv"

    save_dir = model.parent.parent
    if test_dataset != dataset:
        save_dir = save_dir / "cross_dataset" / test_dataset

    save_dir = save_dir / f"retrieval_{dataset}_{split}_{maxk}"
    save_dir.mkdir(exist_ok=True, parents=True)


    cmd = f"""python {pred_file} \\
    --batch-size 32  \\
    --num-workers 16 \\
    --dataset-name {dataset} \\
    --sparse-out \\
    --sparse-k 100 \\
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
