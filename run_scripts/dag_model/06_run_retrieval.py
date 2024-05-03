import yaml
from pathlib import Path
import subprocess
import json

pred_file = "src/ms_pred/dag_pred/predict_smis.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_benchmark.py"
subform_name = "no_subform"
devices = ",".join(["1"])
max_nodes = 100
num_workers = 64
dist = "cos"
binned_out = True

test_entries = [
    {"dataset": "nist20",
     "train_split": "split_1_rnd1",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "canopus_train_public",
     "train_split": "split_1_rnd1",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "nist20",
     "train_split": "split_1_rnd2",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "canopus_train_public",
     "train_split": "split_1_rnd2",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "nist20",
     "train_split": "split_1_rnd3",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "canopus_train_public",
     "train_split": "split_1_rnd3",
     "test_split": "split_1",
     "max_k": 50},
]

pred_filename = "binned_preds.p" if binned_out else "preds.p"

for test_entry in test_entries:
    dataset = test_entry['dataset']
    train_split =  test_entry['train_split']
    split = test_entry['test_split']
    maxk = test_entry['max_k']
    inten_dir = Path(f"results/dag_inten_{dataset}")
    inten_model =  inten_dir / train_split  / "version_0/best.ckpt"
    if not inten_model.exists():
        print(f"Could not find model {inten_model}; skipping\n: {json.dumps(test_entry, indent=1)}")
        continue

    labels = f"data/spec_datasets/{dataset}/retrieval/cands_df_{split}_{maxk}.tsv"

    save_dir = inten_model.parent.parent / f"retrieval_{dataset}_{split}_{maxk}"
    save_dir.mkdir(exist_ok=True)

    args = yaml.safe_load(open(inten_model.parent.parent / "args.yaml", "r"))
    form_folder = Path(args["magma_dag_folder"])
    gen_model = form_folder.parent / "version_0/best.ckpt"

    save_dir = save_dir
    save_dir.mkdir(exist_ok=True)
    cmd = f"""python {pred_file} \\
    --num-workers {num_workers}  \\
    --dataset-name {dataset} \\
    --sparse-out \\
    --sparse-k 100 \\
    --max-nodes {max_nodes} \\
    --split-name {split}.tsv   \\
    --gen-checkpoint {gen_model} \\
    --inten-checkpoint {inten_model} \\
    --save-dir {save_dir} \\
    --dataset-labels {labels} \\
    """
    if binned_out:
        cmd += "--binned-out"
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    # Run retrieval and random baseline
    for dist_fn in [dist, 'random']:
        cmd = f"""python {retrieve_file} \\
        --dataset {dataset} \\
        --formula-dir-name {subform_name}.hdf5 \\
        --pred-file {save_dir / pred_filename} \\
        --dist-fn {dist_fn} \\
        """
        if binned_out:
            cmd += "--binned-pred"

        print(cmd + "\n")
        subprocess.run(cmd, shell=True)
