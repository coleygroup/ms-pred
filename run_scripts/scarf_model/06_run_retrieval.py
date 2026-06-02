import yaml
from pathlib import Path
import subprocess

pred_file = "src/ms_pred/scarf_pred/predict_smis.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_binned.py"
subform_name = "no_subform"
devices = ",".join(["0"])
max_nodes = 300
dist = "cos"

test_entries = [

    # Rnd1

    {"dataset": "nist20",
     "train_split": "split_1_rnd1",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "nist20",
     "train_split": "split_1_rnd1",
     "test_split": "split_1_500",
     "max_k": None},

    {"dataset": "nist20",
     "res_dir": None,
     "train_split": "split_1_rnd1",
     "test_split": "split_1_500",
     "max_k": 50},

    # Rnd2

    {"dataset": "nist20",
     "train_split": "split_1_rnd2",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "nist20",
     "train_split": "split_1_rnd2",
     "test_split": "split_1_500",
     "max_k": None},

    {"dataset": "nist20",
     "res_dir": None,
     "train_split": "split_1_rnd2",
     "test_split": "split_1_500",
     "max_k": 50},

    # Rnd3

    {"dataset": "nist20",
     "train_split": "split_1_rnd3",
     "test_split": "split_1",
     "max_k": 50},

    {"dataset": "nist20",
     "train_split": "split_1_rnd3",
     "test_split": "split_1_500",
     "max_k": None},

    {"dataset": "nist20",
     "res_dir": None,
     "train_split": "split_1_rnd3",
     "test_split": "split_1_500",
     "max_k": 50},

]



for test_entry in test_entries:
    dataset = test_entry['dataset']
    train_split =  test_entry['train_split']
    split = test_entry['test_split']
    maxk = test_entry['max_k']
    inten_dir = test_entry.get("res_dir", None)

    if inten_dir is None:
        inten_dir = f"results/scarf_inten_{dataset}"

    inten_dir = Path(inten_dir)

    inten_model =  inten_dir / train_split  / "version_0/best.ckpt"
    if not inten_model.exists(): 
        print(f"Could not find model {model}; skipping\n: {json.dumps(test_entry, indent=1)}")
        continue

    labels = f"retrieval/cands_df_{split}_{maxk}.tsv"

    save_dir = inten_model.parent.parent / f"retrieval_{dataset}_{split}_{maxk}"
    save_dir.mkdir(exist_ok=True)

    args = yaml.safe_load(open(inten_model.parent.parent / "args.yaml", "r"))

    form_folder = Path(args["formula_folder"])
    gen_model = form_folder.parent / "version_0/best.ckpt"

    labels = f"retrieval/cands_df_{split}_{maxk}.tsv"
    save_dir = save_dir
    save_dir.mkdir(exist_ok=True)
    cmd = f"""python {pred_file} \\
    --batch-size 32  \\
    --dataset-name {dataset} \\
    --sparse-out \\
    --sparse-k 100 \\
    --max-nodes {max_nodes} \\
    --split-name {split}.tsv   \\
    --gen-checkpoint {gen_model} \\
    --inten-checkpoint {inten_model} \\
    --save-dir {save_dir} \\
    --dataset-labels {labels} \\
    --binned-out \\
    """
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    # Run retrieval
    cmd = f"""python {retrieve_file} \\
    --dataset {dataset} \\
    --formula-dir-name {subform_name} \\
    --binned-pred-file {save_dir / 'binned_preds.p'} \\
    --dist-fn {dist} \\
    """

    print(cmd + "\n")
    subprocess.run(cmd, shell=True)
