import yaml
import json
import pandas as pd
import time
from pathlib import Path
import subprocess

python_file = "src/ms_pred/dag_pred/predict_smis.py"
dataset = "nist20"  # canopus_train_public
devices = ",".join(["3"])
node_num = 100
res_folder = Path(f"results/dag_inten_{dataset}/")
ckpts = res_folder.rglob("version_0/*.ckpt")
ckpts = sorted(ckpts)

dataset_labels = "sample_labels.tsv"
labels = Path(f"data/spec_datasets/") / dataset_labels
num_mols = len(pd.read_csv(labels, sep="\t"))

for model in ckpts:
    save_dir = model.parent.parent
    split = save_dir.name
    time_res = save_dir / "time_out.json"
    save_dir = save_dir / "preds_time"
    save_dir.mkdir(exist_ok=True)

    args = yaml.safe_load(open(model.parent.parent / "args.yaml", "r"))
    dag_folder = Path(args["magma_dag_folder"])
    gen_model = dag_folder.parent / "version_0/best.ckpt"

    # Note: Must use preds_train_01
    cmd = f"""python {python_file} \\
    --batch-size 1 \\
    --split-name {split}.tsv \\
    --gen-checkpoint {gen_model} \\
    --inten-checkpoint {model} \\
    --max-nodes {node_num} \\
    --sparse-out \\
    --sparse-k 100 \\
    --save-dir {save_dir} \\
    --num-workers 0 \\
    --subset-datasets none \\
    --dataset-labels {labels} \\
    --binned-out
    #--gpu \\
    #--dataset-name {dataset} \\
    """
    print(cmd + "\n")
    start = time.time()
    subprocess.run(cmd, shell=True)
    end = time.time()
    seconds = end - start
    out_dict = {"time (s)": seconds, "mols": num_mols}
    print(f"Time taken for preds {seconds}")
    json.dump(out_dict, open(time_res, "w"))
