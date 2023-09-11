from pathlib import Path
import subprocess
import argparse

python_file = "src/ms_pred/dag_pred/predict_inten.py"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="nist20")
args = parser.parse_args()

dataset = args.dataset

# dataset = "canopus_train_public"  # canopus_train_public
# dataset = "nist20"  # canopus_train_public


devices = ",".join(["3"])
node_num = 100
res_folder = Path(f"results/dag_inten_{dataset}/")
base_formula_folder = Path(f"results/dag_{dataset}")
ckpts = res_folder.rglob("version_0/*.ckpt")
ckpts = sorted(ckpts)
valid_splits = ["scaffold_1"]
valid_splits = ["scaffold_1", "split_1"]

for model in ckpts:
    save_dir = model.parent.parent
    split = save_dir.name

    if split not in valid_splits:
        continue

    save_dir = save_dir / "preds"

    # Note: Must use preds_train_01
    magma_dag_folder = (
        base_formula_folder / split / f"preds_train_{node_num}/tree_preds"
    )
    print(magma_dag_folder)
    cmd = f"""python {python_file} \\
    --batch-size 32 \\
    --dataset-name {dataset} \\
    --split-name {split}.tsv \\
    --checkpoint {model} \\
    --save-dir {save_dir} \\
    --gpu \\
    --num-workers 0 \\
    --magma-dag-folder {magma_dag_folder} \\
    --subset-datasets test_only \\
    --binned-out
    """
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    # subprocess.run(cmd, shell=True)

    # Eval it
    out_binned = save_dir / "binned_preds.p"
    eval_cmd = f"""
    python analysis/spec_pred_eval.py \\
    --binned-pred-file {out_binned} \\
    --max-peaks 100 \\
    --min-inten 0 \\
    --formula-dir-name no_subform \\
    --dataset {dataset}  \\
    """
    print(eval_cmd)
    subprocess.run(eval_cmd, shell=True)
