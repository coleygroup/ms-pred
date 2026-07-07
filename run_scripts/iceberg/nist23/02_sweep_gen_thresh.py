""" Sweep gen thresh """
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import yaml

batch_size = 64
list_devices = [0, 1]
workers = 32
devices = ",".join([str(_) for _ in list_devices])
python_file = "src/ms_pred/iceberg/predict_gen.py"
max_nodes = [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000]
gpu_workers = [len(list_devices) * _ for _ in [8, 8, 8, 8, 8, 4, 2, 1, 1, 1]]
subform_name = "magma_subform_50.hdf5"
debug = False

res_entries = [
    {"folder": "results/iceberg_nist23/split_1_rnd1/",
     "dataset": "nist23",
     "test_split": "split_1"},

    # {"folder": "results/iceberg_nist23/split_1_rnd2/",
    #  "dataset": "nist23",
    #  "test_split": "split_1"},

    # {"folder": "results/iceberg_nist23/split_1_rnd3/",
    #  "dataset": "nist23",
    #  "test_split": "split_1"},

    # {"folder": "results/iceberg_nist23/scaffold_1_rnd1/",
    #  "dataset": "nist23",
    #  "test_split": "scaffold_1"},

    # {"folder": "results/iceberg_nist23/scaffold_1_rnd2/",
    #  "dataset": "nist23",
    #  "test_split": "scaffold_1"},

    # {"folder": "results/iceberg_nist23/scaffold_1_rnd3/",
    #  "dataset": "nist23",
    #  "test_split": "scaffold_1"},
]

if debug:
    max_nodes = max_nodes[:3]


def prediction_finished(log_file: Path) -> bool:
    if not log_file.exists():
        return False
    return "Program finished in:" in log_file.read_text()


def cleanup_prediction_outputs(save_dir_temp: Path) -> None:
    for file_name in ["tree_preds.hdf5", "dag_gen_pred.log", "pred_eval.yaml", "pred_eval_grouped.tsv"]:
        file_path = save_dir_temp / file_name
        if file_path.exists():
            file_path.unlink()

    decoy_dir = save_dir_temp / "decoy_tree_preds"
    if decoy_dir.exists():
        shutil.rmtree(decoy_dir)


for res_entry in res_entries:
    res_folder = Path(res_entry["folder"])
    dataset = res_entry["dataset"]
    models = sorted(list((res_folder / "ckpt" / "gen").rglob("*.ckpt")))
    split = res_entry["test_split"]
    for model in models:
        save_dir_base = res_folder

        save_dir = save_dir_base / "inten_thresh_sweep"
        save_dir.mkdir(exist_ok=True)

        print(f"Saving inten sweep to: {save_dir}")

        pred_dir_h5s = []
        for max_node, gpu_worker in zip(max_nodes, gpu_workers):
            save_dir_temp = save_dir / str(max_node)
            save_dir_temp.mkdir(exist_ok=True)
            pred_h5 = save_dir_temp / "tree_preds.hdf5"
            pred_log = save_dir_temp / "dag_gen_pred.log"

            cmd = f"""python {python_file} \\
            --batch-size {batch_size} \\
            --dataset-name  {dataset} \\
            --split-name {split}.tsv \\
            --subset-datasets test_only  \\
            --checkpoint {model} \\
            --save-dir {save_dir_temp} \\
            --threshold 0  \\
            --max-nodes {max_node} \\
            --num-cpu-workers {workers} \\
            --num-gpu-workers {gpu_worker} \\
            --gpu
            """

            pred_dir_h5s.append(pred_h5)
            device_str = f"CUDA_VISIBLE_DEVICES={devices}"
            cmd = f"{device_str} {cmd}"
            if pred_h5.exists() and prediction_finished(pred_log):
                print(f"Skipping completed prediction: {save_dir_temp}")
            else:
                print(cmd + "\n")
                subprocess.run(cmd, shell=True, check=True)

        res_files = []
        for pred_h5 in pred_dir_h5s:
            pred_eval = pred_h5.parent / "pred_eval.yaml"
            analysis_cmd = f"""python analysis/form_pred_eval.py \\
                --dataset {dataset} \\
                --tree-pred-obj {pred_h5} \\
                --subform-name {subform_name}
            """
            res_files.append(pred_eval)
            print(analysis_cmd + "\n")
            subprocess.run(analysis_cmd, shell=True, check=True)

        new_entries = []
        for res_file in res_files:
            new_data = yaml.safe_load(open(res_file, "r"))
            thresh = res_file.parent.stem
            new_entry = {"nm_nodes": thresh}
            new_entry.update({k: v for k, v in new_data.items() if "avg" in k})
            new_entries.append(new_entry)

        df = pd.DataFrame(new_entries)
        df.to_csv(save_dir / "summary.tsv", sep="\t", index=None)
