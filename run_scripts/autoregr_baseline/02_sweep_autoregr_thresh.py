import yaml
import pandas as pd
from pathlib import Path
import subprocess

devices = ",".join(["2"])
subform_name = "magma_subform_50"
max_nodes = [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000]

run_models = [
    {"dataset": "nist20", "dir_name":"split_1_rnd1", "test_split": "split_1"},
    {"dataset": "nist20", "dir_name":"split_1_rnd2", "test_split": "split_1"},
    {"dataset": "nist20", "dir_name":"split_1_rnd3", "test_split": "split_1"},
]


python_file = "src/ms_pred/autoregr_gen/predict.py"
for run_entry in run_models:
    dataset = run_entry['dataset']
    dir_name = run_entry['dir_name']
    split = run_entry['test_split']

    res_folder = Path(f"results/autoregr_{dataset}/{dir_name}")
    model = sorted(list(res_folder.rglob("version_0/*.ckpt")))
    if len(model) == 0: continue
    model = model[0]
    save_dir_base = model.parent.parent
    save_dir = save_dir_base / "inten_thresh_sweep"
    save_dir.mkdir(exist_ok=True)

    print(f"Saving inten sweep to: {save_dir}")

    pred_dir_folders = []
    for max_node in max_nodes:
        save_dir_temp = save_dir / str(max_node)
        save_dir_temp.mkdir(exist_ok=True)

        cmd = f"""python {python_file} \\
        --batch-size 128 \\
        --dataset-name  {dataset} \\
        --split-name {split}.tsv \\
        --subset-datasets test_only  \\
        --checkpoint {model} \\
        --save-dir {save_dir_temp} \\
        --num-workers 0 \\
        --max-nodes {max_node} \\
        --gpu"""

        pred_dir_folders.append(save_dir_temp / "form_preds")
        device_str = f"CUDA_VISIBLE_DEVICES={devices}"
        cmd = f"{device_str} {cmd}"
        print(cmd + "\n")
        subprocess.run(cmd, shell=True)

    res_files = []
    for pred_dir in pred_dir_folders:
        analysis_cmd = f"""python analysis/form_pred_eval.py \\
            --dataset {dataset} \\
            --tree-pred-folder {pred_dir} \\
            --subform-name {subform_name}
        """
        res_files.append(pred_dir.parent / "pred_eval.yaml")
        print(analysis_cmd + "\n")
        subprocess.run(analysis_cmd, shell=True)

    # Run cleanup now
    new_entries = []
    for res_file in res_files:
        new_data = yaml.safe_load(open(res_file, "r"))
        thresh = res_file.parent.stem
        new_entry = {"nm_nodes": thresh}
        new_entry.update(
            {
                k: v
                for k, v in new_data.items()
                if "avg" in k or "sem" in k or "std" in k
            }
        )
        new_entries.append(new_entry)

        df = pd.DataFrame(new_entries)
        df.to_csv(save_dir / "summary.tsv", sep="\t", index=None)
