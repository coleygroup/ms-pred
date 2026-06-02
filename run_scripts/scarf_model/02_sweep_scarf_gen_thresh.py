import yaml
import pandas as pd
from pathlib import Path
import subprocess

res_entries = [
    {"folder": "results/scarf_nist20_ablate/",
     "dataset": "nist20",
     'test_split': "split_1"},

    {"folder": "results/scarf_nist20/scaffold_1/",
     "dataset": "nist20",
     "test_split": "scaffold_1"},

    {"folder": "results/scarf_nist20/split_1_rnd1/",
     "dataset": "nist20",
     "test_split": "split_1"},

    {"folder": "results/scarf_nist20/split_1_rnd2/",
     "dataset": "nist20",
     "test_split": "split_1"},

    {"folder": "results/scarf_nist20/split_1_rnd3/",
     "dataset": "nist20",
     "test_split": "split_1"},
]

python_file = "src/ms_pred/scarf_pred/predict_gen.py"
devices = ",".join(["0"])
max_nodes = [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000]
subform_name = "magma_subform_50"


for res_entry in res_entries:
    res_folder = Path(res_entry['folder'])
    dataset = res_entry['dataset']
    models = sorted(list(res_folder.rglob("version_0/*.ckpt")))
    split = res_entry['test_split']


    for model in models:
        save_dir_base = model.parent.parent
        save_dir = save_dir_base / "inten_thresh_sweep"
        save_dir.mkdir(exist_ok=True)

        print(f"Saving inten sweep to: {save_dir}")

        pred_dir_folders = []
        for max_node in max_nodes:
            save_dir_temp = save_dir / str(max_node)
            save_dir_temp.mkdir(exist_ok=True)

            cmd = f"""python {python_file} \\
            --batch-size 32 \\
            --dataset-name  {dataset} \\
            --split-name {split}.tsv \\
            --subset-datasets test_only  \\
            --checkpoint {model} \\
            --save-dir {save_dir_temp} \\
            --num-workers 0 \\
            --threshold 0  \\
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
