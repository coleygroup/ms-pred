""" Sweep gen thresh """
import yaml
import pandas as pd
from pathlib import Path
import subprocess

batch_size = 8
list_devices = [0, 1]
gpu_workers = len(list_devices) * 2
workers = 32
devices = ",".join([str(_) for _ in list_devices])
python_file = "src/ms_pred/marason/predict_gen.py"
max_nodes = [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000]
gpu_workers = [len(list_devices) * _ for _ in [8, 8, 8, 8, 8, 6, 4, 3, 2, 2]]
subform_name = "magma_subform_50.hdf5"
debug = False

res_entries = [
    {"folder": "results/marason_nist20/split_1_rnd1/",
     "dataset": "nist20",
     "test_split": "split_1"},

    # {"folder": "results/marason_nist20/split_1_rnd2/",
    #  "dataset": "nist20",
    #  "test_split": "split_1"},
    #
    # {"folder": "results/marason_nist20/split_1_rnd3/",
    #  "dataset": "nist20",
    #  "test_split": "split_1"},
    #
    # {"folder": "results/marason_nist20/scaffold_1_rnd1/",
    #  "dataset": "nist20",
    #  "test_split": "scaffold_1"},
    #
    # {"folder": "results/marason_nist20/scaffold_1_rnd2/",
    #  "dataset": "nist20",
    #  "test_split": "scaffold_1"},
    #
    # {"folder": "results/marason_nist20/scaffold_1_rnd3/",
    #  "dataset": "nist20",
    #  "test_split": "scaffold_1"},
]

if debug:
    max_nodes = max_nodes[:3]

for res_entry in res_entries:
    res_folder = Path(res_entry['folder'])
    dataset = res_entry['dataset']
    models = sorted(list((res_folder / "version_0").rglob("*.ckpt")))
    split = res_entry['test_split']
    for model in models:
        save_dir_base = model.parent.parent

        save_dir = save_dir_base / "inten_thresh_sweep"
        save_dir.mkdir(exist_ok=True)

        print(f"Saving inten sweep to: {save_dir}")

        pred_dir_folders = []
        form_dir_folders = []
        for max_node, gpu_worker in zip(max_nodes, gpu_workers):
            save_dir_temp = save_dir / str(max_node)
            save_dir_temp.mkdir(exist_ok=True)

            cmd = f"""python {python_file} \\
            --batch-size {batch_size} \\
            --dataset-name  {dataset} \\
            --split-name {split}.tsv \\
            --subset-datasets test_only  \\
            --checkpoint {model} \\
            --save-dir {save_dir_temp} \\
            --threshold 0  \\
            --max-nodes {max_node} \\
            --num-workers {gpu_worker} \\
            --gpu
            """

            pred_dir_folders.append(save_dir_temp)
            device_str = f"CUDA_VISIBLE_DEVICES={devices}"
            cmd = f"{device_str} {cmd}"
            print(cmd + "\n")
            subprocess.run(cmd, shell=True)

            # Convert to form files from dag
        for pred_dir in pred_dir_folders:
            tree_pred_folder = pred_dir / "tree_preds.hdf5"
            form_pred_folder = pred_dir / "form_preds.hdf5"
            cmd = f"""python data_scripts/dag/dag_to_subform.py \\
                --num-workers {workers} \\
                --dag-folder {tree_pred_folder} \\
                --out-dir {form_pred_folder} \\
                --all-h-shifts
            """
            print(cmd + "\n")
            subprocess.run(cmd, shell=True)
            form_dir_folders.append(form_pred_folder)

        res_files = []
        for pred_dir in form_dir_folders:
            analysis_cmd = f"""python analysis/form_pred_eval.py \\
                --dataset {dataset} \\
                --tree-pred-folder {pred_dir} \\
                --subform-name {subform_name}
            """
            res_files.append(pred_dir.parent / "pred_eval.yaml")
            print(analysis_cmd + "\n")
            subprocess.run(analysis_cmd, shell=True)

        ## Run cleanup now
        new_entries = []
        for res_file in res_files:
            new_data = yaml.safe_load(open(res_file, "r"))
            thresh = res_file.parent.stem
            new_entry = {"nm_nodes": thresh}
            new_entry.update({k: v for k, v in new_data.items() if "avg" in k})
            new_entries.append(new_entry)

        df = pd.DataFrame(new_entries)
        df.to_csv(save_dir / "summary.tsv", sep="\t", index=None)
