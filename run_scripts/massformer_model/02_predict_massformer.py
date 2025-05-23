from pathlib import Path
import subprocess
import argparse

num_workers = 32
python_file = "src/ms_pred/massformer_pred/predict.py"
test_entries = [
    {"dataset": "nist20", "split": "split_1", "folder": "split_1_rnd1"},
    {"dataset": "nist20", "split": "split_1", "folder": "split_1_rnd2"},
    {"dataset": "nist20", "split": "split_1", "folder": "split_1_rnd3"},
    {"dataset": "nist20", "split": "scaffold_1", "folder": "scaffold_1_rnd1"},
    {"dataset": "nist20", "split": "scaffold_1", "folder": "scaffold_1_rnd2"},
    {"dataset": "nist20", "split": "scaffold_1", "folder": "scaffold_1_rnd3"},
    {"dataset": "canopus_train_public", "split": "split_1", "folder": "split_1_rnd1"},
    {"dataset": "canopus_train_public", "split": "split_1", "folder": "split_1_rnd2"},
    {"dataset": "canopus_train_public", "split": "split_1", "folder": "split_1_rnd3"},
]

devices = ",".join(["0"])

for test_entry in test_entries:
    split = test_entry['split']
    folder = test_entry['folder']
    dataset_name = test_entry['dataset']
    test_dataset_name = test_entry['test_dataset']

    res_folder = Path(f"results/massformer_baseline_{dataset_name}")
    model = res_folder / f"{folder}/version_0/best.ckpt"

    save_dir = model.parent.parent
    if dataset_name != test_dataset_name:
            save_dir = save_dir / "cross_dataset" / test_dataset_name

    save_dir = save_dir / "preds"

    save_dir.mkdir(exist_ok=True, parents=True)
    cmd = f"""python {python_file} \\
    --batch-size 32 \\
    --dataset-name {test_dataset_name} \\
    --split-name {split}.tsv \\
    --subset-datasets test_only  \\
     --checkpoint {model} \\
    --save-dir {save_dir} \\
    --gpu"""
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    out_binned = save_dir / "binned_preds.hdf5"
    eval_cmd = f"""python analysis/spec_pred_eval.py \\
    --binned-pred-file {out_binned} \\
    --max-peaks 100 \\
    --min-inten 0 \\
    --formula-dir-name no_subform.hdf5 \\
    --dataset {dataset_name}  \\
    """
    print(eval_cmd)
    subprocess.run(eval_cmd, shell=True)
