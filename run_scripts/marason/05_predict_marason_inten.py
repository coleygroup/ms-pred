from pathlib import Path
import subprocess
import argparse

python_file = "src/ms_pred/marason/predict_inten.py"
node_num = 100
num_workers = 64
test_entries = [
    # {"dataset": "nist20", "split": "scaffold_1", "folder": "scaffold_1_rnd1"},
    # {"dataset": "nist20", "split": "scaffold_1", "folder": "scaffold_1_rnd2"},
    # {"dataset": "nist20", "split": "scaffold_1", "folder": "scaffold_1_rnd3"},
    {"dataset": "nist20", "split": "split_1", "folder": "split_1_rnd1"},
    # {"dataset": "nist20", "split": "split_1", "folder": "split_1_rnd2"},
    # {"dataset": "nist20", "split": "split_1", "folder": "split_1_rnd3"},
]
devices = ",".join(["0", "1", "2"])

for test_entry in test_entries:
    print(test_entry)
    split = test_entry['split']
    dataset = test_entry['dataset']
    folder = test_entry['folder']

    base_formula_folder = Path(f"results/marason_{dataset}")
    res_folder = Path(f"results/marason_inten_{dataset}/")
    model = res_folder / folder / "version_0/best.ckpt"
    if test_entry["folder"] == "split_1_rnd1":
        ref_dir = "data/closest_neighbors/infinite"
    elif test_entry["split"] == "scaffold_1_rnd1":
        ref_dir = "data/closest_neighbors/infinite/scaffold"
    elif test_entry["folder"] == "split_1_rnd3":
        ref_dir = "data/closest_neighbors/infinite"
    elif test_entry["folder"] == "scaffold_1_rnd3":
        ref_dir = "data/closest_neighbors/infinite/scaffold"
    elif test_entry["folder"] == "split_1_rnd2":
        ref_dir = "data/closest_neighbors/infinite"
    elif test_entry["folder"] == "scaffold_1_rnd2":
        ref_dir = "data/closest_neighbors/infinite/scaffold"



    if not model.exists(): 
        print(model)
        continue

    save_dir = model.parent.parent

    save_dir = save_dir / "preds"

    # Note: Must use preds_train_01
    magma_dag_folder = (
        base_formula_folder / folder / f"preds_train_{node_num}/tree_preds.hdf5"
    )
    inten_folder = (
        base_formula_folder/folder/"preds_train_100_inten.hdf5"
    )
    print("magma:", magma_dag_folder)
    cmd = f"""python {python_file} \\
    --batch-size {num_workers} \\
    --dataset-name {dataset} \\
    --split-name {split}.tsv \\
    --checkpoint {model} \\
    --save-dir {save_dir} \\
    --gpu \\
    --num-workers 0 \\
    --magma-dag-folder {magma_dag_folder} \\
    --inten-folder {inten_folder} \\
    --subset-datasets test_only \\
    --binned-out \\
    --add-ref \\
    --max-ref-count 3 \\
    --ref-dir {ref_dir} \\
    """
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    # Eval it
    out_binned = save_dir / "binned_preds.hdf5"
    eval_cmd = f"""
    python analysis/spec_pred_eval.py \\
    --binned-pred-file {out_binned} \\
    --max-peaks 100 \\
    --min-inten 0 \\
    --formula-dir-name no_subform.hdf5 \\
    --dataset {dataset}  \\
    """
    print(eval_cmd)
    subprocess.run(eval_cmd, shell=True)
