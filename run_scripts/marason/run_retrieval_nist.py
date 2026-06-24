import yaml
from pathlib import Path
import subprocess
import json
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

pred_file = "src/ms_pred/marason/predict_smis.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_benchmark.py"
subform_name = "no_subform"
devices = [0, 1]
vis_devices = ",".join([str(_) for _ in devices])
num_cpu_workers = len(devices) * 5
num_gpu_workers = 0
max_nodes = 100
batch_size = 8
sparse_k = 100
dist = "cos"
binned_out = True
add_ref = False


test_entries = [
    {
        "dataset": "nist20",
        "train_split": "split_1_rnd1",
        "test_split": "split_1",
        "max_k": 50,
        "ref_dir": "data/closest_neighbors/infinite",
    },
    # {
    #     "dataset": "nist20",
    #     "train_split": "split_1_rnd2",
    #     "test_split": "split_1",
    #     "max_k": 50,
    #     "ref_dir": "data/closest_neighbors/infinite",
    # },
    # {
    #     "dataset": "nist20",
    #     "train_split": "split_1_rnd3",
    #     "test_split": "split_1",
    #     "max_k": 50,
    #     "ref_dir": "data/closest_neighbors/infinite",
    # },
    # {
    #     "dataset": "nist20",
    #     "train_split": "scaffold_1_rnd1",
    #     "test_split": "scaffold_1",
    #     "max_k": 50,
    #     "ref_dir": "data/closest_neighbors/infinite/scaffold",
    # },
    # {
    #     "dataset": "nist20",
    #     "train_split": "scaffold_1_rnd2",
    #     "test_split": "scaffold_1",
    #     "max_k": 50,
    #     "ref_dir": "data/closest_neighbors/infinite/scaffold",
    # },
    # {
    #     "dataset": "nist20",
    #     "train_split": "scaffold_1_rnd3",
    #     "test_split": "scaffold_1",
    #     "max_k": 50,
    #     "ref_dir": "data/closest_neighbors/infinite/scaffold",
    # },
]

pred_filename = "binned_preds.hdf5" if binned_out else "preds.hdf5"

for test_entry in test_entries:
    dataset = test_entry["dataset"]
    train_split = test_entry["train_split"]
    split = test_entry["test_split"]
    maxk = test_entry["max_k"]
    inten_dir = Path(f"results/marason_{dataset}")
    inten_model = inten_dir / train_split / "ckpt/inten/best.ckpt"
    print(inten_model)
    if not inten_model.exists():
        print(f"Could not find model {inten_model}; skipping\n: {json.dumps(test_entry, indent=1)}")
        continue

    labels = test_entry.get(
        "candidate_table",
        f"data/spec_datasets/{dataset}/retrieval/cands_df_{split}_{maxk}.tsv",
    )
    true_labels = test_entry.get("true_labels", f"data/spec_datasets/{dataset}/labels.tsv")

    save_dir = inten_dir / train_split / f"retrieval_{dataset}_{split}_{maxk}"
    save_dir.mkdir(exist_ok=True, parents=True)

    args = yaml.safe_load(open(inten_dir / train_split / "args.yaml", "r"))
    form_folder = Path(args["magma_dag_folder"])
    gen_model = form_folder.parent / "ckpt/gen/best.ckpt"

    save_dir = save_dir
    save_dir.mkdir(exist_ok=True, parents=True)
    ref_dir = test_entry["ref_dir"]
    cmd = f"""python {pred_file} \\
    --batch-size {batch_size}  \\
    --dataset-name {dataset} \\
    --sparse-out \\
    --sparse-k {sparse_k} \\
    --max-nodes {max_nodes} \\
    --split-name {split}.tsv   \\
    --gen-checkpoint {gen_model} \\
    --inten-checkpoint {inten_model} \\
    --save-dir {save_dir} \\
    --ref-dir {ref_dir} \\
    --dataset-labels {labels} \\
    --num-cpu-workers {num_cpu_workers} \\
    --num-gpu-workers {num_gpu_workers} \\
    --magma-dag-folder {args["magma_dag_folder"]} \\
    --adduct-shift \\
    --gpu \\
    --embed-elem-group \\
    """
    if add_ref:
        cmd += "--add-ref --max-ref-count 3 "
    if binned_out:
        cmd += "--binned-out"
    device_str = f"CUDA_VISIBLE_DEVICES={vis_devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True, check=True)

    # Run retrieval
    cmd = f"""python {retrieve_file} \\
    --dataset {dataset} \\
    --true-labels {true_labels} \\
    --full-labels {labels} \\
    --formula-dir-name {subform_name}.hdf5 \\
    --pred-file {save_dir / pred_filename} \\
    --dist-fn {dist} \\
    """
    if binned_out:
        cmd += "--binned-pred"

    print(cmd + "\n")
    subprocess.run(cmd, shell=True, check=True)
