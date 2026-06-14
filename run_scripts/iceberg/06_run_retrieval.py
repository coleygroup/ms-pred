import yaml
from pathlib import Path
import subprocess
import json

pred_file = "src/ms_pred/dag_pred/predict_smis.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_benchmark.py"
# can change the above to retrieval_benchmark_torchmetrics.py to retrieve with a candidate set that had duplicate stereoisomers, e.g.
# the spectrum simulation hit rate evaluation of the MassSpecGym challenge. 
subform_name = "no_subform"
devices = [0, 1]
vis_devices = ",".join([str(_) for _ in devices])
num_workers = len(devices) * 6
max_nodes = 100
batch_size = 8
dist = "entropy"
binned_out = True

test_entries = [
    {"dataset": "nist20",
     "train_split": "split_1_rnd1",
     "test_split": "split_1",
     "max_k": 50},

    # {"dataset": "nist20",
    #  "train_split": "split_1_rnd2",
    #  "test_split": "split_1",
    #  "max_k": 50},
    #
    # {"dataset": "nist20",
    #  "train_split": "split_1_rnd3",
    #  "test_split": "split_1",
    #  "max_k": 50},
    #
    # {"dataset": "nist20",
    #  "train_split": "scaffold_1_rnd1",
    #  "test_split": "scaffold_1",
    #  "max_k": 50},
    #
    # {"dataset": "nist20",
    #  "train_split": "scaffold_1_rnd2",
    #  "test_split": "scaffold_1",
    #  "max_k": 50},
    #
    # {"dataset": "nist20",
    #  "train_split": "scaffold_1_rnd3",
    #  "test_split": "scaffold_1",
    #  "max_k": 50},

    ## MassSpecGym retrieval set
    # {"dataset": "msg",
    #     "train_split": "split_1_rnd1",
    #     "test_split": "split_1",
    #     "max_k": 256,
    #     "candidates": "cands_df_test_formula_256_no_stereo_clean_tauts.tsv",
    #     "true_labels": "labels.tsv",
    # },
    
    ## If using retrieval on full set with stereoisomers (to ensure fair comparison to challenge baselines)
    # Add full_labels flag and change retrieval_benchmark.py to retrieval_benchmark_torchmetrics.py
    # {"dataset": "msg",
    #     "train_split": "split_1_rnd1",
    #     "test_split": "split_1",
    #     "max_k": 256,
    #     "candidates": "cands_df_test_formula_256_no_stereo_clean_tauts.tsv",
    #     "true_labels": "labels.tsv",
    #     "full_labels": "cands_df_test_formula_256.tsv"
    # },

]

pred_filename = "binned_preds.hdf5" if binned_out else "preds.hdf5"

for test_entry in test_entries:
    dataset = test_entry['dataset']
    train_split = test_entry['train_split']
    split = test_entry['test_split']
    maxk = test_entry['max_k']
    inten_dir = Path(f"results/dag_inten_{dataset}")
    inten_model = inten_dir / train_split / "version_1/best.ckpt"  # contrastive learning model is version 1
                                                                   # if no contrastive finetuning, change version_1 to version_0
    if not inten_model.exists():
        print(f"Could not find model {inten_model}; skipping\n: {json.dumps(test_entry, indent=1)}")
        continue

    candidates = f"data/spec_datasets/{dataset}/retrieval/cands_df_{split}_{maxk}.tsv"

    save_dir = inten_model.parent.parent / f"retrieval_{dataset}_{split}_{maxk}"
    save_dir.mkdir(exist_ok=True)

    args = yaml.safe_load(open(inten_model.parent.parent / "args.yaml", "r"))
    form_folder = Path(args["magma_dag_folder"])
    gen_model = form_folder.parent / "version_0/best.ckpt"

    save_dir = save_dir
    save_dir.mkdir(exist_ok=True)
    cmd = f"""python {pred_file} \\
    --batch-size {batch_size}  \\
    --dataset-name {dataset} \\
    --sparse-out \\
    --sparse-k 100 \\
    --max-nodes {max_nodes} \\
    --split-name {split}.tsv   \\
    --gen-checkpoint {gen_model} \\
    --inten-checkpoint {inten_model} \\
    --save-dir {save_dir} \\
    --dataset-labels {candidates} \\
    --num-workers {num_workers} \\
    --gpu \\
    --adduct-shift \\
    """
    if binned_out:
        cmd += "--binned-out"
    device_str = f"CUDA_VISIBLE_DEVICES={vis_devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    # Run retrieval
    cmd = f"""python {retrieve_file} \\
    --dataset {dataset} \\
    --formula-dir-name {subform_name}.hdf5 \\
    --pred-file {save_dir / pred_filename} \\
    --true-labels data/spec_datasets/{dataset}/labels.tsv \\
    --dist-fn {dist} \\
    """
    if binned_out:
        cmd += "--binned-pred"
    if "full_labels" in test_entry:
        cmd += f" --full-labels data/spec_datasets/{dataset}/{test_entry['full_labels']} "

    print(cmd + "\n")
    subprocess.run(cmd, shell=True)
