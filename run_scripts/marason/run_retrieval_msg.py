import yaml
from pathlib import Path
import subprocess
import json
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

pred_file = "src/ms_pred/marason/predict_smis.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_benchmark_msg.py"
subform_name = "no_subform"
devices = [0, 1]
vis_devices = ",".join([str(_) for _ in devices])
num_workers = len(devices) * 2
max_nodes = 250
batch_size = 10
dist = "entropy"
binned_out = True


test_entries = [
    {"dataset": "msg", "test_split": "test_formula", "gen_folder": "split_rnd1", 
     "inten_folder": "split_rnd1_entropy", "max_k": 256,
     "ref_dir":"data/msg/closest_neighbors/infinite"},
    ]

pred_filename = "binned_preds.hdf5" if binned_out else "preds.hdf5"

for test_entry in test_entries:
    dataset = test_entry['dataset']
    inten_split = test_entry['inten_folder']
    split = test_entry['test_split']
    maxk = test_entry['max_k']
    inten_dir = Path(f"results/marason_inten_{dataset}")
    inten_model = inten_dir / inten_split / f"version_1/best.ckpt" 
    
    if not inten_model.exists():
        print(f"Could not find model {inten_model}; skipping\n: {json.dumps(test_entry, indent=1)}")
        continue

    labels = f"data/spec_datasets/{dataset}/retrieval/20250312_cands_df_test_formula_256_no_stereo_no_tauts_clean.tsv"

    save_dir = inten_model.parent.parent / f"retrieval_{dataset}_{split}_{maxk}"
    save_dir.mkdir(exist_ok=True)

    args = yaml.safe_load(open(inten_model.parent.parent / "args.yaml", "r"))
    form_folder = Path(args["magma_dag_folder"])
    gen_model = form_folder.parent / "version_0/best.ckpt"
    print(gen_model)

    save_dir = save_dir
    save_dir.mkdir(exist_ok=True)
    ref_dir = test_entry["ref_dir"]
    cmd = f"""python {pred_file} \\
    --batch-size {batch_size}  \\
    --dataset-name {dataset} \\
    --sparse-out \\
    --sparse-k 250 \\
    --max-nodes {max_nodes} \\
    --split-name split.tsv   \\
    --gen-checkpoint {gen_model} \\
    --inten-checkpoint {inten_model} \\
    --save-dir {save_dir} \\
    --ref-dir {ref_dir} \\
    --dataset-labels {labels} \\
    --num-workers {num_workers} \\
    --magma-dag-folder {args["magma_dag_folder"]} \\
    --adduct-shift \\
    --gpu \\
    --add-ref \\
    --max-ref-count 3 \\
    --embed-elem-group \\
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
    --dataset-labels {'labels_withev_validinst_no_stereo_no_tauts.tsv'} \\
    --formula-dir-name {subform_name}.hdf5 \\
    --pred-file {save_dir / pred_filename} \\
    --dist-fn {dist} \\
    """
    if binned_out:
        cmd += "--binned-pred"

    print(cmd + "\n")
    subprocess.run(cmd, shell=True)
