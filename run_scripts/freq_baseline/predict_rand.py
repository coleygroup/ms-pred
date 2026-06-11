""" freq baseline for scarf for generating fragments for each molecule """
import json
import pandas as pd
from pathlib import Path
import subprocess
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import yaml
import ms_pred.common as common


def predict_rand(input_name, smiles, formula, max_nodes, outdir):
    """predict_rand.

    Args:
        input_name:
        smiles:
        formula:
        max_nodes:
        outdir:
    """
    possible_forms, masses = common.get_all_subsets(formula)
    sample_num = min(len(possible_forms), max_nodes)
    inds = np.random.choice(len(possible_forms), sample_num, replace=False)
    forms = possible_forms[inds]
    form_masses = masses[inds]
    forms = [common.vec_to_formula(i) for i in forms]
    output_tbl = {
        "ms2_inten": None,
        "rel_inten": None,
        "log_prob": None,
        "formula": forms,
        "formula_mass_no_adduct": form_masses.tolist(),
    }
    json_out = {
        "smiles": smiles,
        "spec_name": input_name,
        "output_tbl": output_tbl,
        "cand_form": formula,
    }

    out_file = outdir / f"{input_name}.json"
    with open(out_file, "w") as fp:
        json.dump(json_out, fp, indent=2)

test_entries = [
    {"dataset": "nist20", "split": "split_1"},
]

subform_name = "magma_subform_50"
max_nodes = [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000]

for test_entry in test_entries: 
    dataset = test_entry['dataset']
    split = test_entry['split']

    res_folder = Path(f"results/rand_baseline_{dataset}/")
    res_folder.mkdir(exist_ok=True)
    subform_dir = Path(f"data/spec_datasets/{dataset}/subformulae/{subform_name}")
    labels_file = f"data/spec_datasets/{dataset}/labels.tsv"


    labels_df = pd.read_csv(labels_file, sep="\t")
    spec_to_smiles = dict(labels_df[["spec", "smiles"]].values)
    spec_to_forms = dict(labels_df[["spec", "formula"]].values)
    split_dir = res_folder / split
    split_dir.mkdir(exist_ok=True)
    split_file = f"data/spec_datasets/{dataset}/splits/{split}.tsv"

    split_df = pd.read_csv(split_file, sep="\t")
    train_names = split_df["spec"][split_df["Fold_0"] == "train"].values
    test_names = split_df["spec"][split_df["Fold_0"] == "test"].values

    train_names = train_names
    test_names = test_names

    train_name = list(train_names)[0]

    subform_files = [subform_dir / f"{spec_name}.json" for spec_name in train_names]

    sweep_folder = split_dir / "inten_thresh_sweep"
    sweep_folder.mkdir(exist_ok=True)
    pred_dir_folders = []
    for max_node in max_nodes:
        save_dir_temp = sweep_folder / str(max_node)
        save_dir_temp.mkdir(exist_ok=True)
        export_dir = save_dir_temp / "form_preds"
        export_dir.mkdir(exist_ok=True)

        predict_fn = lambda x: predict_rand(**x)
        predict_dicts = [
            dict(
                input_name=i,
                smiles=spec_to_smiles[i],
                formula=spec_to_forms[i],
                max_nodes=max_node,
                outdir=export_dir,
            )
            for i in test_names
        ]
        common.chunked_parallel(predict_dicts, predict_fn)
        pred_dir_folders.append(export_dir)

    res_files = []
    for pred_dir in pred_dir_folders:
        print(pred_dir)
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
    df.to_csv(sweep_folder / "summary.tsv", sep="\t", index=None)
