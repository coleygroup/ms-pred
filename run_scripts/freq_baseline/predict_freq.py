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

def extract_diff_freqs(subform_file):
    """extract_diff_freqs.

    Args:
        subform_file:
    """
    subforms = json.load(open(subform_file, "r"))
    output_tbl = subforms["output_tbl"]
    main_form = subforms["cand_form"]
    freqs = defaultdict(lambda: 0)
    if output_tbl is None:
        pass
    else:
        new_forms = output_tbl["formula"]
        dense_forms = np.vstack([common.formula_to_dense(i) for i in new_forms])
        base_form = common.formula_to_dense(main_form)
        diffs = base_form[None, :] - dense_forms
        diff_forms = [f"NEG_{common.vec_to_formula(i)}" for i in diffs]
        for i in diff_forms:
            freqs[i] += 1
        for i in new_forms:
            freqs[i] += 1

    return freqs


def merge_diffs(diff_list):
    """merge_diffs.

    Args:
        diff_list:
    """

    total_num = 0
    freq_diffs = defaultdict(lambda: 0)
    for freq_diff in tqdm(diff_list):
        for k in freq_diff.keys():
            freq_diffs[k] += 1
            total_num += 1
    return freq_diffs


def predict_top_k(input_name, smiles, formula, max_nodes, outdir, frags):
    """predict_top_k.

    Args:
        input_name:
        smiles:
        formula:
        max_nodes:
        outdir:
        frags:
    """
    base_form = common.formula_to_dense(formula)
    neg_forms = np.all(frags <= 0, -1)
    possible_frags = np.empty_like(frags)
    possible_frags[~neg_forms] = frags[~neg_forms]
    possible_frags[neg_forms] = base_form + frags[neg_forms]

    is_subset = ~np.any((base_form - possible_frags) < 0, -1)
    possible_frags = possible_frags[is_subset]
    nodes = possible_frags[:max_nodes]
    form_masses = nodes.dot(common.VALID_MONO_MASSES)
    forms = [common.vec_to_formula(i) for i in nodes]

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
    subform_dir = Path(f"data/spec_datasets/{dataset}/subformulae/{subform_name}")
    labels_file = f"data/spec_datasets/{dataset}/labels.tsv"
    res_folder = Path(f"results/freq_baseline_{dataset}/")
    res_folder.mkdir(exist_ok=True)
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

    all_diff_freqs = common.chunked_parallel(subform_files, extract_diff_freqs)
    freq_diff = merge_diffs(all_diff_freqs)
    sorted_freqs = sorted(list(freq_diff.items()), key=lambda x: -x[1])
    dense_frags = [
        common.formula_to_dense(i[0])
        if "NEG_" not in i[0]
        else -common.formula_to_dense(i[0].replace("NEG_", ""))
        for i in sorted_freqs
    ]
    dense_frags = np.vstack(dense_frags)

    sweep_folder = split_dir / "inten_thresh_sweep"
    sweep_folder.mkdir(exist_ok=True)
    pred_dir_folders = []
    for max_node in max_nodes:
        save_dir_temp = sweep_folder / str(max_node)
        save_dir_temp.mkdir(exist_ok=True)
        export_dir = save_dir_temp / "form_preds"
        export_dir.mkdir(exist_ok=True)

        predict_fn = lambda x: predict_top_k(**x)
        predict_dicts = [
            dict(
                input_name=i,
                smiles=spec_to_smiles[i],
                formula=spec_to_forms[i],
                max_nodes=max_node,
                outdir=export_dir,
                frags=dense_frags,
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
