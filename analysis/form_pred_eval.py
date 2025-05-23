""" Formula prediction evaluation

Use to compare scarf predicted formula to actual formulae

"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import argparse
import yaml
from scipy.stats import sem

import ms_pred.common as common


def get_args():
    """get_args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="canopus_train_public")
    parser.add_argument(
        "--subform-name",
        default="magma_outputs",
    )
    parser.add_argument("--num-bins", default=15000, type=int)
    parser.add_argument("--tree-pred-folder",)
    # parser.add_argument("--outfile", default=None)
    return parser.parse_args()


def main(args):
    """main."""
    dataset = args.dataset
    subform_name = args.subform_name
    data_folder = Path(f"data/spec_datasets/{dataset}/subformulae/{subform_name}")

    tree_pred_folder = Path(args.tree_pred_folder)
    outfile = tree_pred_folder.parent / "pred_eval.yaml"
    outfile_grouped = tree_pred_folder.parent / "pred_eval_grouped.tsv"

    tree_pred_h5 = common.HDF5Dataset(tree_pred_folder)
    pred_trees = tree_pred_h5.get_all_names()
    running_lists = defaultdict(lambda: [])
    output_entries = []

    bins = np.linspace(0, 1500, args.num_bins)

    def eval_item(pred_tree, tree_pred_folder, data_folder):
        """eval_item."""
        tree_pred_h5 = common.HDF5Dataset(tree_pred_folder)
        true_h5 = common.HDF5Dataset(data_folder)
        spec_name = Path(pred_tree).stem.replace("pred_", "")
        true_name = f"{spec_name}.json"
        if not true_name in true_h5:
            print(f"Skipping file {true_name} as no tree was found")
            return None

        true_tree = json.loads(true_h5.read_str(true_name))
        pred_tree = json.loads(tree_pred_h5.read_str(pred_tree))

        tree_form = true_tree["cand_form"]
        pred_form = pred_tree["cand_form"]
        pred_smi = pred_tree["smiles"]

        standard_pred_form = common.standardize_form(pred_form)
        standard_tree_form = common.standardize_form(tree_form)
        assert standard_pred_form == standard_tree_form

        if true_tree["output_tbl"] is None:
            return None

        true_tbl = true_tree["output_tbl"]
        pred_tbl = pred_tree["output_tbl"]

        # Step 1: Get overlap
        true_frag_forms = [common.standardize_form(i) for i in true_tbl["formula"]]
        pred_frag_forms = [common.standardize_form(i) for i in pred_tbl["formula"]]

        true_frag_keys = set(true_frag_forms)
        pred_frag_keys = set(pred_frag_forms)

        true_masses = true_tbl["formula_mass_no_adduct"]
        pred_masses = pred_tbl["formula_mass_no_adduct"]

        true_intens = true_tbl["rel_inten"]
        pred_intens = pred_tbl["rel_inten"]

        true_form_to_inten = dict(zip(true_frag_forms, true_intens))
        # pred_form_to_inten = dict(zip(pred_frag_forms, pred_intens))
        pred_frag_forms_set = set(pred_frag_forms)

        total_true_inten = np.sum(true_intens)
        overlap_inten = np.sum(
            [
                true_form_to_inten[i]
                for i in true_form_to_inten
                if i in pred_frag_forms_set
            ]
        )
        inten_covg = float(overlap_inten / (total_true_inten + 1e-22))

        true_digitized = set(np.digitize(true_masses, bins=bins).tolist())
        pred_digitized = set(np.digitize(pred_masses, bins=bins).tolist())

        digitized_overlap = len(true_digitized.intersection(pred_digitized))
        digitized_cvg = digitized_overlap / (len(true_digitized) + 1e-22)

        true_num_frags = len(true_frag_keys)
        pred_num_frags = len(pred_frag_keys)
        intersect_amt = len(true_frag_keys.intersection(pred_frag_keys))
        union_amt = len(true_frag_keys.union(pred_frag_keys))

        jaccard = intersect_amt / union_amt
        coverage = intersect_amt / true_num_frags

        smiles_mass = common.mass_from_smi(pred_smi)
        output_entry = {
            "name": str(spec_name),
            "smiles": pred_smi,
            "num_pred": pred_num_frags,
            "num_true": true_num_frags,
            "jaccard": jaccard,
            "coverage": coverage,
            "compound_mass": smiles_mass,
            "mass_bin": common.bin_mass_results(smiles_mass),
            "digitized_coverage": digitized_cvg,
            "inten_coverage": inten_covg,
        }
        return output_entry

    eval_entries = [
        dict(pred_tree=pred_tree, tree_pred_folder=tree_pred_folder, data_folder=data_folder)
        for pred_tree in pred_trees
    ]
    eval_fn = lambda x: eval_item(**x)
    # output_entries = [eval_fn(i) for i in eval_entries]
    output_entries = common.chunked_parallel(eval_entries, eval_fn)
    output_entries = [i for i in output_entries if i is not None]

    for output_entry in output_entries:
        running_lists["jaccard"].append(output_entry["jaccard"])
        running_lists["coverage"].append(output_entry["coverage"])
        running_lists["inten_coverage"].append(output_entry["inten_coverage"])
        running_lists["digitized_coverage"].append(output_entry["digitized_coverage"])
        running_lists["num_pred"].append(output_entry["num_pred"])
        running_lists["num_true"].append(output_entry["num_true"])

    final_output = {
        "dataset": dataset,
        "tree_folder": str(tree_pred_folder),
        "individuals": sorted(output_entries, key=lambda x: x["jaccard"]),
    }

    for k, v in running_lists.items():
        final_output[f"avg_{k}"] = float(np.mean(v))
        final_output[f"sem_{k}"] = float(sem(v))
        final_output[f"std_{k}"] = float(np.std(v))

    df = pd.DataFrame(output_entries)
    df = df.drop(['name', 'smiles'], axis='columns')
    df_grouped = pd.concat(
        [df.groupby("mass_bin").mean(), df.groupby("mass_bin").size()], axis=1
    )
    df_grouped = df_grouped.rename({0: "num_examples"}, axis=1)

    all_mean = df.drop('mass_bin', axis='columns').mean()
    all_mean["num_examples"] = len(df)
    all_mean.name = "avg"
    df_grouped = df_grouped._append(all_mean)
    df_grouped.to_csv(outfile_grouped, sep="\t")

    with open(outfile, "w") as fp:
        out_str = yaml.dump(final_output, indent=2)
        print(out_str)
        fp.write(out_str)


if __name__ == "__main__":
    """__main__"""
    args = get_args()
    main(args)
