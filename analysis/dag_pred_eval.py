""" DAG prediction evaluation

Use to compare predicted trees to ground truth tree values in terms of coverage

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
    parser.add_argument("--dataset", default="nist20")
    parser.add_argument("--magma-name", default="magma_outputs")
    parser.add_argument("--tree-pred-folder",)
    parser.add_argument("--outfile", default=None)
    return parser.parse_args()


def main(args):
    """main."""
    dataset = args.dataset
    magma_names = args.magma_name
    data_folder = Path(f"data/spec_datasets/{dataset}/{magma_names}/magma_tree")

    tree_pred_folder = Path(args.tree_pred_folder)
    outfile = args.outfile
    if outfile is None:
        outfile = tree_pred_folder.parent / "pred_eval.yaml"
        outfile_grouped = tree_pred_folder.parent / "pred_eval_grouped.tsv"

    pred_trees = tree_pred_folder.glob("*.json")
    running_lists = defaultdict(lambda: [])
    output_entries = []
    for pred_tree in pred_trees:
        spec_name = pred_tree.stem.replace("pred_", "")
        true_file = data_folder / f"{spec_name}.json"
        if not true_file.exists():
            print(f"Skipping file {true_file} as no tree was found")
            continue

        true_tree = json.load(open(true_file, "r"))
        pred_tree = json.load(open(pred_tree, "r"))

        tree_inchi = true_tree["root_inchi"]
        pred_inchi = pred_tree["root_inchi"]
        assert tree_inchi == pred_inchi

        # Step 1: Get overlap
        true_frag_keys = set(list(true_tree["frags"].keys()))
        pred_frag_keys = set(list(pred_tree["frags"].keys()))
        true_num_frags = len(true_frag_keys)
        pred_num_frags = len(pred_frag_keys)
        intersect_amt = len(true_frag_keys.intersection(pred_frag_keys))
        union_amt = len(true_frag_keys.union(pred_frag_keys))

        jaccard = intersect_amt / union_amt
        coverage = intersect_amt / true_num_frags

        smiles_mass = common.mass_from_inchi(tree_inchi)
        output_entry = {
            "name": spec_name,
            "inchi": tree_inchi,
            "num_pred": pred_num_frags,
            "num_true": true_num_frags,
            "jaccard": jaccard,
            "coverage": coverage,
            "compound_mass": smiles_mass,
            "mass_bin": common.bin_mass_results(smiles_mass),
        }
        output_entries.append(output_entry)
        running_lists["jaccard"].append(jaccard)
        running_lists["coverage"].append(coverage)
        running_lists["num_pred"].append(pred_num_frags)
        running_lists["num_true"].append(true_num_frags)

    final_output = {
        "dataset": dataset,
        "tree_folder": str(tree_pred_folder),
        "individuals": sorted(output_entries, key=lambda x: x["jaccard"]),
    }

    for k, v in running_lists.items():
        final_output[f"avg_{k}"] = float(np.mean(v))
        # Also calc sem and std
        final_output[f"sem_{k}"] = float(sem(v))
        final_output[f"std_{k}"] = float(np.std(v))

    df = pd.DataFrame(output_entries)
    df_grouped = pd.concat(
        [df.groupby("mass_bin").mean(), df.groupby("mass_bin").size()], 1
    )
    df_grouped = df_grouped.rename({0: "num_examples"}, axis=1)

    all_mean = df.mean()
    all_mean["num_examples"] = len(df)
    all_mean.name = "avg"
    df_grouped = df_grouped.append(all_mean)
    df_grouped.to_csv(outfile_grouped, sep="\t")

    with open(outfile, "w") as fp:
        out_str = yaml.dump(final_output, indent=2)
        print(out_str)
        fp.write(out_str)


if __name__ == "__main__":
    """__main__"""
    args = get_args()
    main(args)
