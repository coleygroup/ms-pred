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
    parser.add_argument("--tree-pred-obj",)
    # parser.add_argument("--outfile", default=None)
    return parser.parse_args()


def main(args):
    """main."""
    dataset = args.dataset
    subform_name = args.subform_name
    data_path = Path(f"data/spec_datasets/{dataset}/subformulae/{subform_name}")

    tree_pred_path = Path(args.tree_pred_obj)
    outfile = tree_pred_path.parent / "pred_eval.yaml"
    outfile_grouped = tree_pred_path.parent / "pred_eval_grouped.tsv"

    pred_names = common.PredSpecDB(tree_pred_path).get_all_names()
    running_lists = defaultdict(lambda: [])

    bins = np.linspace(0, 1500, args.num_bins)

    def eval_item(pred_name, tree_pred_path, data_path):
        """eval_item."""
        tree_pred_h5 = common.PredSpecDB(tree_pred_path)
        colli_engs, _ = tree_pred_h5.get_entries(pred_name)
        true_h5 = common.HDF5Dataset(data_path)
        spec_name = Path(pred_name).stem.replace("pred_", "")

        output_entries = []
        for energy in colli_engs:
            if energy is not None:
                true_name = f"{spec_name}_collision {energy}.json"
            else:
                true_name = f"{spec_name}.json"
            if not true_name in true_h5:
                print(f"Skipping file {true_name} as no tree was found")
                output_entries.append(None)
                break

            true_tree = json.loads(true_h5.read_str(true_name))
            pred_tree = tree_pred_h5.read(pred_name, energy)
            pred_tree.add_hydrogen_shift()  # add hydrogen shift to fragment DAG

            tree_form = true_tree["cand_form"]
            pred_form = pred_tree.root_form

            standard_pred_form = common.standardize_form(pred_form)
            standard_tree_form = common.standardize_form(tree_form)
            assert standard_pred_form == standard_tree_form

            if true_tree["output_tbl"] is None:
                output_entries.append(None)
                break

            true_tbl = true_tree["output_tbl"]

            # Step 1: Get overlap
            true_frag_forms = [common.standardize_form(i) for i in true_tbl["formula"]]
            pred_frag_forms = [common.standardize_form(i) for i in pred_tree.frag_form]

            true_frag_keys = set(true_frag_forms)
            pred_frag_keys = set(pred_frag_forms)

            true_masses = true_tbl["formula_mass_no_adduct"]
            pred_masses = pred_tree.masses_no_adduct

            true_intens = true_tbl["rel_inten"]

            true_form_to_inten = dict(zip(true_frag_forms, true_intens))
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

            smiles_mass = common.mass_from_smi(pred_tree.root_canonical_smiles)
            output_entries.append({
                "name": str(spec_name),
                "smiles": pred_tree.root_canonical_smiles,
                "num_pred": pred_num_frags,
                "num_true": true_num_frags,
                "jaccard": jaccard,
                "coverage": coverage,
                "compound_mass": smiles_mass,
                "mass_bin": common.bin_mass_results(smiles_mass),
                "digitized_coverage": digitized_cvg,
                "inten_coverage": inten_covg,
            })
        return output_entries

    eval_entries = [
        dict(pred_name=pred_name, tree_pred_path=tree_pred_path, data_path=data_path)
        for pred_name in pred_names
    ]
    eval_fn = lambda x: eval_item(**x)
    output_entries = common.chunked_parallel(eval_entries, eval_fn)
    output_entries = [i for j in output_entries for i in j if i is not None]

    for output_entry in output_entries:
        running_lists["jaccard"].append(output_entry["jaccard"])
        running_lists["coverage"].append(output_entry["coverage"])
        running_lists["inten_coverage"].append(output_entry["inten_coverage"])
        running_lists["digitized_coverage"].append(output_entry["digitized_coverage"])
        running_lists["num_pred"].append(output_entry["num_pred"])
        running_lists["num_true"].append(output_entry["num_true"])

    final_output = {
        "dataset": dataset,
        "tree_path": str(tree_pred_path),
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
