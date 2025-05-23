import itertools

import pandas as pd
from pathlib import Path
from ms_pred import common
import pickle
from tqdm import tqdm
import numpy as np
import yaml
from collections import defaultdict

from run_metfrag import metfrag_ion_mapping

datasets = ["nist20"]
splits = ["split_1", "scaffold_1"]
formula_dir_name = "no_subform.hdf5"

for dataset, split in itertools.product(datasets, splits):
    data_folder = Path(f"data/spec_datasets/{dataset}")
    data_df = pd.read_csv(data_folder / "labels.tsv", sep="\t")

    name_to_ion = dict(data_df[["spec", "ionization"]].values)

    labels = f"data/spec_datasets/{dataset}/labels.tsv"
    label_df = pd.read_csv(labels, sep="\t")

    candidates = f"data/spec_datasets/{dataset}/retrieval/cands_pickled_{split}_50.p"
    id_to_dict = pickle.load(open(candidates, 'rb'))

    res_folder = Path(f"results/metfrag_{dataset}/{split}")
    metfrag_output_rank = res_folder / "metfrag_out"
    out_folder = res_folder / f"retrieval"
    out_folder.mkdir(exist_ok=True)

    outfile = out_folder / f"rerank_eval_met_frag.yaml"
    outfile_grouped_ion = (
            out_folder / f"rerank_eval_grouped_ion_met_frag.tsv"
    )
    outfile_grouped_mass = (
            out_folder / f"rerank_eval_grouped_mass_met_frag.tsv"
    )
    outfile_grouped_peak = (
            out_folder / f"rerank_eval_grouped_npeak_met_frag.tsv" # this file is not supported
    )

    all_out = []
    for spec_id, info_dict in tqdm(id_to_dict.items()):
        if info_dict["ionization"] not in metfrag_ion_mapping.keys(): # not supported adduct
            continue
        rank_df = pd.read_csv(metfrag_output_rank / f"{info_dict['spec']}.csv")

        # process and collect result
        gt_entry = rank_df[rank_df["InChIKey1"] == info_dict['inchikey'].split('-')[0]] # first part of InChIKey is stereo invariant
        ind_found = gt_entry.index
        if len(ind_found) == 1:
            ind_found = ind_found[0]
            ind_found = ind_found + 1  # Add 1 in case it was first to be top 1 not zero
        elif len(ind_found) == 0:  # no prediction is made
            ind_found = len(info_dict['cands'])
        else:
            raise ValueError(f'More than one InChI key matching is found for {spec_id}!')

        true_mass = common.mass_from_smi(info_dict['smiles'])
        mass_bin = common.bin_mass_results(true_mass)

        all_out.append({
            "ind_recovered": float(ind_found),
            "total_decoys": len(info_dict['cands']),
            "mass": float(true_mass),
            "mass_bin": mass_bin,
            "num_peaks_avg": -1,
            "peak_bin_avg": -1,
            "peak_bin_max": -1,
            "peak_bin_min": -1,
            "true_dist": -1,
            "spec_name": str(info_dict['spec']),
        })

    # Compute avg and individual stats
    k_vals = list(range(1, 11))
    running_lists = defaultdict(lambda: [])
    output_entries = []
    for out in all_out:
        output_entries.append(out)
        for k in k_vals:
            below_k = out["ind_recovered"] <= k
            running_lists[f"top_{k}"].append(below_k)
            out[f"top_{k}"] = below_k
        running_lists["total_decoys"].append(out["total_decoys"])
        running_lists["true_dist"].append(out["true_dist"])

    final_output = {
        "dataset": dataset,
        "data_folder": str(data_folder),
        "dist_fn": "metfrag",
        "individuals": sorted(output_entries, key=lambda x: x["ind_recovered"]),
    }

    for k, v in running_lists.items():
        final_output[f"avg_{k}"] = float(np.mean(v))

    for i in output_entries:
        i["ion"] = name_to_ion[i["spec_name"]]

    df = pd.DataFrame(output_entries)

    for group_key, out_name in zip(
            ["mass_bin", "ion", "peak_bin_avg"], [outfile_grouped_mass, outfile_grouped_ion, outfile_grouped_peak]
    ):
        df_grouped = pd.concat(
            [df.groupby(group_key).mean(numeric_only=True), df.groupby(group_key).size()], axis=1
        )
        df_grouped = df_grouped.rename({0: "num_examples"}, axis=1)

        all_mean = df.mean(numeric_only=True)
        all_mean["num_examples"] = len(df)
        all_mean.name = "avg"
        df_grouped = pd.concat([df_grouped, all_mean.to_frame().T], axis=0)
        df_grouped.to_csv(out_name, sep="\t")

    with open(outfile, "w") as fp:
        out_str = yaml.dump(final_output, indent=2)
        print(out_str)
        fp.write(out_str)
