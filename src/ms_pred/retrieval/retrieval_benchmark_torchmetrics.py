import ast
from pathlib import Path
import json
import argparse
import yaml
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from typing import Dict, List
import copy

import torch
from torchmetrics.retrieval import RetrievalHitRate
import pygmtools as pygm
import numpy as np
from numpy.linalg import norm
import pandas as pd
import ms_pred.common as common

def get_args():
    """get_args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="canopus_train_public")
    parser.add_argument("--formula-dir-name", default="subform_20")
    parser.add_argument(
        "--pred-file",
        default="results/ffn_baseline_cos/retrieval/split_1/fp_preds.p",
    )
    parser.add_argument("--outfile", default=None)
    parser.add_argument("--dist-fn", default="cos")
    parser.add_argument(
        "--ignore-parent-peak",
        action="store_true",
        default=False,
        help="If true, ignore the precursor peak",
    )
    # Defines the specific query spectra to test
    parser.add_argument(
        "--true-labels",
        action="store",
        required=True,
        help="Path to the condensed, 1:1 labels file defining the query spectra.",
    )
    # Defines the full universe of compounds for metadata lookup
    parser.add_argument(
        "--full-labels",
        action="store",
        default=None,
        help="Path to the full, redundant label set for all compounds. If None, defaults based on --dataset.",
    )
    parser.add_argument(
        "--binned-pred",
        action="store_true",
        default=False,
        help="If true, the spec predictions are binned",
    )
    parser.add_argument(
        "--denoise-true",
        choices=['None', 'Simple', 'Stratified', 'spectral'],
        default="None",
        help="Denoise true spectrum according to selected strategy"
    )
    parser.add_argument(
        "--denoise-pred",
        choices=['None', 'Simple', 'Stratified', 'spectral'],
        default="None",
        help="Denoise predicted spectrum according to selected strategy"
    )
    return parser.parse_args()


def process_spec_file(spec_name, name_to_colli: dict, spec_dir: Path, num_bins: int = -1, upper_limit: int = -1,
                      binned_spec: bool=True, dataset: str="nist20"):
    """process_spec_file."""
    if binned_spec:
        assert num_bins > 0 and upper_limit > 0

    if spec_dir.suffix == '.hdf5':
        spec_h5 = common.HDF5Dataset(spec_dir)
    else:
        spec_h5 = None
    return_dict = {}
    
    # Use .get() to avoid KeyError if a spec_name is not in the collision map
    colli_energies = name_to_colli.get(spec_name, [])
    for colli_label in colli_energies:
        if spec_h5 is not None:
            if dataset == "msg":
                spec_file = f"{spec_name}_collision {colli_label} eV.json"
                spec_file_alt = f"{spec_name}_collision {colli_label} eV [imputed].json"
            else:
                spec_file = f"{spec_name}_collision {colli_label}.json"
                spec_file_alt = None
            if spec_file in spec_h5:
                loaded_json = json.loads(spec_h5.read_str(spec_file))
            elif spec_file_alt is not None and spec_file_alt in spec_h5:
                loaded_json = json.loads(spec_h5.read_str(spec_file_alt))
            else:
                print(f"Cannot find spec {spec_file}")
                return_dict[colli_label] = np.zeros(num_bins) if binned_spec else np.zeros((0, 2))
                continue
        else:
            spec_file = spec_dir / f"{spec_name}.json"
            if not spec_file.exists():
                print(f"Cannot find spec {spec_file}")
                return np.zeros(num_bins) if binned_spec else np.zeros((0, 2))
            loaded_json = json.load(open(spec_file, "r"))

        if loaded_json.get("output_tbl") is None:
            return_dict[colli_label] = None
            continue

        mz = loaded_json["output_tbl"]["mono_mass"]
        inten = loaded_json["output_tbl"]["ms2_inten"]
        spec_ar = np.vstack([mz, inten]).transpose(1, 0)
        if binned_spec:
            binned = common.bin_spectra([spec_ar], num_bins, upper_limit)
            return_dict[colli_label] = binned[0]
        else:
            return_dict[colli_label] = spec_ar
    return return_dict


def dist_bin(cand_preds_dict: List[Dict], true_spec_dict: dict, sparse=True, ignore_peak=None, func='cos', selected_evs=None, agg=True) -> np.ndarray:
    """Distance function; defaults to cos_dist for binned spectrum

    Args:
        cand_preds_dict: List of candidates
        true_spec_dict:
        ignore_peak: ignore peaks whose indices are larger than this value

    Returns:
        np.ndarray:
    """
    dist = []
    true_npeaks = []

    if selected_evs:
        true_spec_dict = {k: v for k, v in true_spec_dict.items() if str(k) in selected_evs
    true_spec_dict = {f'{float(k):.0f}': v for k, v in true_spec_dict.items()}
    cand_preds_dict = [{f'{float(k):.0f}': v for k, v in cand_dict.items()} for cand_dict in cand_preds_dict]
    
    for colli_eng in true_spec_dict.keys():
        true_spec = true_spec_dict[colli_eng]
        try:
            cand_preds = np.stack([i.get(colli_eng, np.zeros_like(true_spec)) for i in cand_preds_dict], axis=0)
            stacked = True
        except ValueError:
            cand_preds = [i.get(colli_eng, np.zeros((0,2))) for i in cand_preds_dict]
            stacked = False
        
        if sparse:
            pred_specs = np.zeros((len(cand_preds), true_spec.shape[0]))
            if stacked:
                inds = cand_preds[:, :, 0].astype(int)
                pos_1 = np.repeat(np.arange(inds.shape[0])[:, np.newaxis], inds.shape[1], axis=1)
                valid_mask = inds < pred_specs.shape[1]
                pred_specs[pos_1[valid_mask], inds[valid_mask]] = cand_preds[:, :, 1][valid_mask]
            else:
                 for i, cand_pred in enumerate(cand_preds):
                    if cand_pred.ndim == 2 and cand_pred.shape[1] == 2:
                        inds = cand_pred[:, 0].astype(int)
                        valid_mask = inds < pred_specs.shape[1]
                        pred_specs[i, inds[valid_mask]] = cand_pred[valid_mask, 1]
        else:
            pred_specs = cand_preds

        if ignore_peak:
            pred_specs[:, int(ignore_peak):] = 0
            true_spec = true_spec.copy()
            true_spec[int(ignore_peak):] = 0

        true_npeaks.append(np.sum(true_spec > 0))

        if func == 'cos':
            norm_pred = norm(pred_specs, axis=-1) + 1e-22
            norm_true = norm(true_spec, axis=-1) + 1e-22
            dist.append(1 - np.dot(pred_specs, true_spec) / (norm_pred * norm_true))
        elif func == 'entropy':
            def norm_peaks(prob):
                return prob / (prob.sum(axis=-1, keepdims=True) + 1e-22)

            def entropy(prob):
                return -np.sum(prob * np.log(prob + 1e-22), axis=-1)

            norm_pred = norm_peaks(pred_specs)
            norm_true = norm_peaks(true_spec)
            zeros = (pred_specs.sum(axis=-1) == 0)
            entropy_pred = entropy(norm_pred)
            entropy_targ = entropy(norm_true)
            entropy_mix = entropy((norm_pred + norm_true) / 2)
            entropy_dists = (2 * entropy_mix - entropy_pred - entropy_targ) / np.log(4)
            entropy_dists[zeros] = 1 # mask empty spectra to be distance 1
            dist.append(entropy_dists)

    dist = np.array(dist)  # num of colli energy x number of candidates
    # if >=5 peaks: weight=4, elif >=1 peaks: weight=1, else: weight=0
    weights = (np.array(true_npeaks) >= 5) * 3 + (np.array(true_npeaks) >= 1) * 1
    weights = weights / (weights.sum() + 1e-9)
    if agg:
        return np.sum(dist * weights[:, None], axis=0)
    else:
        # return both
        dist = dist[weights > 0] # exclude any objectives that have zero-weights based on filter
        return dist, np.sum(dist * weights[:, None], axis=0)  # number of candidates


# Define distance functions
cos_dist_bin = partial(dist_bin, func='cos')
entropy_dist_bin = partial(dist_bin, func='entropy')

def main(args):
    """main."""
    dataset = args.dataset
    formula_dir_name = args.formula_dir_name
    dist_fn = args.dist_fn
    ignore_parent_peak = args.ignore_parent_peak
    binned_pred = args.binned_pred
    data_folder = Path(f"data/spec_datasets/{dataset}")
    form_folder = data_folder / f"subformulae/{formula_dir_name}/"

    # Load query labels
    print(f"Loading query labels from: {args.true_labels}")
    query_df = pd.read_csv(data_folder / args.true_labels, sep="\t")
    query_specs = query_df['spec'].unique()
    
    # Load full labels for metadata lookup
    full_labels_path = args.full_labels
    if full_labels_path is None:
        if dataset == 'nist20':
            full_labels_path = data_folder / "labels.tsv"
        elif dataset == 'msg':
            # Subset of entries that have a valid instrument
            full_labels_path = data_folder / "labels_withev_validinst.tsv"
    
    print(f"Loading full metadata labels from: {full_labels_path}")
    full_df = pd.read_csv(full_labels_path, sep="\t")
    # Create mapping dicts from the full dataset
    name_to_ikey = dict(query_df[["spec", "inchikey"]].values)
    name_to_smi = dict(query_df[["spec", "smiles"]].values)
    name_to_ion = dict(query_df[["spec", "ionization"]].values)
    name_to_colli = dict(query_df[["spec", "collision_energies"]].values)
    if "instrument" in query_df.columns:
        name_to_instrument = dict(query_df[["spec", "instrument"]].values)

    # Set up output files
    pred_file = Path(args.pred_file)
    outfile = args.outfile
    if outfile is None:
        outfile = pred_file.parent / f"rerank_eval_{dist_fn}_torchmetrics.yaml"
    else:
        outfile = Path(outfile)

    # Load prediction data
    pred_specs = common.HDF5Dataset(pred_file)
    if binned_pred:
        upper_limit = pred_specs.attrs["upper_limit"]
        num_bins = pred_specs.attrs["num_bins"]
    use_sparse = pred_specs.attrs["sparse_out"]
    assert use_sparse, "Only sparse output is currently supported."

    inchikey_check = list(name_to_ikey.values())[0]

    pred_spec_ars, pred_ikeys, pred_spec_names = [], [], []
    print("Loading predictions from HDF5...")
    for pred_spec_obj in tqdm(list(pred_specs.h5_obj.values()), "Initial collection"):
        for smiles_obj in pred_spec_obj.values():
            ikey, name, spec_dict = None, None, {}
            for collision_eng_key, collision_eng_obj in smiles_obj.items():
                if name is None: name = collision_eng_obj.attrs['spec_name']
                if ikey is None: ikey = collision_eng_obj.attrs['ikey']
                if "-" not in inchikey_check:
                    ikey = ikey.split("-")[0]
                collision_eng_key = common.get_collision_energy(collision_eng_key)
                spec_dict[collision_eng_key] = collision_eng_obj['spec'][:]
            pred_spec_ars.append(spec_dict)
            pred_ikeys.append(ikey)
            pred_spec_names.append(name)
    
    pred_spec_ars = np.array(pred_spec_ars, dtype=object)
    pred_ikeys = np.array(pred_ikeys)
    pred_spec_names = np.array(pred_spec_names)

    # Use unique spec names from the query file as the ground truth set of queries
    query_spec_names_unique = sorted(list(query_specs))

    # Clean collision energies
    for spec_name in name_to_colli.keys():
        name_to_colli[spec_name] = [
            ce for ce in ast.literal_eval(name_to_colli[spec_name]) if 'nan' not in ce
        ]
    for idx in range(len(pred_spec_ars)):
        pred_spec_ars[idx] = {k: v for k, v in pred_spec_ars[idx].items() if 'nan' not in k}

    
    read_spec = partial(
        process_spec_file, 
        name_to_colli=name_to_colli, 
        spec_dir=form_folder, 
        binned_spec=binned_pred, 
        dataset=dataset)
    if binned_pred:
        # Add bin info + upper limit. 
        read_spec = partial(read_spec, num_bins=num_bins, upper_limit=upper_limit)
    
    true_specs = common.chunked_parallel(
        query_spec_names_unique, read_spec, chunks=100, max_cpu=96, task_name="Loading true spectra"
    )
    name_to_spec = dict(zip(query_spec_names_unique, true_specs))

    all_preds_torch, all_targets_torch, all_indexes_torch = [], [], []
    output_entries_for_df = []
    missing_cands_count = 0
    inchikey_poor_map = 0
    inchikey_true_mismatch = 0
    for query_idx, spec_name in enumerate(tqdm(query_spec_names_unique, desc="Processing and ranking entries...")):
        true_spec = name_to_spec.get(spec_name)
        
        if true_spec is None or not any(v.size > 0 for v in true_spec.values()):
            # print(f"Skipping {spec_name} due to missing or empty true spectrum.")
            continue

        # Get all candidates for this query from the prediction file
        bool_sel = pred_spec_names == spec_name
        cand_ikeys = pred_ikeys[bool_sel]
        cand_preds = pred_spec_ars[bool_sel]
        
        true_ikey = name_to_ikey[spec_name]
        if true_ikey not in cand_ikeys:
            missing_cands_count += 1
            continue

        true_smi = name_to_smi[spec_name]
        true_ion = name_to_ion[spec_name]
        parent_mass = common.mass_from_smi(true_smi) + common.ion2mass[true_ion]
        parent_mass_idx = (parent_mass - 1) * num_bins / upper_limit if ignore_parent_peak and binned_pred else None
        
        # Calculate distances
        if dist_fn == "cos" and binned_pred:
            dist = cos_dist_bin(cand_preds_dict=cand_preds, true_spec_dict=true_spec, ignore_peak=parent_mass_idx)
        elif dist_fn == "entropy" and binned_pred:
            dist = entropy_dist_bin(cand_preds_dict=cand_preds, true_spec_dict=true_spec, ignore_peak=parent_mass_idx)
            cosine_dist = cos_dist_bin(cand_preds_dict=cand_preds, true_spec_dict=true_spec, ignore_peak=parent_mass_idx)
        else:
            raise NotImplementedError()
        
        # Prepare tensors for torchmetrics
        # Score is negative distance (higher is better)

        # Before doing so: duplicate scores according to indexing in full_df since the full candidate set had stereoisomers
        # And we deduplicated those during prediction to save compute.
        full_inchikey_list = full_df[full_df['spec'] == spec_name]['inchikey'].values
        inchikey_to_dist = dict(zip(cand_ikeys, dist))
        dist_full = np.vectorize(inchikey_to_dist.get)(full_inchikey_list)
        
        inchikey_to_cos = dict(zip(cand_ikeys, cosine_dist))
        cosine_dist_full = np.vectorize(inchikey_to_cos.get)(full_inchikey_list)

        if np.any(np.isnan(dist_full)):
            # print num of nans; could set to 1 and hope for the best?
            # print("Number of nans for spec", spec_name, ":", np.sum(np.isnan(dist_full)), "out of", len(dist_full))
            # is true candidate missing - check!
            inchikey_poor_map += 1 
            if np.isnan(dist_full[full_inchikey_list == true_ikey]).all():
                print("True candidate missing for spec", spec_name, "with inchikey", true_ikey)
                inchikey_true_mismatch += 1 
                continue
            dist_full = np.nan_to_num(dist_full, nan=1.0)
            cosine_dist_full = np.nan_to_num(cosine_dist_full, nan=1.0)

        all_preds_torch.append(torch.from_numpy(-dist_full))
        target = (full_inchikey_list == true_ikey)
        all_targets_torch.append(torch.from_numpy(target))
        all_indexes_torch.append(torch.full((len(full_inchikey_list),), query_idx, dtype=torch.long))

        # Calculate rank for detailed dataframe logging
        sorted_indices = np.argsort(dist_full)
        rank = np.where(target[sorted_indices])[0][0] + 1
        topk_bools = {f"top_{k}": rank <= k for k in [1, 5, 10, 20]}
        output_entries_for_df.append({"ind_recovered": rank, 
                                      "total_decoys": len(cand_ikeys), 
                                      "decoys_in_full": len(full_inchikey_list),
                                      "spec_name": spec_name,
                                      "true_dist": dist[cand_ikeys == true_ikey][0],
                                      "cosine_dist": cosine_dist[cand_ikeys == true_ikey][0],
                                      **topk_bools})

    print(f"Total queries where true candidate was missing: {missing_cands_count}")
    print(f"Total queries with poor inchikey mapping (can be poor map, or bad cleanup that was dropped): {inchikey_poor_map}")
    print(f"Total queries with true candidate missing from predictions: {inchikey_true_mismatch}")
    # --- TorchMetrics Calculation ---
    preds_tensor = torch.cat(all_preds_torch)
    target_tensor = torch.cat(all_targets_torch)
    indexes_tensor = torch.cat(all_indexes_torch)

    k_vals = [1, 5, 10, 20]
    hit_rates = {}
    print("Calculating final hit rates with torchmetrics...")
    for k in k_vals:
        metric = RetrievalHitRate(top_k=k)
        metric.update(preds_tensor, target_tensor, indexes=indexes_tensor)
        hit_rates[f"hit_rate_at_{k}"] = metric.compute().item()
        print(f"Hit Rate @{k}: {hit_rates[f'hit_rate_at_{k}']:.4f}")

    final_output = {
        "dataset": dataset,
        "dist_fn": dist_fn,
        "metrics": hit_rates,
        "individuals": sorted(output_entries_for_df, key=lambda x: x["ind_recovered"]),
    }

    if output_entries_for_df:
        df = pd.DataFrame(output_entries_for_df)
        df['ion'] = df['spec_name'].map(name_to_ion)
        if "instrument" in query_df.columns:
            df['instrument'] = df['spec_name'].map(name_to_instrument)
        for k in k_vals:
            df[f'top_{k}'] = df['ind_recovered'] <= k

        out_parent = Path(outfile).parent
        out_stem = Path(outfile).stem
        outfile_tuples = [("ion", out_parent / f"{out_stem}_grouped_ion_torchmetrics.tsv")]
        if "instrument" in query_df.columns:
            outfile_tuples.append(("instrument", out_parent / f"{out_stem}_grouped_inst_torchmetrics.tsv"))

        for group_key, out_name in outfile_tuples:
            df_grouped = pd.concat([df.groupby(group_key).mean(numeric_only=True), df.groupby(group_key).size().rename("num_examples")], axis=1)
            all_mean = df.mean(numeric_only=True)
            all_mean["num_examples"] = len(df)
            all_mean.name = "avg"
            df_grouped = pd.concat([df_grouped, all_mean.to_frame().T], axis=0)
            df_grouped.to_csv(out_name, sep="\t")

    with open(outfile, "w") as fp:
        yaml.dump(final_output, fp, indent=2)
        print(f"Saved final results to {outfile}")

if __name__ == "__main__":
    """__main__"""
    args = get_args()
    main(args)
