import ast
from pathlib import Path
import json
import argparse
import yaml
import pickle
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from typing import Dict, List
import copy

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
    parser.add_argument(
        "--binned-pred",
        action="store_true",
        default=False,
        help="If true, the spec predictions are binned",
    )
    parser.add_argument(
        "--dataset-labels",
        default="labels.tsv",
        help="The labels file for the dataset",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="If true, only run on a subset of the data",
    )
    return parser.parse_args()


def process_spec_file(spec_name, name_to_colli: dict, spec_dir: Path, num_bins: int = -1, upper_limit: int = -1,
                      binned_spec: bool=True):
    """process_spec_file."""
    if binned_spec:
        assert num_bins > 0
        assert upper_limit > 0

    if spec_dir.suffix == '.hdf5': # is h5file
        spec_h5 = common.HDF5Dataset(spec_dir)
    else: # is directory
        spec_h5 = None
    return_dict = {}
    for colli_label in name_to_colli[spec_name]:
        if spec_h5 is not None: # is h5file
            spec_file = f"{spec_name}_collision {colli_label} eV.json"
            if not spec_file in spec_h5:
                spec_file_adj = f"{spec_name}_collision {colli_label} eV [imputed].json"
                if not spec_file_adj in spec_h5:
                    print(f"Cannot find spec {spec_file}")
                    return_dict[colli_label] = np.zeros(num_bins) if binned_spec else np.zeros((0, 2))
                    continue
                else:
                    loaded_json = json.loads(spec_h5.read_str(spec_file_adj))
            else:
                loaded_json = json.loads(spec_h5.read_str(spec_file))
        else: # is directory
            spec_file = spec_dir / f"{spec_name}.json"
            if not spec_file.exists():
                print(f"Cannot find spec {spec_file}")
                return np.zeros(num_bins) if binned_spec else np.zeros((0, 2))
            loaded_json = json.load(open(spec_file, "r"))

        if loaded_json.get("output_tbl") is None:
            return_dict[colli_label] = None
            continue

        # Load with adduct involved
        mz = loaded_json["output_tbl"]["mono_mass"]
        inten = loaded_json["output_tbl"]["ms2_inten"]
        spec_ar = np.vstack([mz, inten]).transpose(1, 0)
        if binned_spec:
            binned = common.bin_spectra([spec_ar], num_bins, upper_limit)
            avged = binned[0]
            return_dict[colli_label] = avged
        else:
            return_dict[colli_label] = spec_ar
    return return_dict


def dist_bin(cand_preds_dict: List[Dict], true_spec_dict: dict, sparse=True, ignore_peak=None, func='cos', selected_evs=None, agg=True) -> np.ndarray:
    """cos_dist for binned spectrum

    Args:
        cand_preds_dict: List of candidates
        true_spec_dict:
        ignore_peak: ignore peaks whose indices are larger than this value

    Returns:
        np.ndarray:
    """
    dist = []
    true_npeaks = []
    ## sampled_evs = np.random.choice(evs, 3, p = ())
    if selected_evs:
        true_spec_dict = {k: v for k, v in true_spec_dict.items() if str(k) in selected_evs}
    for idx, colli_eng in enumerate(true_spec_dict.keys()):
        cand_preds = np.stack([i[colli_eng] for i in cand_preds_dict], axis=0)
        true_spec = true_spec_dict[colli_eng]

        if sparse:
            # sparse_output
            pred_specs = np.zeros((cand_preds.shape[0], true_spec.shape[0]))
            inds = cand_preds[:, :, 0].astype(int)
            pos_1 = np.ones(inds.shape) * np.arange(inds.shape[0])[:, None]
            pred_specs[pos_1.flatten().astype(int), inds.flatten()] = cand_preds[
                :, :, 1
            ].flatten()
        else:
            pred_specs = cand_preds

        if ignore_peak:
            pred_specs[:, int(ignore_peak):] = 0
            true_spec = copy.deepcopy(true_spec)
            true_spec[int(ignore_peak):] = 0

        true_npeaks.append(np.sum(true_spec > 0))

        if func == 'cos':
            norm_pred = norm(pred_specs, axis=-1) + 1e-22  # , ord=2) #+ 1e-22
            norm_true = norm(true_spec, axis=-1) + 1e-22  # , ord=2) #+ 1e-22

            # Cos
            dist.append(1 - np.dot(pred_specs, true_spec) / (norm_pred * norm_true))
        elif func == 'entropy':
            def norm_peaks(prob):
                return prob / (prob.sum(axis=-1, keepdims=True) + 1e-22)

            def entropy(prob):
                # assert np.all(np.abs(prob.sum(axis=-1) - 1) < 5e-3), f"Diff to 1: {np.max(np.abs(prob.sum(axis=-1) - 1))}"
                return -np.sum(prob * np.log(prob + 1e-22), axis=-1)

            norm_pred = norm_peaks(pred_specs)
            norm_true = norm_peaks(true_spec)
            entropy_pred = entropy(norm_pred)
            entropy_targ = entropy(norm_true)
            entropy_mix = entropy((norm_pred + norm_true) / 2)
            dist.append((2 * entropy_mix - entropy_pred - entropy_targ) / np.log(4))

        elif func == "emd":
            import ot
            bins = np.linspace(0, 1500, 15000, dtype=np.float64)
            def norm_peaks(prob):
                return prob / (prob.sum(axis=-1, keepdims=True) + 1e-22)
            norm_pred = norm_peaks(pred_specs)
            norm_true = norm_peaks(true_spec)
            emds = []
            for i in tqdm(range(norm_pred.shape[0])):
                # this takes 10 seconds
                # TODO: figure out how one can incorporate the entropy weighting into matrix?
                #emd = ot.emd2_1d(x_a=bins, x_b=bins, a = -norm_pred[i,:] * np.log(norm_pred[i,:]), 
                #                                     b = -norm_true * np.log(norm_true), metric='sqeuclidean')
                # takes like 10 minutes..
                emd = ot.emd2(norm_pred[i,:], norm_true, np.abs(bins[:, None] - bins))
                #emd = ot.sinkhorn2(norm_pred[i,: ], norm_true, np.abs(bins[:, None] - bins), reg=0)
                # print(emd)
                #print(np.abs(np.cumsum(norm_pred[i, :], axis=-1) - np.cumsum(norm_true)) @ np.diff(bins, append=15000))
                emds.append(emd)
            # closed form for 1p 1-d emd
            # emd = np.abs(np.cumsum(norm_pred, axis=-1) - np.cumsum(norm_true)) @ np.diff(bins, append=15000)

            #emd = -np.exp(-emd) # top 3? # super small values?
            # emd = 1 - 1/emd # top 4, 0.85 to 1
            # emd = np.log1p(emd)
            # emd = np.tanh(emd)  # saturates everything, not good.
            # the relative distances end up making a difference below b/c of dot product!
            dist.append(emds)


    dist = np.array(dist)  # num of colli energy x number of candidates
    # if >=5 peaks: weight=4, elif >=1 peaks: weight=1, else: weight=0
    weights = (np.array(true_npeaks) >= 5) * 3 + (np.array(true_npeaks) >= 1) * 1
    # weights = np.ones(dist.shape[0])
    weights = weights / weights.sum()

    if agg:
        return np.sum(dist * weights[:, None], axis=0)  # number of candidates
    else:
        # return both
        dist = dist[weights > 0] # exclude any objectives that have zero-weights based on filter
        return dist, np.sum(dist * weights[:, None], axis=0)  # number of candidates

# define cosine/entropy functions
cos_dist_bin = partial(dist_bin, func='cos')
entropy_dist_bin = partial(dist_bin, func='entropy')
emd_dist_bin = partial(dist_bin, func='emd')


def cos_dist_hun(cand_preds_dict: List[Dict], true_spec_dict: dict, parent_mass: float, ignore_peak=False) -> np.ndarray:
    """cos_dist for sparse spectrum using Hungarian algorithm to match peaks

    Args:
        cand_preds_dict: List of candidates
        true_spec_dict:
        ignore_peak: ignore peaks whose indices are larger than this value

    Returns:
        np.ndarray:
    """
    dist = 0
    for idx, colli_eng in enumerate(true_spec_dict.keys()):
        cand_preds = common.np_stack_padding([i[colli_eng] for i in cand_preds_dict], axis=0)
        true_spec = true_spec_dict[colli_eng]

        if ignore_peak:
            cand_preds, true_spec = copy.deepcopy(cand_preds), copy.deepcopy(true_spec)
            cand_preds[cand_preds[:, :, 0] > parent_mass - 1, 1] = 0
            true_spec[true_spec[:, 0] > parent_mass - 1, 1] = 0

        norm_pred = norm(cand_preds[:, :, 1], axis=-1) + 1e-22
        norm_true = norm(true_spec[:, 1], axis=-1) + 1e-22

        tol = parent_mass * 2e-5  # 20ppm tolerance
        mask = np.abs(cand_preds[:, :, None, 0] - true_spec[None, None, :, 0]) < tol
        score = cand_preds[:, :, None, 1] * true_spec[None, None, :, 1] / (norm_pred[:, None, None] * norm_true)
        score = score * mask
        assign = pygm.hungarian(score)
        dist += 1 - np.sum(assign * score, axis=(1, 2))

    return dist / len(true_spec_dict)


def rank_test_entry(
    cand_ikeys,
    cand_preds,
    true_spec,
    true_ikey,
    spec_name,
    true_smiles,
    parent_mass,
    parent_mass_idx,
    dist_fn="cos",
    binned_pred=True,
    cand_smiles=None,
    **kwargs,
):
    """rank_test_entry.

    Args:
        cand_ikeys:
        cand_preds:
        true_spec:
        true_ikey:
        spec_name:
        true_smiles:
        kwargs:
    """
    if dist_fn == "cos" and binned_pred:
        dist = cos_dist_bin(cand_preds_dict=cand_preds, true_spec_dict=true_spec, ignore_peak=parent_mass_idx)
    elif dist_fn == "cos" and not binned_pred:
        dist = cos_dist_hun(cand_preds_dict=cand_preds, true_spec_dict=true_spec, parent_mass=parent_mass, ignore_peak=parent_mass_idx is not None)
    elif dist_fn == "entropy" and binned_pred:
        dist = entropy_dist_bin(cand_preds_dict=cand_preds, true_spec_dict=true_spec, ignore_peak=parent_mass_idx)
    elif dist_fn == "random":
        dist = np.random.randn(cand_preds.shape[0])
    else:
        raise NotImplementedError()
    
    # if cand_ikeys has hyphen, cut first
    cand_ikeys = np.array([i.split('-')[0] for i in cand_ikeys])
    true_ind = np.argwhere(cand_ikeys == true_ikey).flatten()

    # Now need to find which position 0 is in  --> should be 28th
    resorted = np.argsort(dist)
    # inds_found = np.argsort(resorted)

    # resorted_dist = dist[resorted]
    # NOTE: resorted is out of bounds
    resorted_ikeys = cand_ikeys[resorted]
    resorted_dist = dist[resorted]
    if len(true_ind) > 1:
        print(f"Multiple true indices: {true_ind}")
        raise
    elif len(true_ind) == 0:
        print(f"True index not found: {true_ikey}")
        return {
            "ind_recovered": np.nan,
            "total_decoys": len(resorted_ikeys),
            "mass": np.nan,
            "mass_bin": np.nan,
            "peak_bin_avg": np.nan,
            "peak_bin_max": np.nan,
            "peak_bin_min": np.nan,
            "true_dist": np.nan,
            "spec_name": str(spec_name),
        }
    assert len(true_ind) == 1

    true_ind = true_ind[0]

    true_dist = dist[true_ind]
    # ind_found = inds_found[true_ind]
    # tie_shift = np.sum(true_dist == dist) - 1
    # ind_found_init = ind_found
    # ind_found = ind_found + tie_shift
    ind_found = np.argwhere(resorted_dist == true_dist).flatten()[-1]

    # Add 1 in case it was first to be top 1 not zero
    ind_found = ind_found + 1

    true_mass = common.mass_from_smi(true_smiles)
    mass_bin = common.bin_mass_results(true_mass)
    peak_bin_avg = common.bin_peak_results(true_spec, binned_spec=binned_pred, reduction='mean')
    peak_bin_max = common.bin_peak_results(true_spec, binned_spec=binned_pred, reduction='max')
    peak_bin_min = common.bin_peak_results(true_spec, binned_spec=binned_pred, reduction='min')

    return {
        "ind_recovered": float(ind_found),
        "total_decoys": len(resorted_ikeys),
        "mass": float(true_mass),
        "mass_bin": mass_bin,
        "peak_bin_avg": peak_bin_avg,
        "peak_bin_max": peak_bin_max,
        "peak_bin_min": peak_bin_min,
        "true_dist": float(true_dist),
        "spec_name": str(spec_name),
    }


def main(args):
    """main."""
    dataset = args.dataset
    labels_file = args.dataset_labels
    debug = args.debug
    formula_dir_name = args.formula_dir_name
    dist_fn = args.dist_fn
    ignore_parent_peak = args.ignore_parent_peak
    binned_pred = args.binned_pred
    data_folder = Path(f"data/spec_datasets/{dataset}")
    form_folder = data_folder / f"subformulae/{formula_dir_name}/"
    data_df = pd.read_csv(data_folder / labels_file, sep="\t")

    name_to_ikey = dict(data_df[["spec", "inchikey"]].values)
    name_to_smi = dict(data_df[["spec", "smiles"]].values)
    name_to_ion = dict(data_df[["spec", "ionization"]].values)
    name_to_colli = dict(data_df[["spec", "collision_energies"]].values)

    from rdkit import Chem
    pred_file = Path(args.pred_file)
    outfile = args.outfile
    if outfile is None:
        outfile = pred_file.parent / f"rerank_eval_{dist_fn}.yaml"
        outfile_grouped_ion = (
            pred_file.parent / f"rerank_eval_grouped_ion_{dist_fn}.tsv"
        )
        outfile_grouped_mass = (
            pred_file.parent / f"rerank_eval_grouped_mass_{dist_fn}.tsv"
        )
        outfile_grouped_peak = (
                pred_file.parent / f"rerank_eval_grouped_npeak_{dist_fn}.tsv"
        )
    else:
        outfile = Path(outfile)
        outfile_grouped_ion = outfile.parent / f"{outfile.stem}_grouped_ion.tsv"
        outfile_grouped_mass = outfile.parent / f"{outfile.stem}_grouped_mass.tsv"
        outfile_grouped_peak = outfile.parent / f"{outfile.stem}_grouped_npeak.tsv"

    pred_specs = common.HDF5Dataset(pred_file)
    if binned_pred:
        upper_limit = pred_specs.attrs["upper_limit"]
        num_bins = pred_specs.attrs["num_bins"]
    use_sparse = pred_specs.attrs["sparse_out"]

    pred_spec_ars = []
    pred_ikeys = []
    pred_spec_names = []
    pred_smiles = []
    # iterate over h5 layers
    for pred_spec_obj in pred_specs.h5_obj.values():
        for smiles_obj in pred_spec_obj.values():
            ikey = None
            spec_dict = {}
            name = None
            smi = None
            for collision_eng_key, collision_eng_obj in smiles_obj.items():
                if name is None:
                    name = collision_eng_obj.attrs['spec_name']
                if ikey is None:
                    ikey = collision_eng_obj.attrs['ikey']
                if smi is None:
                    smi = collision_eng_obj.attrs['smiles']
                    smi = common.rm_stereo(smi)
                    smi = common.smiles_from_inchi(common.inchi_from_smiles(smi))
                    ikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
                collision_eng_key = common.get_collision_energy(collision_eng_key)
                spec_dict[collision_eng_key] = collision_eng_obj['spec'][:]
            pred_spec_ars.append(spec_dict)
            pred_ikeys.append(ikey)
            pred_smiles.append(smi)
            pred_spec_names.append(name)

    pred_spec_ars = np.array(pred_spec_ars)
    pred_ikeys = np.array(pred_ikeys)
    pred_smiles = np.array(pred_smiles)
    pred_spec_names = np.array(pred_spec_names)
    # print(len(pred_spec_ars), len(pred_ikeys), len(pred_spec_names))
    pred_spec_names_unique = np.unique(pred_spec_names)
    if debug:
        pred_spec_names_unique = pred_spec_names_unique[-5:]

    # pred_specs = pickle.load(open(pred_file, "rb"))
    # pred_spec_ars = pred_specs["preds"]
    # pred_ikeys = np.array(pred_specs["ikeys"])
    # pred_spec_names = np.array(pred_specs["spec_names"])
    # pred_spec_names_unique = np.unique(pred_spec_names)
    # use_sparse = pred_specs["sparse_out"]
    # if binned_pred:
    #     upper_limit = pred_specs["upper_limit"]
    #     num_bins = pred_specs["num_bins"]

    # if args.merged_specs:  # only keep collision_energy == nan
    #     for spec_name in name_to_colli.keys():  # filter true spec
    #         colli_engs = ast.literal_eval(name_to_colli[spec_name])
    #         new_colli_engs = []
    #         for colli_key in colli_engs:
    #             if 'nan' in colli_key:
    #                 new_colli_engs.append(colli_key)
    #         if len(new_colli_engs) == 0:
    #             new_colli_engs.append('nan')
    #         name_to_colli[spec_name] = new_colli_engs
    #
    #     for idx in range(len(pred_spec_ars)):  # filter predicted spec
    #         pred_spec_ars[idx] = {k: v for k, v in pred_spec_ars[idx].items() if 'nan' in k}
    #
    # else:

    # only keep collision_energy != nan
    for spec_name in name_to_colli.keys():  # filter true spec
        colli_engs = ast.literal_eval(name_to_colli[spec_name])
        new_colli_engs = []
        for colli_key in colli_engs:
            if 'nan' not in colli_key:
                new_colli_engs.append(colli_key)
        name_to_colli[spec_name] = new_colli_engs

    for idx in range(len(pred_spec_ars)):  # filter predicted spec
        pred_spec_ars[idx] = {k: v for k, v in pred_spec_ars[idx].items() if 'nan' not in k}

    # Only use sparse valid for now
    assert use_sparse
    if binned_pred:
        read_spec = partial(
            process_spec_file,
            name_to_colli=name_to_colli,
            num_bins=num_bins,
            upper_limit=upper_limit,
            spec_dir=form_folder,
            binned_spec=True,  # load binned true spectrum
        )
    else:
        read_spec = partial(
            process_spec_file,
            name_to_colli=name_to_colli,
            spec_dir=form_folder,
            binned_spec=False,  # load sparse true spectrum
        )
    true_specs = common.chunked_parallel(
        pred_spec_names_unique,
        read_spec,
        chunks=100,
        max_cpu=16,
        task_name="Collecting true spectra"
    )
    name_to_spec = dict(zip(pred_spec_names_unique, true_specs))

    # Create a list of dicts, bucket by mass, etc.
    all_entries = []
    for spec_name in tqdm(pred_spec_names_unique):

        # Get candidates
        bool_sel = pred_spec_names == spec_name
        cand_ikeys = pred_ikeys[bool_sel]
        cand_smiles = pred_smiles[bool_sel]
        cand_preds = np.array(pred_spec_ars)[bool_sel]
        true_spec = name_to_spec[spec_name]
        true_smi = name_to_smi[spec_name]
        true_ion = name_to_ion[spec_name]
        parent_mass = common.mass_from_smi(true_smi) + common.ion2mass[true_ion]
        new_entry = {
            "cand_ikeys": cand_ikeys,
            "cand_preds": cand_preds,
            "cand_smiles": cand_smiles,
            "true_spec": true_spec,
            "true_smiles": true_smi,
            "true_ikey": name_to_ikey[spec_name],
            "spec_name": spec_name,
            "parent_mass_idx": (parent_mass - 1) * num_bins / upper_limit if ignore_parent_peak else None,
            "parent_mass": parent_mass,
        }

        if true_spec is None:
            continue
        all_entries.append(new_entry)

    rank_test_entry_ = partial(rank_test_entry, dist_fn=dist_fn, binned_pred=binned_pred)
    # all_out = common.chunked_parallel(
    #     pred_spec_names_unique,
    #     read_spec,
    #     chunks=100,
    #     max_cpu=16,
    #     task_name="Collecting true spectra"
    # )
    all_out = [rank_test_entry_(**test_entry) for test_entry in all_entries]

    # Compute avg and individual stats
    k_vals = list(range(1, 21)) # changed from 10 to 21 so that we can rank top 20 for MSG
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
        "dist_fn": dist_fn,
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
        # print(out_str)
        fp.write(out_str)


if __name__ == "__main__":
    """__main__"""
    args = get_args()
    main(args)
