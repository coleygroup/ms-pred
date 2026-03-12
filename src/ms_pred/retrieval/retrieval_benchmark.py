import ast
from pathlib import Path
import json
import argparse
import yaml
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List
import copy

import pygmtools as pygm

import numpy as np
from numpy.linalg import norm

import pandas as pd

import ms_pred.common as common


_WORKER_PREDSPEC_DB = None
_SCORE_WORKER_STATE = None


def _read_and_bin_pred_spec_name(
    entry,
    pred_file: str,
    num_bins: int,
    upper_limit: int,
    pool_fn: str = "add",
):
    global _WORKER_PREDSPEC_DB
    if _WORKER_PREDSPEC_DB is None:
        _WORKER_PREDSPEC_DB = common.PredSpecDB(h5_path=Path(pred_file), mode="r")

    spec_name = entry
    spec_dict, has_remark = _WORKER_PREDSPEC_DB.read_from_name(spec_name)
    output = []

    if not has_remark:
        return output

    for cand_ikey, ce_spec_dict in spec_dict.items():
        pred_spec = {}
        for ce, spec_data in ce_spec_dict.items():
            if "nan" in ce:
                continue

            has_matching_bins = (
                spec_data.has_binned_spec
                and getattr(spec_data, "_num_bins", None) == num_bins
                and getattr(spec_data, "_mass_upper_limit", None) == upper_limit
            )
            if has_matching_bins:
                binned = spec_data.binned_spec
            else:
                binned = spec_data.bin_spectrum(
                    mass_upper_limit=upper_limit,
                    num_bins=num_bins,
                    pool_fn=pool_fn,
                )
            pred_spec[ce] = binned.astype(np.float32, copy=False)

        output.append((pred_spec, cand_ikey.strip("ikey "), spec_name.strip("pred_")))
    return output


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
    parser.add_argument("--num-bins", default=15000, help="Number of bins for spectra")
    parser.add_argument("--upper-limit", default=1500, help="Largest m/z value")
    parser.add_argument("--pool-fn", default="add", choices=["add", "max"])
    parser.add_argument("--num-cpu-workers", default=16, type=int)
    return parser.parse_args()


def process_spec_file(
    spec_name,
    name_to_colli: dict,
    spec_dir: Path,
    num_bins: int = -1,
    upper_limit: int = -1,
    binned_spec: bool = True,
):
    """process_spec_file."""
    if binned_spec:
        assert num_bins > 0
        assert upper_limit > 0

    if spec_dir.suffix == ".hdf5":
        spec_h5 = common.HDF5Dataset(spec_dir)
    else:
        spec_h5 = None
    return_dict = {}
    shared_loaded_json = None
    shared_spec_ar = None
    shared_binned = None
    for colli_label in name_to_colli[spec_name]:
        if spec_h5 is not None:
            spec_file = f"{spec_name}_collision {colli_label}.json"
            if spec_file not in spec_h5:
                print(f"Cannot find spec {spec_file}")
                return_dict[colli_label] = (
                    np.zeros(num_bins, dtype=np.float32)
                    if binned_spec
                    else np.zeros((0, 2), dtype=np.float32)
                )
                continue
            loaded_json = json.loads(spec_h5.read_str(spec_file))
        else:
            if shared_loaded_json is None:
                spec_file = spec_dir / f"{spec_name}.json"
                if not spec_file.exists():
                    print(f"Cannot find spec {spec_file}")
                    return (
                        np.zeros(num_bins, dtype=np.float32)
                        if binned_spec
                        else np.zeros((0, 2), dtype=np.float32)
                    )
                with open(spec_file, "r") as fp:
                    shared_loaded_json = json.load(fp)

                if shared_loaded_json.get("output_tbl") is not None:
                    mz = shared_loaded_json["output_tbl"]["mono_mass"]
                    inten = shared_loaded_json["output_tbl"]["ms2_inten"]
                    shared_spec_ar = np.vstack([mz, inten]).transpose(1, 0)
                    if binned_spec:
                        shared_binned = common.bin_spectra(
                            [shared_spec_ar], num_bins, upper_limit
                        )[0].astype(np.float32, copy=False)
            loaded_json = shared_loaded_json

        if loaded_json.get("output_tbl") is None:
            return_dict[colli_label] = None
            continue

        if binned_spec:
            if spec_h5 is None:
                return_dict[colli_label] = shared_binned
            else:
                mz = loaded_json["output_tbl"]["mono_mass"]
                inten = loaded_json["output_tbl"]["ms2_inten"]
                spec_ar = np.vstack([mz, inten]).transpose(1, 0)
                binned = common.bin_spectra([spec_ar], num_bins, upper_limit)
                return_dict[colli_label] = binned[0].astype(np.float32, copy=False)
        else:
            if spec_h5 is None:
                return_dict[colli_label] = shared_spec_ar
            else:
                mz = loaded_json["output_tbl"]["mono_mass"]
                inten = loaded_json["output_tbl"]["ms2_inten"]
                spec_ar = np.vstack([mz, inten]).transpose(1, 0)
                return_dict[colli_label] = spec_ar
    return return_dict


def dist_bin(
    cand_preds_dict: List[Dict],
    true_spec_dict: dict,
    sparse=True,
    ignore_peak=None,
    func="cos",
    selected_evs=None,
    agg=True,
) -> np.ndarray:
    """distance function for binned spectrum"""
    dist = []
    true_npeaks = []
    if selected_evs:
        true_spec_dict = {
            k: v for k, v in true_spec_dict.items() if str(k) in selected_evs
        }

    true_spec_dict = {common.get_collision_energy(k): v for k, v in true_spec_dict.items()}
    cand_preds_dict = [
        {common.get_collision_energy(k): v for k, v in cand_dict.items()}
        for cand_dict in cand_preds_dict
    ]

    for _, colli_eng in enumerate(true_spec_dict.keys()):
        cand_preds = np.stack([i[colli_eng] for i in cand_preds_dict], axis=0)
        true_spec = true_spec_dict[colli_eng]

        if sparse:
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

        if func == "cos":
            norm_pred = norm(pred_specs, axis=-1) + 1e-22
            norm_true = norm(true_spec, axis=-1) + 1e-22
            dist.append(1 - np.dot(pred_specs, true_spec) / (norm_pred * norm_true))
        elif func == "entropy":
            def norm_peaks(prob):
                return prob / (prob.sum(axis=-1, keepdims=True) + 1e-22)

            def entropy(prob):
                return -np.sum(prob * np.log(prob + 1e-22), axis=-1)

            norm_pred = norm_peaks(pred_specs)
            norm_true = norm_peaks(true_spec)
            zeros = pred_specs.sum(axis=-1) == 0
            entropy_pred = entropy(norm_pred)
            entropy_targ = entropy(norm_true)
            entropy_mix = entropy((norm_pred + norm_true) / 2)
            entropy_dists = (2 * entropy_mix - entropy_pred - entropy_targ) / np.log(4)
            entropy_dists[zeros] = 1
            dist.append(entropy_dists)

    dist = np.array(dist)
    weights = (np.array(true_npeaks) >= 5) * 3 + (np.array(true_npeaks) >= 1) * 1
    weights = weights / weights.sum()

    if agg:
        return np.sum(dist * weights[:, None], axis=0)
    else:
        dist = dist[weights > 0]
        return dist, np.sum(dist * weights[:, None], axis=0)


cos_dist_bin = partial(dist_bin, func="cos")
entropy_dist_bin = partial(dist_bin, func="entropy")


def cos_dist_hun(
    cand_preds_dict: List[Dict],
    true_spec_dict: dict,
    parent_mass: float,
    ignore_peak=False,
) -> np.ndarray:
    """cos_dist for sparse spectrum using Hungarian algorithm to match peaks"""
    dist = 0
    for _, colli_eng in enumerate(true_spec_dict.keys()):
        cand_preds = common.np_stack_padding(
            [i[colli_eng] for i in cand_preds_dict], axis=0
        )
        true_spec = true_spec_dict[colli_eng]

        if ignore_peak:
            cand_preds, true_spec = copy.deepcopy(cand_preds), copy.deepcopy(true_spec)
            cand_preds[cand_preds[:, :, 0] > parent_mass - 1, 1] = 0
            true_spec[true_spec[:, 0] > parent_mass - 1, 1] = 0

        norm_pred = norm(cand_preds[:, :, 1], axis=-1) + 1e-22
        norm_true = norm(true_spec[:, 1], axis=-1) + 1e-22

        tol = parent_mass * 2e-5
        mask = np.abs(cand_preds[:, :, None, 0] - true_spec[None, None, :, 0]) < tol
        score = (
            cand_preds[:, :, None, 1]
            * true_spec[None, None, :, 1]
            / (norm_pred[:, None, None] * norm_true)
        )
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
    **kwargs,
):
    """rank_test_entry."""
    if dist_fn == "cos" and binned_pred:
        dist = cos_dist_bin(
            cand_preds_dict=cand_preds,
            true_spec_dict=true_spec,
            sparse=False,
            ignore_peak=parent_mass_idx,
        )
    elif dist_fn == "cos" and not binned_pred:
        dist = cos_dist_hun(
            cand_preds_dict=cand_preds,
            true_spec_dict=true_spec,
            parent_mass=parent_mass,
            ignore_peak=parent_mass_idx is not None,
        )
    elif dist_fn == "entropy" and binned_pred:
        dist = entropy_dist_bin(
            cand_preds_dict=cand_preds,
            true_spec_dict=true_spec,
            sparse=False,
            ignore_peak=parent_mass_idx,
        )
    elif dist_fn == "random":
        dist = np.random.randn(cand_preds.shape[0])
    else:
        raise NotImplementedError()

    true_ind = np.argwhere(cand_ikeys == true_ikey).flatten()
    resorted = np.argsort(dist)
    resorted_ikeys = cand_ikeys[resorted]
    resorted_dist = dist[resorted]

    assert len(true_ind) == 1

    true_ind = true_ind[0]
    true_dist = dist[true_ind]
    ind_found = np.argwhere(resorted_dist == true_dist).flatten()[-1]
    ind_found = ind_found + 1

    true_mass = common.mass_from_smi(true_smiles)
    mass_bin = common.bin_mass_results(true_mass)

    if binned_pred:
        num_peaks_avg = np.mean([np.sum(sp > 0) for sp in true_spec.values()])
    else:
        num_peaks_avg = np.mean([np.sum(sp[:, 1] > 0) for sp in true_spec.values()])
    num_collision_engs = len(true_spec)
    peak_bin_avg = common.bin_peak_results(
        true_spec, binned_spec=binned_pred, reduction="mean"
    )
    peak_bin_max = common.bin_peak_results(
        true_spec, binned_spec=binned_pred, reduction="max"
    )
    peak_bin_min = common.bin_peak_results(
        true_spec, binned_spec=binned_pred, reduction="min"
    )

    return_dict = {
        "ind_recovered": float(ind_found),
        "total_decoys": len(resorted_ikeys),
        "mass": float(true_mass),
        "mass_bin": mass_bin,
        "num_peaks_avg": float(num_peaks_avg),
        "num_collision_engs": int(num_collision_engs),
        "peak_bin_avg": peak_bin_avg,
        "peak_bin_max": peak_bin_max,
        "peak_bin_min": peak_bin_min,
        "true_dist": float(true_dist),
        "spec_name": str(spec_name),
    }
    for k in range(0, min(50, len(resorted_dist))):
        return_dict[f"top_{k + 1}_dist"] = resorted_dist[k].item()
    return return_dict


def _score_rank_entry_by_name(spec_name):
    global _SCORE_WORKER_STATE
    if _SCORE_WORKER_STATE is None:
        raise RuntimeError("Score worker state has not been initialized")

    state = _SCORE_WORKER_STATE
    selected_indices = state["pred_name_to_indices"][spec_name]
    cand_ikeys = state["pred_ikeys"][selected_indices]
    cand_preds = np.asarray(
        [state["pred_spec_ars"][i] for i in selected_indices],
        dtype=object,
    )
    true_spec = state["true_spec_by_name"][spec_name]
    true_smi = state["name_to_smi"][spec_name]
    true_ion = state["name_to_ion"][spec_name]
    parent_mass = common.mass_from_smi(true_smi) + common.ion2mass[true_ion]

    return rank_test_entry(
        cand_ikeys=cand_ikeys,
        cand_preds=cand_preds,
        true_spec=true_spec,
        true_ikey=state["name_to_ikey"][spec_name],
        spec_name=spec_name,
        true_smiles=true_smi,
        parent_mass=parent_mass,
        parent_mass_idx=(
            (parent_mass - 1) * state["num_bins"] / state["upper_limit"]
            if state["ignore_parent_peak"]
            else None
        ),
        dist_fn=state["dist_fn"],
        binned_pred=True,
    )


def main(args):
    """main."""
    dataset = args.dataset
    formula_dir_name = args.formula_dir_name
    dist_fn = args.dist_fn
    ignore_parent_peak = args.ignore_parent_peak
    num_bins = args.num_bins
    upper_limit = args.upper_limit
    pool_fn = args.pool_fn
    data_folder = Path(f"data/spec_datasets/{dataset}")
    form_folder = data_folder / f"subformulae/{formula_dir_name}/"
    data_df = pd.read_csv(data_folder / "labels.tsv", sep="\t")
    class_df = pd.read_csv(data_folder / "chemical_class_labels.tsv", sep="\t")

    name_to_ikey = dict(data_df[["spec", "inchikey"]].values)
    name_to_smi = dict(data_df[["spec", "smiles"]].values)
    name_to_ion = dict(data_df[["spec", "ionization"]].values)
    name_to_colli = dict(data_df[["spec", "collision_energies"]].values)
    name_to_class = dict(class_df[["spec", "class"]].values)

    pred_file = Path(args.pred_file)
    outfile = args.outfile
    if outfile is None:
        outfile = pred_file.parent / f"rerank_eval_{dist_fn}.yaml"
        outfile_grouped_ion = pred_file.parent / f"rerank_eval_grouped_ion_{dist_fn}.tsv"
        outfile_grouped_mass = pred_file.parent / f"rerank_eval_grouped_mass_{dist_fn}.tsv"
        outfile_grouped_peak = pred_file.parent / f"rerank_eval_grouped_npeak_{dist_fn}.tsv"
        outfile_grouped_class = pred_file.parent / f"rerank_eval_grouped_class_{dist_fn}.tsv"
    else:
        outfile = Path(outfile)
        outfile_grouped_ion = outfile.parent / f"{outfile.stem}_grouped_ion.tsv"
        outfile_grouped_mass = outfile.parent / f"{outfile.stem}_grouped_mass.tsv"
        outfile_grouped_peak = outfile.parent / f"{outfile.stem}_grouped_npeak.tsv"
        outfile_grouped_class = outfile.parent / f"{outfile.stem}_grouped_class.tsv"

    pred_specs = common.PredSpecDB(h5_path=pred_file, mode="r")

    pred_spec_ars = []
    pred_ikeys = []
    pred_name_to_indices = defaultdict(list)

    def collect_pred_outputs(out_iter):
        for out_list in out_iter:
            for pred_spec, pred_ikey, pred_name in out_list:
                pred_idx = len(pred_spec_ars)
                pred_spec_ars.append(pred_spec)
                pred_ikeys.append(pred_ikey)
                pred_name_to_indices[pred_name].append(pred_idx)

    pred_names = list(pred_specs.get_all_names())
    bin_name_fn = partial(
        _read_and_bin_pred_spec_name,
        pred_file=str(pred_file),
        num_bins=num_bins,
        upper_limit=upper_limit,
        pool_fn=pool_fn,
    )
    common.chunked_parallel(
        pred_names,
        bin_name_fn,
        chunks=1000,
        max_cpu=args.num_cpu_workers,
        task_name="Binning predicted spectra",
        output_func=collect_pred_outputs,
    )
    pred_ikeys = np.array(pred_ikeys)
    pred_spec_names_unique = sorted(pred_name_to_indices.keys())

    parsed_colli_cache = {}
    for spec_name, colli_raw in name_to_colli.items():
        if colli_raw not in parsed_colli_cache:
            colli_engs = ast.literal_eval(colli_raw)
            parsed_colli_cache[colli_raw] = [
                colli_key for colli_key in colli_engs if "nan" not in colli_key
            ]
        name_to_colli[spec_name] = parsed_colli_cache[colli_raw]

    read_spec = partial(
        process_spec_file,
        name_to_colli=name_to_colli,
        num_bins=num_bins,
        upper_limit=upper_limit,
        spec_dir=form_folder,
        binned_spec=True,
    )

    true_specs = common.chunked_parallel(
        pred_spec_names_unique,
        read_spec,
        chunks=100,
        max_cpu=args.num_cpu_workers,
        task_name="Loading true spectra",
    )

    k_vals = list(range(1, 11))
    running_lists = defaultdict(lambda: [])

    true_spec_by_name = {
        spec_name: true_spec
        for spec_name, true_spec in zip(pred_spec_names_unique, true_specs)
        if true_spec is not None
    }
    score_spec_names = list(true_spec_by_name.keys())

    global _SCORE_WORKER_STATE
    _SCORE_WORKER_STATE = {
        "pred_name_to_indices": pred_name_to_indices,
        "pred_ikeys": pred_ikeys,
        "pred_spec_ars": pred_spec_ars,
        "true_spec_by_name": true_spec_by_name,
        "name_to_smi": name_to_smi,
        "name_to_ion": name_to_ion,
        "name_to_ikey": name_to_ikey,
        "num_bins": num_bins,
        "upper_limit": upper_limit,
        "ignore_parent_peak": ignore_parent_peak,
        "dist_fn": dist_fn,
    }

    with ThreadPoolExecutor(max_workers=args.num_cpu_workers) as executor:
        rank_outputs = list(executor.map(_score_rank_entry_by_name, score_spec_names))

    output_entries = []
    for out in rank_outputs:
        output_entries.append(out)

        for k in k_vals:
            below_k = out["ind_recovered"] <= k
            running_lists[f"top_{k}"].append(below_k)
            out[f"top_{k}"] = below_k
        running_lists["total_decoys"].append(out["total_decoys"])
        running_lists["true_dist"].append(out["true_dist"])
    _SCORE_WORKER_STATE = None

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
        i["class"] = name_to_class[i["spec_name"]]

    df = pd.DataFrame(output_entries)
    tsv_df = df.loc[:, ~df.columns.str.fullmatch(r"top_\d+_dist")]

    for group_key, out_name in zip(
        ["mass_bin", "ion", "peak_bin_avg", "class"],
        [
            outfile_grouped_mass,
            outfile_grouped_ion,
            outfile_grouped_peak,
            outfile_grouped_class,
        ],
    ):
        df_grouped = pd.concat(
            [
                tsv_df.groupby(group_key).mean(numeric_only=True),
                tsv_df.groupby(group_key).size(),
            ],
            axis=1,
        )
        df_grouped = df_grouped.rename({0: "num_examples"}, axis=1)

        all_mean = tsv_df.mean(numeric_only=True)
        all_mean["num_examples"] = len(tsv_df)
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