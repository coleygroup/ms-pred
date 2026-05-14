import argparse
import ast
import json
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torchmetrics.retrieval import RetrievalHitRate

import ms_pred.common as common


_WORKER_PREDSPEC_DB = None


def dense_to_sparse_spec(spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Represent a binned spectrum as (bin_indices, intensities)."""
    spec = np.asarray(spec)
    if spec.size == 0:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)
    if spec.ndim == 2:
        if spec.shape[1] != 2:
            raise ValueError(f"Expected sparse spectrum with shape [n, 2], got {spec.shape}")
        vals = spec[:, 1]
        keep = vals > 0
        return spec[keep, 0].astype(np.uint32, copy=False), vals[keep].astype(np.float32, copy=False)

    inds = np.flatnonzero(spec > 0)
    if inds.size == 0:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)
    return inds.astype(np.uint32, copy=False), spec[inds].astype(np.float32, copy=False)


def _sparse_indices_values(spec: np.ndarray, ignore_peak=None) -> Tuple[np.ndarray, np.ndarray]:
    if spec is None:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

    if isinstance(spec, tuple) and len(spec) == 2:
        inds = np.asarray(spec[0], dtype=np.int64)
        vals = np.asarray(spec[1], dtype=np.float32)
        keep = vals > 0
        inds = inds[keep]
        vals = vals[keep]
        if ignore_peak is not None:
            keep = inds < int(ignore_peak)
            inds = inds[keep]
            vals = vals[keep]
        return inds, vals

    spec = np.asarray(spec)
    if spec.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    if spec.ndim == 1:
        inds = np.flatnonzero(spec > 0).astype(np.int64, copy=False)
        vals = spec[inds].astype(np.float32, copy=False)
    else:
        inds = spec[:, 0].astype(np.int64, copy=False)
        vals = spec[:, 1].astype(np.float32, copy=False)
        keep = vals > 0
        inds = inds[keep]
        vals = vals[keep]

    if ignore_peak is not None:
        keep = inds < int(ignore_peak)
        inds = inds[keep]
        vals = vals[keep]
    return inds, vals


def _num_sparse_peaks(spec: np.ndarray) -> int:
    _, vals = _sparse_indices_values(spec)
    return int(np.sum(vals > 0))


def _has_sparse_peaks(spec: np.ndarray) -> bool:
    try:
        return _num_sparse_peaks(spec) > 0
    except Exception:
        return False


def _transform_sparse_intensities(vals: np.ndarray, sqrt_inten: bool = True) -> np.ndarray:
    """Return intensities in the space requested by the distance metric."""
    if sqrt_inten:
        return vals
    return np.square(vals, dtype=np.float32)


def _cosine_sparse(
    cand_preds_dict: List[Dict],
    true_spec: np.ndarray,
    colli_eng,
    ignore_peak=None,
    sqrt_inten=True,
) -> np.ndarray:
    true_inds, true_vals = _sparse_indices_values(true_spec, ignore_peak=ignore_peak)
    true_vals = _transform_sparse_intensities(true_vals, sqrt_inten=sqrt_inten)
    true_norm = float(np.sqrt(np.dot(true_vals, true_vals))) + 1e-22
    true_lookup = {int(ind): float(val) for ind, val in zip(true_inds, true_vals)}

    out = np.ones(len(cand_preds_dict), dtype=np.float32)
    for pred_idx, cand_dict in enumerate(cand_preds_dict):
        cand_pred = cand_dict.get(colli_eng)
        if cand_pred is None:
            continue
        cand_inds, cand_vals = _sparse_indices_values(cand_pred, ignore_peak=ignore_peak)
        if cand_vals.size == 0:
            continue
        cand_vals = _transform_sparse_intensities(cand_vals, sqrt_inten=sqrt_inten)
        pred_norm = float(np.sqrt(np.dot(cand_vals, cand_vals))) + 1e-22
        dot = sum(float(val) * true_lookup.get(int(ind), 0.0) for ind, val in zip(cand_inds, cand_vals))
        out[pred_idx] = 1 - dot / (pred_norm * true_norm)
    return out


def _entropy_from_probs(vals: np.ndarray) -> float:
    vals = vals[vals > 0]
    return float(-np.sum(vals * np.log(vals + 1e-22)))


def _entropy_sparse(
    cand_preds_dict: List[Dict],
    true_spec: np.ndarray,
    colli_eng,
    ignore_peak=None,
    sqrt_inten=True,
) -> np.ndarray:
    true_inds, true_vals = _sparse_indices_values(true_spec, ignore_peak=ignore_peak)
    true_vals = _transform_sparse_intensities(true_vals, sqrt_inten=sqrt_inten)
    true_sum = float(true_vals.sum())
    if true_sum <= 0:
        return np.ones(len(cand_preds_dict), dtype=np.float32)

    q_vals = true_vals / true_sum
    q_lookup = {int(ind): float(val) for ind, val in zip(true_inds, q_vals)}
    h_true = _entropy_from_probs(q_vals)
    out = np.ones(len(cand_preds_dict), dtype=np.float32)

    for pred_idx, cand_dict in enumerate(cand_preds_dict):
        cand_pred = cand_dict.get(colli_eng)
        if cand_pred is None:
            continue
        cand_inds, cand_vals = _sparse_indices_values(cand_pred, ignore_peak=ignore_peak)
        cand_vals = _transform_sparse_intensities(cand_vals, sqrt_inten=sqrt_inten)
        pred_sum = float(cand_vals.sum())
        if pred_sum <= 0:
            continue

        p_vals = cand_vals / pred_sum
        h_pred = _entropy_from_probs(p_vals)
        h_mix = 0.0
        p_lookup = {}
        for ind, p_val in zip(cand_inds, p_vals):
            ind = int(ind)
            p_val = float(p_val)
            p_lookup[ind] = p_lookup.get(ind, 0.0) + p_val

        for ind, p_val in p_lookup.items():
            if ind not in q_lookup:
                mix_val = p_val / 2
                h_mix -= mix_val * np.log(mix_val + 1e-22)
        for ind, q_val in q_lookup.items():
            mix_val = (q_val + p_lookup.get(ind, 0.0)) / 2
            h_mix -= mix_val * np.log(mix_val + 1e-22)

        out[pred_idx] = (2 * h_mix - h_pred - h_true) / np.log(4)
    return out


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="canopus_train_public")
    parser.add_argument("--formula-dir-name", default="subform_20")
    parser.add_argument(
        "--pred-file",
        default="results/ffn_baseline_cos/retrieval/split_1/fp_preds.p",
    )
    parser.add_argument("--outfile", default=None)
    parser.add_argument("--dist-fn", default="cos", choices=["cos", "entropy", "random"])
    parser.add_argument(
        "--ignore-parent-peak",
        action="store_true",
        default=False,
        help="If true, ignore the precursor peak.",
    )
    parser.add_argument("--num-bins", default=15000, type=int)
    parser.add_argument("--upper-limit", default=1500, type=int)
    parser.add_argument("--pool-fn", default="add", choices=["add", "max"])
    parser.add_argument("--num-cpu-workers", default=16, type=int)
    parser.add_argument(
        "--true-labels",
        default=None,
        help="Optional query-label subset.",
    )
    parser.add_argument(
        "--full-labels",
        default=None,
        help="Full candidate-label file.",
    )
    parser.add_argument(
        "--binned-pred",
        action="store_true",
        default=True,
        help="Deprecated flag retained for compatibility. Predictions are always evaluated as binned spectra.",
    )
    parser.add_argument(
        "--no-sqrt-inten",
        dest="sqrt_inten",
        action="store_false",
        default=True,
        help="Square stored sqrt-intensity values before scoring, recovering original intensities.",
    )
    return parser.parse_args()


def default_true_labels_path(data_folder: Path) -> Path:
    labels_path = data_folder / "labels.tsv"
    if labels_path.exists():
        return labels_path
    raise ValueError(
        "Could not infer --true-labels because data/spec_datasets/<dataset>/labels.tsv does not exist."
    )


def normalize_inchikey(ikey: str) -> str:
    return str(ikey).replace("ikey ", "").strip()


def normalize_query_structure(smiles: str, inchikey: str) -> Tuple[str, str]:
    """Normalize query structures the same way retrieval candidates are normalized."""
    norm_smi = common.smi_inchi_round_smi(smiles)
    if norm_smi is None:
        return smiles, normalize_inchikey(inchikey)

    norm_ikey = common.inchikey_from_smiles(norm_smi)
    if not norm_ikey:
        return norm_smi, normalize_inchikey(inchikey)

    return norm_smi, normalize_inchikey(norm_ikey)


def stereo_group_key(ikey: str) -> str:
    ikey = normalize_inchikey(ikey)
    return ikey.split("-")[0]


def normalize_collision_dict(spec_dict: Dict) -> Dict[str, np.ndarray]:
    return {
        common.get_collision_energy(key): value
        for key, value in spec_dict.items()
        if value is not None and "nan" not in str(key)
    }


def _read_pred_spec_name(
    entry: str,
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
            if "nan" in str(ce):
                continue

            has_matching_bins = (
                spec_data.has_binned_spec
                and getattr(spec_data, "_num_bins", None) == num_bins
                and getattr(spec_data, "_mass_upper_limit", None) == upper_limit
            )
            if not has_matching_bins:
                spec_data.bin_spectrum(
                    mass_upper_limit=upper_limit,
                    num_bins=num_bins,
                    pool_fn=pool_fn,
                )
            pred_spec[ce] = spec_data.binned_spec_sparse

        output.append(
            (
                pred_spec,
                normalize_inchikey(cand_ikey),
                spec_name.removeprefix("pred_"),
            )
        )
    return output


def process_spec_file(
    spec_name: str,
    name_to_colli: dict,
    spec_dir: Path,
    num_bins: int = -1,
    upper_limit: int = -1,
    binned_spec: bool = True,
    dataset: str = "nist20",
):
    if binned_spec:
        assert num_bins > 0
        assert upper_limit > 0

    spec_h5 = common.HDF5Dataset(spec_dir) if spec_dir.suffix == ".hdf5" else None
    return_dict = {}
    shared_loaded_json = None
    shared_spec_ar = None
    shared_binned = None

    for colli_label in name_to_colli.get(spec_name, []):
        if spec_h5 is not None:
            if dataset == "msg":
                spec_files = [
                    f"{spec_name}_collision {colli_label} eV.json",
                    f"{spec_name}_collision {colli_label} eV [imputed].json",
                ]
            else:
                spec_files = [f"{spec_name}_collision {colli_label}.json"]

            loaded_json = None
            for spec_file in spec_files:
                if spec_file in spec_h5:
                    loaded_json = json.loads(spec_h5.read_str(spec_file))
                    break

            if loaded_json is None:
                print(f"Cannot find spec for {spec_name} collision {colli_label}")
                return_dict[colli_label] = (
                    np.zeros((0, 2), dtype=np.float32)
                    if binned_spec
                    else np.zeros((0, 2), dtype=np.float32)
                )
                continue
        else:
            if shared_loaded_json is None:
                spec_file = spec_dir / f"{spec_name}.json"
                if not spec_file.exists():
                    print(f"Cannot find spec {spec_file}")
                    missing_spec = (
                        np.zeros((0, 2), dtype=np.float32)
                        if binned_spec
                        else np.zeros((0, 2), dtype=np.float32)
                    )
                    return {
                        colli_label: missing_spec
                        for colli_label in name_to_colli.get(spec_name, [])
                    }
                with open(spec_file, "r") as fp:
                    shared_loaded_json = json.load(fp)
                if shared_loaded_json.get("output_tbl") is not None:
                    mz = shared_loaded_json["output_tbl"]["mono_mass"]
                    inten = shared_loaded_json["output_tbl"]["ms2_inten"]
                    shared_spec_ar = np.vstack([mz, inten]).transpose(1, 0)
                    if binned_spec:
                        shared_binned = dense_to_sparse_spec(
                            common.bin_spectra([shared_spec_ar], num_bins, upper_limit)[0]
                        )
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
                binned = common.bin_spectra([spec_ar], num_bins, upper_limit)[0]
                return_dict[colli_label] = dense_to_sparse_spec(binned)
        else:
            if spec_h5 is None:
                return_dict[colli_label] = shared_spec_ar
            else:
                mz = loaded_json["output_tbl"]["mono_mass"]
                inten = loaded_json["output_tbl"]["ms2_inten"]
                return_dict[colli_label] = np.vstack([mz, inten]).transpose(1, 0)
    return return_dict


def dist_bin(
    cand_preds_dict: List[Dict],
    true_spec_dict: dict,
    sparse: bool = True,
    ignore_peak=None,
    func: str = "cos",
    sqrt_inten: bool = True,
    selected_evs: Optional[Iterable[str]] = None,
    agg: bool = True,
) -> np.ndarray:
    dist = []
    true_npeaks = []

    true_spec_dict = normalize_collision_dict(true_spec_dict)
    cand_preds_dict = [normalize_collision_dict(cand_dict) for cand_dict in cand_preds_dict]

    if selected_evs is not None:
        selected_evs = {str(common.get_collision_energy(ev)) for ev in selected_evs}
        true_spec_dict = {
            key: value for key, value in true_spec_dict.items() if str(key) in selected_evs
        }

    for colli_eng, true_spec in true_spec_dict.items():
        true_npeaks.append(_num_sparse_peaks(true_spec))
        if func == "cos":
            dist.append(
                _cosine_sparse(
                    cand_preds_dict=cand_preds_dict,
                    true_spec=true_spec,
                    colli_eng=colli_eng,
                    ignore_peak=ignore_peak,
                    sqrt_inten=sqrt_inten,
                )
            )
        elif func == "entropy":
            dist.append(
                _entropy_sparse(
                    cand_preds_dict=cand_preds_dict,
                    true_spec=true_spec,
                    colli_eng=colli_eng,
                    ignore_peak=ignore_peak,
                    sqrt_inten=sqrt_inten,
                )
            )
        else:
            raise NotImplementedError()

    dist = np.array(dist)
    if dist.shape[0] == 0:
        return np.ones(len(cand_preds_dict), dtype=np.float32)

    weights = (np.array(true_npeaks) >= 5) * 3 + (np.array(true_npeaks) >= 1) * 1
    weights = weights / (weights.sum() + 1e-9)
    if agg:
        return np.sum(dist * weights[:, None], axis=0)

    dist = dist[weights > 0]
    return dist, np.sum(dist * weights[:, None], axis=0)


cos_dist_bin = partial(dist_bin, func="cos")
entropy_dist_bin = partial(dist_bin, func="entropy")


def rank_from_distances(dist: np.ndarray, target_mask: np.ndarray) -> Tuple[int, np.ndarray]:
    sorted_indices = np.argsort(dist, kind="stable")
    sorted_targets = target_mask[sorted_indices]
    recovered = np.flatnonzero(sorted_targets)
    if recovered.size == 0:
        raise ValueError("True candidate was not present in the ranked candidate set.")
    return int(recovered[0] + 1), sorted_indices


def build_output_entry(
    spec_name: str,
    true_spec: dict,
    true_smiles: str,
    dist_full: np.ndarray,
    target_mask: np.ndarray,
    candidate_count: int,
    full_candidate_count: int,
) -> dict:
    ind_found, sorted_indices = rank_from_distances(dist_full, target_mask)
    sorted_dist = dist_full[sorted_indices]
    true_dist = float(np.min(dist_full[target_mask]))
    true_mass = common.mass_from_smi(true_smiles)

    num_peaks_avg = np.mean([_num_sparse_peaks(sp) for sp in true_spec.values()])
    num_collision_engs = len(true_spec)

    out = {
        "ind_recovered": float(ind_found),
        "total_decoys": int(candidate_count),
        "decoys_in_full": int(full_candidate_count),
        "mass": float(true_mass),
        "mass_bin": common.bin_mass_results(true_mass),
        "num_peaks_avg": float(num_peaks_avg),
        "num_collision_engs": int(num_collision_engs),
        "peak_bin_avg": common.bin_peak_results(true_spec, reduction="mean"),
        "peak_bin_max": common.bin_peak_results(true_spec, reduction="max"),
        "peak_bin_min": common.bin_peak_results(true_spec, reduction="min"),
        "true_dist": true_dist,
        "spec_name": str(spec_name),
    }
    for k in range(0, min(50, len(sorted_dist))):
        out[f"top_{k + 1}_dist"] = float(sorted_dist[k])
    return out


def expand_scores_to_full_candidates(
    cand_ikeys: Sequence[str],
    dist: np.ndarray,
    full_candidate_ikeys: Sequence[str],
) -> np.ndarray:
    exact_map = {}
    stereo_map = {}
    for cand_ikey, cand_dist in zip(cand_ikeys, dist):
        exact_key = normalize_inchikey(cand_ikey)
        stereo_key = stereo_group_key(exact_key)
        exact_map[exact_key] = float(cand_dist)
        stereo_map[stereo_key] = float(cand_dist)

    expanded = np.empty(len(full_candidate_ikeys), dtype=np.float32)
    expanded.fill(np.nan)
    for idx, full_ikey in enumerate(full_candidate_ikeys):
        exact_key = normalize_inchikey(full_ikey)
        stereo_key = stereo_group_key(exact_key)
        if exact_key in exact_map:
            expanded[idx] = exact_map[exact_key]
        elif stereo_key in stereo_map:
            expanded[idx] = stereo_map[stereo_key]
    return expanded


def make_torchmetric_scores(dist_full: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    return -dist_full.astype(np.float64, copy=False)


def load_prediction_entries(
    pred_file: Path,
    pred_names: Sequence[str],
    num_bins: int,
    upper_limit: int,
    pool_fn: str,
    num_cpu_workers: int,
):
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

    read_name_fn = partial(
        _read_pred_spec_name,
        pred_file=str(pred_file),
        num_bins=num_bins,
        upper_limit=upper_limit,
        pool_fn=pool_fn,
    )
    common.chunked_parallel(
        pred_names,
        read_name_fn,
        chunks=1000,
        max_cpu=num_cpu_workers,
        task_name="Loading predicted spectra",
        output_func=collect_pred_outputs,
    )
    return np.array(pred_ikeys), pred_spec_ars, pred_name_to_indices


def build_grouped_outputs(df: pd.DataFrame, output_map: Dict[str, Path]):
    tsv_df = df.loc[:, ~df.columns.str.fullmatch(r"top_\d+_dist")]
    for group_key, out_name in output_map.items():
        if group_key not in tsv_df.columns:
            continue
        grouped = pd.concat(
            [
                tsv_df.groupby(group_key).mean(numeric_only=True),
                tsv_df.groupby(group_key).size().rename("num_examples"),
            ],
            axis=1,
        )
        all_mean = tsv_df.mean(numeric_only=True)
        all_mean["num_examples"] = len(tsv_df)
        all_mean.name = "avg"
        grouped = pd.concat([grouped, all_mean.to_frame().T], axis=0)
        grouped.to_csv(out_name, sep="\t")


def main(args):
    dataset = args.dataset
    data_folder = Path(f"data/spec_datasets/{dataset}")
    form_folder = data_folder / f"subformulae/{args.formula_dir_name}/"

    true_labels_path = args.true_labels
    if true_labels_path is None:
        true_labels_path = default_true_labels_path(data_folder)

    full_cands_path = args.full_labels
    if full_cands_path is None:
        raise ValueError(
            "Missing required --full-labels. Provide the retrieval candidate table to evaluate against."
        )

    query_df = pd.read_csv(true_labels_path, sep="\t")
    full_df = pd.read_csv(full_cands_path, sep="\t")
    full_group_sizes = full_df.groupby("spec").size()
    if len(full_group_sizes) > 0 and int(full_group_sizes.max()) <= 1:
        raise ValueError(
            "The full candidate label file provides at most one candidate per spectrum. "
            "Use a retrieval candidate table such as data/spec_datasets/<dataset>/retrieval/cands_df_<split>_*.tsv."
        )

    class_labels_path = data_folder / "chemical_class_labels.tsv"
    class_df = pd.read_csv(class_labels_path, sep="\t") if class_labels_path.exists() else None

    query_df = query_df.copy()
    normalized_query_structures = query_df.apply(
        lambda row: normalize_query_structure(row["smiles"], row["inchikey"]),
        axis=1,
    )
    query_df["smiles"] = [norm_smi for norm_smi, _ in normalized_query_structures]
    query_df["inchikey"] = [norm_ikey for _, norm_ikey in normalized_query_structures]

    name_to_ikey = dict(query_df[["spec", "inchikey"]].values)
    name_to_smi = dict(query_df[["spec", "smiles"]].values)
    name_to_ion = dict(query_df[["spec", "ionization"]].values)
    name_to_colli = dict(query_df[["spec", "collision_energies"]].values)
    name_to_class = (
        dict(class_df[["spec", "class"]].values)
        if class_df is not None and {"spec", "class"}.issubset(class_df.columns)
        else {}
    )

    parsed_colli_cache = {}
    for spec_name, colli_raw in name_to_colli.items():
        if colli_raw not in parsed_colli_cache:
            colli_engs = ast.literal_eval(colli_raw)
            parsed_colli_cache[colli_raw] = [
                common.get_collision_energy(colli_key)
                for colli_key in colli_engs
                if "nan" not in str(colli_key)
            ]
        name_to_colli[spec_name] = parsed_colli_cache[colli_raw]

    pred_file = Path(args.pred_file)
    pred_specs = common.PredSpecDB(h5_path=pred_file, mode="r")
    pred_names = list(pred_specs.get_all_names())

    pred_ikeys, pred_spec_ars, pred_name_to_indices = load_prediction_entries(
        pred_file=pred_file,
        pred_names=pred_names,
        num_bins=args.num_bins,
        upper_limit=args.upper_limit,
        pool_fn=args.pool_fn,
        num_cpu_workers=args.num_cpu_workers,
    )
    query_spec_names = set(query_df["spec"].astype(str).unique())
    pred_spec_names_unique = sorted(
        spec_name for spec_name in pred_name_to_indices.keys() if spec_name in query_spec_names
    )

    read_spec = partial(
        process_spec_file,
        name_to_colli=name_to_colli,
        spec_dir=form_folder,
        num_bins=args.num_bins,
        upper_limit=args.upper_limit,
        binned_spec=True,
        dataset=dataset,
    )
    true_specs = common.chunked_parallel(
        pred_spec_names_unique,
        read_spec,
        chunks=100,
        max_cpu=args.num_cpu_workers,
        task_name="Loading true spectra",
    )
    true_spec_by_name = {
        spec_name: true_spec
        for spec_name, true_spec in zip(pred_spec_names_unique, true_specs)
        if spec_name in name_to_ikey and true_spec is not None
    }

    full_candidates_by_spec = {
        spec_name: grp["inchikey"].astype(str).tolist()
        for spec_name, grp in full_df.groupby("spec", sort=False)
    }

    k_vals = list(range(1, 11)) + [20]
    hit_rates = {}
    all_preds_torch = []
    all_targets_torch = []
    all_indexes_torch = []
    output_entries = []
    running_lists = defaultdict(list)
    missing_true_candidate = 0
    missing_candidate_mapping = 0
    dropped_empty_true_spectra = []
    skipped_empty_true_specs = []

    for query_idx, spec_name in enumerate(sorted(true_spec_by_name)):
        selected_indices = pred_name_to_indices.get(spec_name)
        if not selected_indices:
            continue

        full_candidate_ikeys = full_candidates_by_spec.get(spec_name)
        if not full_candidate_ikeys:
            continue

        cand_ikeys = pred_ikeys[selected_indices]
        cand_preds = np.asarray([pred_spec_ars[i] for i in selected_indices], dtype=object)
        raw_true_spec = true_spec_by_name[spec_name]
        true_spec = {
            colli_eng: spec_value
            for colli_eng, spec_value in raw_true_spec.items()
            if _has_sparse_peaks(spec_value)
        }
        empty_colli_engs = [
            str(colli_eng)
            for colli_eng, spec_value in raw_true_spec.items()
            if colli_eng not in true_spec
        ]
        if empty_colli_engs:
            dropped_empty_true_spectra.append(
                {"spec_name": str(spec_name), "collision_energies": empty_colli_engs}
            )
        if len(true_spec) == 0:
            skipped_empty_true_specs.append(str(spec_name))
            continue
        true_smi = name_to_smi[spec_name]
        true_ion = name_to_ion[spec_name]
        true_ikey = normalize_inchikey(name_to_ikey[spec_name])
        true_stereo_group = stereo_group_key(true_ikey)

        parent_mass = common.mass_from_smi(true_smi) + common.ion2mass[true_ion]
        parent_mass_idx = (
            (parent_mass - 1) * args.num_bins / args.upper_limit
            if args.ignore_parent_peak
            else None
        )

        if args.dist_fn == "cos":
            dist = cos_dist_bin(
                cand_preds_dict=cand_preds,
                true_spec_dict=true_spec,
                sparse=False,
                ignore_peak=parent_mass_idx,
                sqrt_inten=args.sqrt_inten,
            )
        elif args.dist_fn == "entropy":
            dist = entropy_dist_bin(
                cand_preds_dict=cand_preds,
                true_spec_dict=true_spec,
                sparse=False,
                ignore_peak=parent_mass_idx,
                sqrt_inten=args.sqrt_inten,
            )
        elif args.dist_fn == "random":
            dist = np.random.randn(len(cand_preds))
        else:
            raise NotImplementedError(f"dist_fn={args.dist_fn} is not implemented.")

        target_mask = np.array(
            [stereo_group_key(ikey) == true_stereo_group for ikey in full_candidate_ikeys],
            dtype=bool,
        )
        if not np.any(target_mask):
            missing_true_candidate += 1
            continue

        dist_full = expand_scores_to_full_candidates(cand_ikeys, dist, full_candidate_ikeys)
        if np.isnan(dist_full[target_mask]).all():
            missing_candidate_mapping += 1
            continue
        dist_full = np.nan_to_num(dist_full, nan=1.0)

        entry = build_output_entry(
            spec_name=spec_name,
            true_spec=true_spec,
            true_smiles=true_smi,
            dist_full=dist_full,
            target_mask=target_mask,
            candidate_count=len(cand_ikeys),
            full_candidate_count=len(full_candidate_ikeys),
        )
        entry["ion"] = true_ion
        if spec_name in name_to_class:
            entry["class"] = name_to_class[spec_name]
        if "instrument" in query_df.columns:
            instrument = query_df.loc[query_df["spec"] == spec_name, "instrument"].iloc[0]
            entry["instrument"] = instrument
        output_entries.append(entry)

        for k in k_vals:
            below_k = entry["ind_recovered"] <= k
            running_lists[f"top_{k}"].append(below_k)
            entry[f"top_{k}"] = below_k
        running_lists["total_decoys"].append(entry["total_decoys"])
        running_lists["decoys_in_full"].append(entry["decoys_in_full"])
        running_lists["true_dist"].append(entry["true_dist"])

        scores_torch = make_torchmetric_scores(dist_full, target_mask)
        all_preds_torch.append(torch.from_numpy(scores_torch))
        all_targets_torch.append(torch.from_numpy(target_mask))
        all_indexes_torch.append(
            torch.full((len(full_candidate_ikeys),), query_idx, dtype=torch.long)
        )

    if all_preds_torch:
        preds_tensor = torch.cat(all_preds_torch)
        target_tensor = torch.cat(all_targets_torch)
        indexes_tensor = torch.cat(all_indexes_torch)
        for k in k_vals:
            metric = RetrievalHitRate(top_k=k)
            metric.update(preds_tensor, target_tensor, indexes=indexes_tensor)
            hit_rates[f"hit_rate_at_{k}"] = float(metric.compute().item())

    inten_suffix = "_no_sqrt" if not args.sqrt_inten else ""
    if args.outfile is None:
        outfile = pred_file.parent / f"rerank_eval_{args.dist_fn}.yaml"
        grouped_output_map = {
            "ion": pred_file.parent / f"rerank_eval_grouped_ion_{args.dist_fn}{inten_suffix}.tsv",
            "mass_bin": pred_file.parent / f"rerank_eval_grouped_mass_{args.dist_fn}{inten_suffix}.tsv",
            "peak_bin_avg": pred_file.parent / f"rerank_eval_grouped_npeak_{args.dist_fn}{inten_suffix}.tsv",
            "class": pred_file.parent / f"rerank_eval_grouped_class_{args.dist_fn}{inten_suffix}.tsv",
        }
    else:
        outfile = Path(args.outfile)
        grouped_output_map = {
            "ion": outfile.parent / f"{outfile.stem}_grouped_ion{inten_suffix}.tsv",
            "mass_bin": outfile.parent / f"{outfile.stem}_grouped_mass{inten_suffix}.tsv",
            "peak_bin_avg": outfile.parent / f"{outfile.stem}_grouped_npeak{inten_suffix}.tsv",
            "class": outfile.parent / f"{outfile.stem}_grouped_class{inten_suffix}.tsv",
        }
    if "instrument" in query_df.columns:
        grouped_output_map["instrument"] = outfile.parent / f"{outfile.stem}_grouped_instrument{inten_suffix}.tsv"

    final_output = {
        "dataset": dataset,
        "data_folder": str(data_folder),
        "dist_fn": args.dist_fn,
        "sqrt_inten": bool(args.sqrt_inten),
        "true_labels": str(true_labels_path),
        "full_labels": str(full_cands_path),
        "metrics": hit_rates,
        "missing_true_candidate": int(missing_true_candidate),
        "missing_candidate_mapping": int(missing_candidate_mapping),
        "dropped_empty_true_spectra": dropped_empty_true_spectra,
        "skipped_empty_true_specs": skipped_empty_true_specs,
        "individuals": sorted(output_entries, key=lambda x: x["ind_recovered"]),
    }
    for k, v in running_lists.items():
        final_output[f"avg_{k}"] = float(np.mean(v)) if len(v) > 0 else float("nan")
    for k, v in hit_rates.items():
        suffix = k.replace("hit_rate_at_", "top_")
        final_output[f"avg_{suffix}"] = v

    if output_entries:
        df = pd.DataFrame(output_entries)
        build_grouped_outputs(
            df=df,
            output_map=grouped_output_map,
        )

    with open(outfile, "w") as fp:
        yaml.dump(final_output, fp, indent=2)


if __name__ == "__main__":
    args = get_args()
    main(args)