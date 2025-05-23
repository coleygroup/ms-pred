import numpy as np
from pathlib import Path
import pickle
import argparse
import pandas as pd
from functools import partial
from tqdm import tqdm

from ms_pred.retrieval.retrieval_benchmark import cos_dist
from ms_pred import common


def get_args():
    """get_args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec-folder",
        default="data/elucidation/broad_distress/spec_files"
    )
    parser.add_argument(
        "--binned-pred-file",
        default="results/dag_inten_nist20/split_1_rnd1/elucidation_broad_distress/binned_preds.p",
    )
    parser.add_argument("--outfile", default=None)
    parser.add_argument("--dist-fn", default="cos")
    return parser.parse_args()


def process_spec_file(spec_name: str, num_bins: int, upper_limit: int, spec_dir: Path):
    """process_spec_file."""
    if spec_dir.suffix == '.hdf5': # is h5file
        spec_h5 = common.HDF5Dataset(spec_dir)
        spec_file = spec_h5.read_str(spec_name).split('\n')
    else: # is directory
        spec_files = list(spec_dir.glob(f'{spec_name}.[mM][sS]'))
        if len(spec_files) == 0:
            raise FileNotFoundError(f'{spec_dir / spec_name}.ms does not exist')
        spec_file = spec_files[0]
    meta, all_spec = common.parse_spectra(spec_file)
    all_spec = common.process_spec_file(meta, all_spec, merge_specs=False)

    return_dict = {}
    for colli_eng, spec in all_spec.items():
        colli_eng = common.get_collision_energy(colli_eng)
        binned = common.bin_spectra([spec], num_bins, upper_limit)
        avged = binned[0]
        return_dict[colli_eng] = avged
    return return_dict


def rank_unknown_entry(
    cand_smiles,
    cand_preds,
    true_spec,
    spec_name,
    dist_fn="cos",
):
    if dist_fn == "cos":
        dist = cos_dist(cand_preds_dict=cand_preds, true_spec_dict=true_spec)
    elif dist_fn == "random":
        dist = np.random.randn(cand_preds.shape[0])
    else:
        raise NotImplementedError()

    resorted = np.argsort(dist)
    resorted_smiles = cand_smiles[resorted]
    resorted_dist = dist[resorted]

    return {
        "ranked_smiles": resorted_smiles,
        "ranked_dist": resorted_dist,
        "spec_name": str(spec_name),
    }


def main(args):
    """main."""
    dist_fn = args.dist_fn
    spec_folder = Path(args.spec_folder)
    binned_pred_file = Path(args.binned_pred_file)
    outfile = args.outfile
    if outfile is None:
        outfile = binned_pred_file.parent / 'ranked_smiles.tsv'
    else:
        outfile = Path(outfile)

    pred_specs = pickle.load(open(binned_pred_file, "rb"))
    pred_spec_ars = pred_specs["preds"]
    pred_smiles = np.array(pred_specs['smiles'])
    pred_spec_names = np.array(pred_specs["spec_names"])
    pred_spec_names_unique = np.unique(pred_spec_names)
    upper_limit = pred_specs["upper_limit"]
    num_bins = pred_specs["num_bins"]
    use_sparse = pred_specs["sparse_out"]

    # Only use sparse valid for now
    assert use_sparse

    read_spec = partial(
        process_spec_file,
        num_bins=num_bins,
        upper_limit=upper_limit,
        spec_dir=spec_folder,
    )
    true_specs = common.chunked_parallel(
        pred_spec_names_unique,
        read_spec,
        chunks=1000,
        max_cpu=16,
    )
    name_to_spec = dict(zip(pred_spec_names_unique, true_specs))

    # Create a list of dicts, bucket by mass, etc.
    all_entries = []
    for spec_name in tqdm(pred_spec_names_unique):

        # Get candidates
        bool_sel = pred_spec_names == spec_name
        cand_smiles = pred_smiles[bool_sel]
        cand_preds = np.array(pred_spec_ars)[bool_sel]
        true_spec = name_to_spec[spec_name]
        new_entry = {
            "cand_smiles": cand_smiles,
            "cand_preds": cand_preds,
            "true_spec": true_spec,
            "spec_name": spec_name,
        }

        if true_spec is None:
            continue
        all_entries.append(new_entry)

    rank_unknown_entry_ = partial(rank_unknown_entry, dist_fn=dist_fn)
    all_out = [rank_unknown_entry_(**test_entry) for test_entry in all_entries]

    # Output a sheet of structural elucidations
    out_entries = []
    for out_info in all_out:
        spec_name = out_info["spec_name"]
        for smi, dist in zip(out_info['ranked_smiles'], out_info['ranked_dist']):
            out_entries.append({
                "spec_name": spec_name,
                "predict_smiles": smi,
                "cosine_distance": dist,
            })
    df = pd.DataFrame(out_entries)
    df.to_csv(outfile, sep='\t')


if __name__ == "__main__":
    args = get_args()
    main(args)
