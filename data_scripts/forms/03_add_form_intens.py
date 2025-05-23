""" Add formulae intensities """
import json
import numpy as np
import argparse
import copy
from pathlib import Path
from tqdm import tqdm

import ms_pred.magma.run_magma as run_magma
import ms_pred.common as common


def get_args():
    """get_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--pred-form-folder", action="store")
    parser.add_argument("--true-form-folder", action="store")
    parser.add_argument("--out-form-folder", action="store", default=None)
    parser.add_argument("--binned-add", action="store_true", default=False)
    parser.add_argument("--add-raw", action="store_true", default=False)
    return parser.parse_args()


def relabel_tree(
    pred_form_folder: Path,
    true_form_folder: Path,
    pred_form_name: str,
    true_form_name: str,
    out_form_name: str,
    add_binned: bool,
    add_raw,
):
    """relabel_tree"""
    if not true_form_name:
        return

    pred_form_h5 = common.HDF5Dataset(pred_form_folder)
    true_form_h5 = common.HDF5Dataset(true_form_folder)

    pred_form = json.loads(pred_form_h5.read_str(pred_form_name))
    true_form = json.loads(true_form_h5.read_str(true_form_name))

    if true_form["output_tbl"] is None:
        return

    pred_tbl, true_tbl = pred_form.get("output_tbl", None), true_form["output_tbl"]
    if pred_tbl is None:
        pred_tbl = {"mono_mass": []}

    # Use rel inten
    true_form_to_inten = dict(zip(true_tbl["formula"], true_tbl["rel_inten"]))

    if add_binned:
        bins = np.linspace(0, 1500, 15000)
        true_pos = np.digitize(true_tbl["mono_mass"], bins)
        pred_pos = np.digitize(pred_tbl["mono_mass"], bins)
        bin_to_inten = dict()
        for i, j in zip(true_pos, true_tbl["rel_inten"]):
            bin_to_inten[i] = max(j, bin_to_inten.get(i, 0))
        new_intens = [bin_to_inten.get(i, 0) for i in pred_pos]
    else:
        new_intens = [true_form_to_inten.get(i, 0.0) for i in pred_tbl["formula"]]

    if add_raw:
        raw_spec = list(zip(true_tbl["mono_mass"], true_tbl["rel_inten"]))
        pred_tbl["raw_spec"] = raw_spec

    pred_tbl["rel_inten"] = new_intens
    pred_tbl["ms2_inten"] = new_intens

    return out_form_name, json.dumps(pred_form, indent=2)


def main():
    """main."""
    args = get_args()
    pred_form_folder = Path(args.pred_form_folder)
    true_form_folder = Path(args.true_form_folder)
    out_form_folder = args.out_form_folder
    add_binned = args.binned_add
    add_raw = args.add_raw

    if out_form_folder is None:
        out_form_folder = pred_form_folder
    out_form_folder = Path(out_form_folder)
    out_form_folder.parent.mkdir(exist_ok=True)

    num_workers = args.num_workers
    pred_form_h5 = common.HDF5Dataset(pred_form_folder)
    pred_form_names = list(pred_form_h5.get_all_names())
    true_form_h5 = common.HDF5Dataset(true_form_folder)
    true_form_names = [i.replace("pred_", "") for i in pred_form_names]
    true_form_names = [i if i in true_form_h5 else "" for i in true_form_names]
    out_form_names = pred_form_names
    pred_form_h5.close()
    true_form_h5.close()

    arg_dicts = [
        {
            "pred_form_folder": pred_form_folder,
            "true_form_folder": true_form_folder,
            "pred_form_name": i,
            "true_form_name": j,
            "out_form_name": k,
            "add_raw": add_raw,
            "add_binned": add_binned,
        }
        for i, j, k in zip(pred_form_names, true_form_names, out_form_names)
    ]

    # Run
    wrapper_fn = lambda arg_dict: relabel_tree(**arg_dict)

    def output_fn(out_dicts):
        h5 = common.HDF5Dataset(out_form_folder, mode='w')
        for tup in out_dicts:
            h5.write_str(tup[0], tup[1])
        h5.close()

    # Debug
    if num_workers == 0:
        out_dicts = [wrapper_fn(i) for i in tqdm(arg_dicts)]
        output_fn(out_dicts)
    else:
        common.chunked_parallel(arg_dicts, wrapper_fn, output_func=output_fn, chunks=1000, max_cpu=num_workers)


if __name__ == "__main__":
    main()
