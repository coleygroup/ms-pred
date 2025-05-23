""" Add dag intensities

Given a set of predicted dags, add intensities to them from the gold standard

"""
import json
import argparse
import copy
from pathlib import Path
from typing import Tuple

from tqdm import tqdm
import re

import ms_pred.magma.run_magma as run_magma
import ms_pred.common as common


def get_args():
    """get_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--pred-dag-folder", action="store")
    parser.add_argument("--true-dag-folder", action="store")
    parser.add_argument("--out-dag-folder", action="store")
    parser.add_argument(
        "--add-raw",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def relabel_tree(
    pred_dag_h5: Path,
    true_dag_h5: Path,
    pred_dag_name: str,
    true_dag_name: str,
    out_dag_name: str,
    max_bonds: int,
    add_raw: bool,
) -> Tuple[str, str]:
    """relabel_tree."""
    zero_vec = [0] * (2 * max_bonds + 1)
    pred_dag_h5 = common.HDF5Dataset(pred_dag_h5)
    true_dag_h5 = common.HDF5Dataset(true_dag_h5)

    if not true_dag_name in true_dag_h5:
        return None

    pred_dag = json.loads(pred_dag_h5.read_str(pred_dag_name))
    true_dag = json.loads(true_dag_h5.read_str(true_dag_name))

    assert 'root_canonical_smiles' in pred_dag
    assert 'frags' in pred_dag
    assert 'collision_energy' in pred_dag
    assert 'adduct' in pred_dag

    if add_raw:
        true_tbl = true_dag["output_tbl"]
        raw_spec = list(zip(true_tbl["mono_mass"], true_tbl["rel_inten"]))
        pred_dag["raw_spec"] = raw_spec
    else:
        pred_frags, true_frags = pred_dag["frags"], true_dag["frags"]

        for k, pred_frag in pred_frags.items():
            if k in true_frags:
                true_frag = true_frags[k]
                pred_frag["intens"] = true_frag["intens"]
            else:
                pred_frag["intens"] = copy.deepcopy(zero_vec)

    return out_dag_name, json.dumps(pred_dag, indent=2)


def main():
    """main."""
    args = get_args()
    pred_dag_folder = Path(args.pred_dag_folder)
    true_dag_folder = Path(args.true_dag_folder)
    out_dag_folder = Path(args.out_dag_folder)
    add_raw = args.add_raw

    out_dag_folder.parent.mkdir(exist_ok=True)

    max_bonds = run_magma.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"]

    num_workers = args.num_workers
    pred_dag_names, true_dag_names, out_dag_names = [], [], []
    true_dag_h5 = common.HDF5Dataset(true_dag_folder)
    for true_dag_n in tqdm(true_dag_h5.get_all_names()):
        spec_id = 'pred_' + true_dag_n
        pred_dag_names.append(spec_id)
        true_dag_names.append(true_dag_n)
        out_dag_names.append(true_dag_n)
    true_dag_h5.close()

    arg_dicts = [
        {
            "pred_dag_h5": pred_dag_folder,
            "true_dag_h5": true_dag_folder,
            "pred_dag_name": i,
            "true_dag_name": j,
            "out_dag_name": k,
            "max_bonds": max_bonds,
            "add_raw": add_raw,
        }
        for i, j, k in zip(pred_dag_names, true_dag_names, out_dag_names)
    ]

    # Run
    wrapper_fn = lambda arg_dict: relabel_tree(**arg_dict)
    if num_workers == 0:
        outs = [wrapper_fn(i) for i in arg_dicts]
    else:
        outs = common.chunked_parallel(arg_dicts, wrapper_fn, max_cpu=num_workers, chunks=1000)

    # Write output to HDF5 file
    out_h5 = common.HDF5Dataset(out_dag_folder, mode='w')
    out_h5.write_list_of_tuples(outs)
    out_h5.close()


if __name__ == "__main__":
    main()
