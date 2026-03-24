""" Add dag intensities

Given a set of predicted dags, add intensities to them from the gold standard

"""
import json
import argparse
import copy
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

import ms_pred.magma.fragmentation as fragmentation
import ms_pred.common as common


def get_args():
    """get_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--pred-dag-path", action="store")
    parser.add_argument("--true-dag-path", action="store")
    parser.add_argument("--out-dag-path", action="store")
    return parser.parse_args()


def relabel_tree(
    pred_dag_db: common.PredSpecDB,
    true_dag_h5: Path,
    pred_dag_name: str,
    true_dag_name: str,
    out_dag_name: str,
    collision_energy: str,
    max_bonds: int,
) -> Tuple[str, str]:
    """relabel_tree."""
    zero_vec = [0] * (2 * max_bonds + 1)
    true_dag_h5 = common.HDF5Dataset(true_dag_h5)

    if not true_dag_name in true_dag_h5:
        return None

    pred_dag = pred_dag_db.read(pred_dag_name, collision_energy)
    true_dag = json.loads(true_dag_h5.read_str(true_dag_name))

    assert pred_dag.root_canonical_smiles is not None
    assert pred_dag.frags is not None
    assert pred_dag.collision_energy is not None
    assert pred_dag.adduct is not None

    true_tbl = true_dag["output_tbl"]
    raw_spec = list(zip(true_tbl["mono_mass"], true_tbl["rel_inten"]))
    pred_dag.meta["raw_spec"] = raw_spec

    return out_dag_name, pred_dag


def main():
    """main."""
    args = get_args()
    pred_dag_path = Path(args.pred_dag_path)
    true_dag_path = Path(args.true_dag_path)
    out_dag_path = Path(args.out_dag_path)

    out_dag_path.parent.mkdir(exist_ok=True)

    max_bonds = fragmentation.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"]
    pred_dag_db = common.PredSpecDB(pred_dag_path)
    pred_dag_name_set = set(pred_dag_db.get_all_names())

    num_workers = args.num_workers
    pred_dag_names, true_dag_names, out_dag_names, colli_engs = [], [], [], []
    true_dag_h5 = common.HDF5Dataset(true_dag_path)
    for true_dag_n in tqdm(true_dag_h5.get_all_names()):
        spec_id = common.rm_collision_str(true_dag_n)
        colli_eng = common.get_collision_energy(true_dag_n)
        pred_dag_name = 'pred_' + spec_id
        if pred_dag_name not in pred_dag_name_set:
            continue
        pred_dag_names.append(pred_dag_name)
        true_dag_names.append(true_dag_n)
        out_dag_names.append(spec_id)
        colli_engs.append(colli_eng)
    true_dag_h5.close()
    arg_dicts = [
        {
            "pred_dag_db": pred_dag_db,
            "true_dag_h5": true_dag_path,
            "pred_dag_name": i,
            "true_dag_name": j,
            "out_dag_name": k,
            "collision_energy": l,
            "max_bonds": max_bonds,
        }
        for i, j, k, l in zip(pred_dag_names, true_dag_names, out_dag_names, colli_engs)
    ]

    def write_func(outs):
        out_db = common.PredSpecDB(out_dag_path, mode='w')
        for out in outs:
            out_db.write(*out)
        out_db.close()

    # Run
    wrapper_fn = lambda arg_dict: relabel_tree(**arg_dict)
    if num_workers == 0:
        outs = [wrapper_fn(i) for i in arg_dicts]
        write_func(outs)
    else:
        common.chunked_parallel(arg_dicts, wrapper_fn, output_func=write_func, max_cpu=num_workers, chunks=1000)
    print("success!")


if __name__ == "__main__":
    main()
