""" Convert dag folder into a single mgf file """
import json
import yaml
import argparse
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import ms_pred.common as common
import ms_pred.magma.fragmentation as fragmentation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--dag-folder", action="store")
    parser.add_argument("--out-mgf", action="store")
    parser.add_argument("--max-peaks", action="store", type=int, default=100)
    return parser.parse_args()


def dag_to_spec(dag: dict, max_peaks: int, min_inten: float, precision=4):
    """bin_dag.

    Args:
        dag (dict): dag
        max_peaks (int): max_peaks
        min_inten (float): min_inten
        precision (float): precision
    """
    root = dag["root_inchi"]
    frags = dag["frags"]
    engine = fragmentation.FragmentEngine(mol_str=root, mol_str_type="inchi")
    adduct = dag["adduct"]
    parentmass = engine.full_weight + common.ion2mass[adduct]

    mz, intens = [], []
    for k, v in frags.items():
        new_masses = v["mz_charge"]
        new_intens = v["intens"]
        mz.extend(new_masses)
        intens.extend(new_intens)

    # Cap at min
    mz, intens = np.array(mz), np.array(intens)
    min_inten_mask = intens > min_inten
    mz = mz[min_inten_mask]
    intens = intens[min_inten_mask]

    fused_tuples = [[x, y] for x, y in zip(mz, intens)]

    # Merge by max function
    mz_to_inten_pair = {}
    new_tuples = []
    for tup in fused_tuples:
        mz, inten = tup
        mz_ind = np.round(mz, precision)
        cur_pair = mz_to_inten_pair.get(mz_ind)
        if cur_pair is None:
            mz_to_inten_pair[mz_ind] = tup
            new_tuples.append(tup)
        elif inten > cur_pair[1]:
            cur_pair[1] = inten
        else:
            pass

    merged_spec = np.vstack(new_tuples)
    mz, intens = merged_spec[:, 0], merged_spec[:, 1]

    # Sort and thresh
    inds = np.argsort(intens)[::-1][:max_peaks]
    mz = mz[inds]
    intens = intens[inds]

    meta = dict(
        inchi=dag["root_inchi"],
        parentmass=parentmass,
        smiles=common.smiles_from_inchi(dag["root_inchi"]),
        ID=dag["name"],
        ionization=adduct,
        adduct=adduct,
    )
    spec_ar = np.vstack([mz, intens]).transpose(1, 0)
    return meta, spec_ar


def file_to_spec(dag_file: dict, max_peaks: int, min_inten: float):
    """file_to_spec.

    Args:
        dag_file (dict): dag_file
        max_peaks (int): max_peaks
        min_inten (float): min_inten

    Returns:
    """
    dag = json.load(open(dag_file, "r"))
    return dag_to_spec(dag, max_peaks=max_peaks, min_inten=min_inten)


def main():
    """main."""
    args = get_args()
    out = args.out_mgf
    num_workers = args.num_workers
    max_peaks = 100
    min_inten = 0
    dag_folder = Path(args.dag_folder)
    dag_files = list(dag_folder.glob("*.json"))
    spec_names = [i.stem.replace("pred_", "") for i in dag_files]

    # Test case
    dag_file = dag_files[0]
    out_obj = file_to_spec(dag_file, max_peaks=max_peaks, min_inten=min_inten)
    read_dag_file = partial(file_to_spec, max_peaks=max_peaks, min_inten=min_inten)

    if num_workers > 0:
        outs = common.chunked_parallel(
            dag_files,
            read_dag_file,
            max_cpu=num_workers,
        )
    else:
        outs = [read_dag_file(i) for i in dag_files]

    mgf_str = common.build_mgf_str(outs)
    with open(out, "w") as fp:
        fp.write(mgf_str)


if __name__ == "__main__":
    main()
