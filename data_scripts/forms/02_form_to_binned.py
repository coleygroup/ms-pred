""" Form to binned

Convert dag folder into a binned spec file

"""
import json
import argparse
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import ms_pred.common as common


def get_args():
    """get_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--form-folder", action="store")
    parser.add_argument(
        "--min-inten",
        type=float,
        default=1e-5,
        help="Minimum intensity to call a peak in prediction",
    )
    parser.add_argument(
        "--max-peaks", type=int, default=100, help="Max num peaks to call"
    )
    parser.add_argument("--num-bins", type=int, default=1500, help="Num bins")
    parser.add_argument("--upper-limit", type=int, default=1500, help="upper lim")
    parser.add_argument("--out", action="store")
    return parser.parse_args()


def bin_forms(
    forms: dict, max_peaks: int, upper_limit: int, num_bins: int, min_inten: float
) -> np.ndarray:
    """bin_forms.

    Args:
        forms (dict): forms
        max_peaks (int): max_peaks
        upper_limit (int): upper_limit
        num_bins (int): num_bins
        min_inten (float): min_inten

    Returns:
        np.ndarray:
    """
    smiles = forms.get("smiles")

    tbl = forms.get("output_tbl")
    if tbl is None:
        tbl = {"rel_inten": [], "mono_mass": []}

    intens = tbl["rel_inten"]
    mz = tbl["mono_mass"]

    # Cap at min
    mz, intens = np.array(mz), np.array(intens)
    min_inten_mask = intens > min_inten
    mz = mz[min_inten_mask]
    intens = intens[min_inten_mask]

    # Keep 100 max
    argsort_intens = np.argsort(intens)[::-1][:max_peaks]
    mz = mz[argsort_intens]
    intens = intens[argsort_intens]
    spec_ar = np.vstack([mz, intens]).transpose(1, 0)

    # Bin intensities
    binned = common.bin_spectra([spec_ar], num_bins, upper_limit, pool_fn="max")[0]
    return binned, smiles


def bin_form_file(
    form_filename: dict, h5_path: str, max_peaks: int, upper_limit: int, num_bins: int, min_inten: float
):
    """bin_dag_file.

    Args:
        dag_file (dict): dag_file
        max_peaks (int): max_peaks
        upper_limit (int): upper_limit
        num_bins (int): num_bins
        min_inten (float): min_inten

    Returns:
        np.ndarray:
        str
    """
    h5 = common.HDF5Dataset(h5_path)
    forms = json.loads(h5.read_str(form_filename))
    return bin_forms(forms, max_peaks, upper_limit, num_bins, min_inten)


def main():
    """main."""
    args = get_args()
    out = args.out

    max_peaks, min_inten = args.max_peaks, args.min_inten
    num_bins, upper_limit = args.num_bins, args.upper_limit
    num_workers = args.num_workers
    form_folder = Path(args.form_folder)
    form_h5 = common.HDF5Dataset(form_folder)

    if out is None:
        out = form_folder.parent / f"{form_folder.stem}_binned.hdf5"

    form_files = list(form_h5.get_all_names())
    spec_names = [Path(i).stem.replace("pred_", "") for i in form_files]

    # Test case
    # dag_file = dag_files[0]
    # binned, root = bin_dag_file(dag_file, max_peaks=max_peaks,
    #                            upper_limit=upper_limit, num_bins=num_bins,
    #                            min_inten=min_inten)

    read_dag_file = partial(
        bin_form_file,
        h5_path=form_folder,
        max_peaks=max_peaks,
        upper_limit=upper_limit,
        num_bins=num_bins,
        min_inten=min_inten,
    )

    if num_workers > 0:
        outs = common.chunked_parallel(
            form_files,
            read_dag_file,
            max_cpu=num_workers,
        )
        binned, smis = zip(*outs)
    else:
        outs = [read_dag_file(i) for i in form_files]
        binned, smis = zip(*outs)

    binned_h5 = common.HDF5Dataset(out, mode='w')
    binned_h5.attrs['num_bins'] = num_bins
    binned_h5.attrs['upper_limit'] = upper_limit
    binned_h5.attrs['sparse_out'] = False
    for bin_spec, smi, spec_name_with_ce in zip(binned, smis, spec_names):
        spec_name = common.rm_collision_str(spec_name_with_ce)
        colli_eng = common.get_collision_energy(spec_name_with_ce)
        inchikey = common.inchikey_from_smiles(smi)
        h5_name = f'pred_{spec_name}/ikey {inchikey}/collision {colli_eng}'
        binned_h5.write_data(h5_name + '/spec', bin_spec)
        binned_h5.update_attr(h5_name, {'smiles': smi, 'ikey': inchikey, 'spec_name': spec_name})
    binned_h5.close()


if __name__ == "__main__":
    main()
