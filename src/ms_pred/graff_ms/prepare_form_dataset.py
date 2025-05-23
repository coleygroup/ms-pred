import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import copy
import json
from tqdm import tqdm

import ms_pred.common as common
from ms_pred.graff_ms import graff_ms_data


def parse_dataset_spec(
        df, num_bins, form_map, form_h5, num_workers, upper_limit,
):
    valid_spec_ids = set([common.rm_collision_str(i) for i in form_map])
    valid_specs = [i in valid_spec_ids for i in df["spec"].values]
    df_sub = df[valid_specs]

    if len(df_sub) == 0:
        spec_names = []
        smiles = []
        name_to_dict = {}
    else:
        ori_label_map = df_sub.set_index("spec").drop("collision_energies", axis=1).to_dict(
            orient="index")  # labels without collision energy
        spec_names = []
        smiles = []
        name_to_dict = {}
        for i in form_map:
            ori_id = common.rm_collision_str(i)
            if ori_id in ori_label_map:
                label_dict = ori_label_map[ori_id]
                spec_names.append(i)
                smiles.append(label_dict["smiles"])
                name_to_dict[i] = copy.deepcopy(label_dict)

    for i in name_to_dict:
        name_to_dict[i]["formula_file"] = form_map[i]

    print(f"{len(df_sub)} of {len(df)} spec have {len(spec_names)} form dicts.")
    form_files = [
        name_to_dict[i]["formula_file"] for i in spec_names
    ]

    # Read in all specs
    spec_files = form_files

    def read_spec(name):
        if not name.endswith('.json'):
            name += '.json'
        h5_obj = common.HDF5Dataset(form_h5)
        spec_form_obj = graff_ms_data.process_form_str(h5_obj.read_str(name), upper_limit, num_bins)
        return json.dumps(spec_form_obj)

    if num_workers > 1:
        spec_outputs = common.chunked_parallel(spec_files, read_spec, chunks=1000, max_cpu=num_workers)
    else:
        spec_outputs = [read_spec(i) for i in tqdm(spec_files, desc='reading spec')]
    name_to_forms = dict(zip(spec_names, spec_outputs))
    return name_to_forms


def add_graff_ms_train_args(parser):
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")
    parser.add_argument(
        "--form-dir-name", default="magma_subform_50_with_raw.hdf5", action="store"
    )
    parser.add_argument("--num-bins", default=15000, action="store", type=int)
    parser.add_argument("--num-fixed-forms", default=10000, action="store", type=int)
    parser.add_argument("--out-path", default="")
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_graff_ms_train_args(parser)
    return parser.parse_args()

def main():
    args = get_args()
    kwargs = args.__dict__
    upper_limit = 1500


    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = Path("data/spec_datasets") / dataset_name
    labels = data_dir / kwargs["dataset_labels"]
    split_file = data_dir / "splits" / kwargs["split_name"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")

    spec_names = df["spec"].values

    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]

    subform_stem = kwargs["form_dir_name"]
    subformula_h5_path = Path(data_dir) / "subformulae" / subform_stem
    subformula_h5 = common.HDF5Dataset(subformula_h5_path)
    form_map = {Path(i).stem: i for i in subformula_h5.get_all_names()}

    num_bins = kwargs.get("num_bins")
    num_workers = kwargs.get("num_workers", 0)

    for _df, split in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        spec_dic = parse_dataset_spec(_df, num_bins, form_map, subformula_h5_path, num_workers, upper_limit)

        # Dump to hdf5
        outp_path = Path(kwargs["out_path"]) / f'graffms_spec_{split_file.stem}_{split}.hdf5'
        outp_path.parent.mkdir(exist_ok=True)
        out_h5_obj = common.HDF5Dataset(outp_path, 'w')
        out_h5_obj.write_dict(spec_dic)
        out_h5_obj.close()


if __name__ == '__main__':
    main()
