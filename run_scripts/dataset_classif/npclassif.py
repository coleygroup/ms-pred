import json
import argparse
from pathlib import Path
import pickle
import time
import requests
from tqdm import tqdm

import pandas as pd
import numpy as np

from ms_pred import common


output_folder = "results/dataset_analyses/"
labels_files = [
    "data/spec_datasets/nist20/labels.tsv",
]
save_str = "ikey_to_classes.p"
debug = False

# Get large cache with everything
big_dict = {}
for file in labels_files:
    file = Path(file)
    dataset_name = file.parent.name
    save_name = Path(output_folder) / dataset_name / save_str
    save_name.parent.mkdir(exist_ok=True, parents=True)
    if save_name.exists():
        with open(save_name, "rb") as fp:
            full_out = pickle.load(fp)
            big_dict.update(full_out)


for file in labels_files:
    file = Path(file)
    dataset_name = file.parent.name
    save_name = Path(output_folder) / dataset_name / save_str
    save_name.parent.mkdir(exist_ok=True, parents=True)

    # Load progress cache already
    full_out = {}
    if save_name.exists():
        with open(save_name, "rb") as fp:
            full_out = pickle.load(fp)
            assert isinstance(full_out, dict)

    # Get labels file
    labels = pd.read_csv(file, sep="\t")
    rows = labels[["spec", "smiles"]].values
    all_smiles = rows[:, 1]
    all_ikeys = common.chunked_parallel(all_smiles, common.inchikey_from_smiles, 20)
    all_ikeys = np.array(all_ikeys)
    if debug:
        all_ikeys = all_ikeys[:100]
        all_smiles = all_smiles[:100]

    smi_to_ikey = dict(zip(all_smiles, all_ikeys))

    all_ikeys = all_ikeys
    all_smiles = all_smiles

    # Update cache with big dict
    full_out_update = {i: big_dict[i] for i in all_ikeys if i in big_dict}
    full_out.update(full_out_update)

    row_mask = [i not in full_out for i in all_ikeys]
    all_smiles = all_smiles[row_mask]
    all_ikeys = all_ikeys[row_mask]

    all_smiles = np.array(list(set(all_smiles)))
    all_ikeys = np.array([smi_to_ikey[i] for i in all_smiles])

    # Go in batches of 10
    all_batches = list(common.batches(all_smiles, 10))
    save_num = 500
    print(f"Number of smiles to run: {len(all_smiles)}")
    for input_ex in tqdm(all_batches):
        all_datas = common.simple_parallel(
            input_ex, common.npclassifier_query, max_cpu=20
        )
        temp_out = {}
        for i in all_datas:
            temp_out.update(i)

        # Add to running output
        full_out.update(temp_out)
        if len(full_out) % save_num == 0:
            # Export
            with open(save_name, "wb") as fp:
                print(f"Len of full out: {len(full_out)}")
                pickle.dump(full_out, fp)
    # Update big dict
    big_dict.update(full_out)

    with open(save_name, "wb") as fp:
        print(f"Len of full out: {len(full_out)}")
        pickle.dump(full_out, fp)
