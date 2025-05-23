""" Make retrieval lists

Convert the dataset into a list of 10k different retrieval entries.

"""
from pathlib import Path
from functools import partial
import pandas as pd
import numpy as np

import pickle
from tqdm import tqdm

import ms_pred.common as common


def process_example(obj, max_k=50) -> dict:
    """process_example.

    Convert obj dict to output


    Args:
        obj:
        max_k:

    Returns:
        dict:
    """
    if max_k is None:
        max_k = int(1e10)
    # Note this takes a while and should be done upfront
    # Some didn't pass round trip in frag engine, so repeat it here
    cands_list = [
        i for i in obj["cands"] if common.smi_inchi_round_mol(i[0]) is not None
    ]
    cands_list = np.array(cands_list, dtype=object)
    cands = [common.get_morgan_fp_smi(j[0]) for j in cands_list]
    true_cand = np.array([obj["smiles"], obj["inchikey"]])[None, :]
    true_tani = np.array([1.0])
    if len(cands) == 0:
        obj["cands"] = true_cand
        obj["tani_sims"] = true_tani
    else:
        cands = np.vstack(cands)
        orig_fp = common.get_morgan_fp_smi(obj["smiles"])

        intersect = np.einsum("ij, j -> i", cands, orig_fp)
        union = cands.sum(-1) + orig_fp.sum(-1) - intersect
        tani_sim = intersect / (union + 1e-22)

        # sort high to low
        sort_order = np.argsort(tani_sim)[::-1]
        sorted_tanis = tani_sim[sort_order][: max_k - 1]
        sorted_cands = cands_list[sort_order][: max_k - 1]

        obj["cands"] = np.concatenate([true_cand, sorted_cands])
        obj["tani_sims"] = np.concatenate([true_tani, sorted_tanis])

    return obj


def main(max_k, workers, input_map, input_dataset_folder, split_file):
    """main.

    Args:
        max_k:
        workers:
        input_map:
        input_dataset_folder:
        split_file:
    """

    retrieval_folder = input_dataset_folder / "retrieval"
    retrieval_folder.mkdir(exist_ok=True)

    output_pickle = retrieval_folder / "cands_pickled.p"
    output_df = retrieval_folder / "cands_df.tsv"

    input_smiles_labels = input_dataset_folder / "labels.tsv"
    df = pd.read_csv(input_smiles_labels, sep="\t")

    split_stem = ""
    if split_file is not None:
        split_stem = f"_{Path(split_file).stem}"
        split_file = input_dataset_folder / "splits" / split_file
        split_df = pd.read_csv(split_file, sep="\t")
        name_col, split_col = split_df.columns
        split_vals = split_df[split_col].values
        test_names = split_df[name_col].values[split_vals == "test"]
        df_mask = df["spec"].isin(test_names)
        df = df[df_mask].reset_index()

    output_pickle = retrieval_folder / f"cands_pickled{split_stem}_{max_k}.p"
    output_df = retrieval_folder / f"cands_df{split_stem}_{max_k}.tsv"

    # spec, ikey, smiles, formula
    headers = ["spec", "inchikey", "smiles", "formula", "ionization", "precursor", "collision_energies"]

    # DEBUG
    # df = df[df['spec'] == 'CCMSLIB00003111357']

    obj_list_dict = [dict(zip(headers, i)) for i in df[headers].values]

    form_map = pickle.load(open(input_map, "rb"))

    out_list = []
    for ct, obj in tqdm(enumerate(obj_list_dict)):
        all_isomers = form_map.get(obj["formula"], [])
        if len(all_isomers) == 0:
            continue

        # Filter down to ensure only unique isomers
        all_isomers_ars = np.array([(i, j) for i, j in all_isomers])
        _, uniq_inds = np.unique(all_isomers_ars[:, 1], return_index=True)

        # All isomers barring the true (smi, inchikey)
        all_isomers = {
            (i, j) for i, j in all_isomers_ars[uniq_inds] if j != obj["inchikey"]
        }
        # all_isomers.add((obj['smiles'], obj['inchikey']))

        obj["cands"] = list(all_isomers)

        if debug and ct > 5:
            break

        # Debug
        out_list.append(obj)

    process_fn = partial(process_example, max_k=max_k)

    # print(f"Num in database with incihikey found: {np.mean(num_found)}")
    if debug or len(out_list) == 0:
        processed = [process_fn(i) for i in out_list]
        processed = {i["spec"]: i for i in processed}
    else:
        processed = common.chunked_parallel(out_list, process_fn, max_cpu=workers)
        processed = {i["spec"]: i for i in processed}

        # Debugging
        # name = "CCMSLIB00003111357"
        # temp_ikey = "DQKMNCLZNGAXNX-UHFFFAOYSA-N"
        # process_input = [i for i in out_list
        #                 if i['spec'] == name]
        # used_smi = [(i,j ) for i,j in process_input[0]['cands'] if j == temp_ikey][0][0]
        # true_smi = process_input[0]['smiles']
        # used_fp, true_fp = common.get_morgan_fp_smi(used_smi), common.get_morgan_fp_smi(true_smi)

    print("Dumping to pickle")
    # Export 1: Convert to pickle file
    with open(output_pickle, "wb") as fp:
        pickle.dump(processed, fp)

    print("Dumping to df")
    # Export 2: Convert to data frame
    entries = []
    for spec, entry in tqdm(processed.items()):
        true_ikey = entry["inchikey"]
        formula = entry["formula"]
        true_smiles = entry["smiles"]
        true_ion = entry["ionization"]
        true_precursor = entry["precursor"]
        colli_engs = entry["collision_energies"]
        for (cand_smi, cand_ikey), cand_tani in zip(entry["cands"], entry["tani_sims"]):
            new_entry = {
                "spec": spec,
                # "true_ikey": true_ikey,
                # "true_smiles": true_smiles,
                # "formula": formula,
                "smiles": cand_smi,
                "ionization": true_ion,
                "inchikey": cand_ikey,
                "precursor": true_precursor,
                "collision_energies": colli_engs,
            }
            entries.append(new_entry)
    df_out = pd.DataFrame(entries)
    df_out.to_csv(
        output_df,
        sep="\t",
        index=None,
    )


if __name__ == "__main__":
    from rdkit import rdBase
    from rdkit import RDLogger

    rdBase.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.*")
    workers = 32
    debug = False

    # Modify here to create various datasets
    compute_entries = [
        {"dataset": "nist20", "max_k": 50, "split": "split_1.tsv"},
        {"dataset": "nist20", "max_k": 50, "split": "scaffold_1.tsv"},
        # {"dataset": "canopus_train_public", "max_k": 50, "split": "split_1.tsv"},
    ]

    for test_entry in compute_entries:
        dataset = test_entry['dataset']
        max_k= test_entry['max_k']
        split_file = test_entry['split']

        input_map = f"data/retrieval/pubchem/pubchem_formula_map_{dataset}.p"
        input_dataset_folder = Path(f"data/spec_datasets/{dataset}/")
        print(f"Starting split {split_file}")
        main(
            max_k=max_k,
            workers=workers,
            input_map=input_map,
            input_dataset_folder=input_dataset_folder,
            split_file=split_file,
        )
