"""predict.py

Make predictions with trained model

"""

import logging
import random
import multiprocess.process
from datetime import datetime
import yaml
import argparse
import json
import ast
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import torch
import pytorch_lightning as pl

import ms_pred.common as common
import ms_pred.dag_pred.gen_model as gen_model
from foam.opt_graph_ga_fc.mutate import mutate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    parser.add_argument("--num-decoys", default=0, action="store", type=int)
    parser.add_argument("--pubchem-map-path", default='/home/runzhong/foam/data/pubchem/pubchem_formulae_inchikey.hdf5')
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_tree_pred/")

    parser.add_argument(
        "--checkpoint-pth",
        help="name of checkpoint file",
        default="results/debug_dag_canopus_train_public/split_1/version_0/best.ckpt",
    )
    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_1.tsv")
    parser.add_argument(
        "--subset-datasets",
        default="none",
        action="store",
        choices=["none", "train_only", "test_only", "debug_special"],
    )
    parser.add_argument("--threshold", default=0.5, action="store", type=float)
    parser.add_argument("--max-nodes", default=100, action="store", type=int)
    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__

    save_dir = Path(kwargs["save_dir"])
    common.setup_logger(save_dir, log_name="dag_gen_pred.log", debug=kwargs["debug"])
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(save_dir / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = common.get_data_dir(dataset_name)
    labels = data_dir / kwargs["dataset_labels"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")

    if kwargs["subset_datasets"] != "none":
        splits = pd.read_csv(data_dir / "splits" / kwargs["split_name"], sep="\t")
        folds = set(splits.keys())
        folds.remove("spec")
        fold_name = list(folds)[0]
        if kwargs["subset_datasets"] == "train_only":
            names = splits[splits[fold_name] == "train"]["spec"].tolist()
        elif kwargs["subset_datasets"] == "test_only":
            names = splits[splits[fold_name] == "test"]["spec"].tolist()
        elif kwargs["subset_datasets"] == "debug_special":
            names = ["CCMSLIB00000577858"]
        else:
            raise NotImplementedError()

        df = df[df["spec"].isin(names)]

    # Create model and load
    best_checkpoint = kwargs["checkpoint_pth"]

    # Load from checkpoint
    gpu = kwargs["gpu"]
    model = gen_model.FragGNN.load_from_checkpoint(best_checkpoint)
    avail_gpu_num = torch.cuda.device_count()

    logging.info(f"Loaded model with from {best_checkpoint}")
    save_path = save_dir / ("tree_preds.hdf5" if kwargs["num_decoys"] == 0 else "decoy_tree_preds.hdf5")
    save_path.parent.mkdir(exist_ok=True)
    with torch.no_grad():
        model.eval()
        model.freeze()

        def prepare_entry(entry):
            torch.set_num_threads(1)

            smi = entry["smiles"]
            name = entry["spec"]
            adduct = entry["ionization"]
            precursor_mz = entry["precursor"]
            collision_energies = [i for i in ast.literal_eval(entry["collision_energies"])]
            mol = Chem.MolFromSmiles(smi)
            inchi = Chem.MolToInchi(mol)
            inchikey = Chem.MolToInchiKey(mol)

            tup_to_process = []
            num_decoys = kwargs["num_decoys"]
            if num_decoys > 0:  # decoys only
                # generate 50% mutation decoys + 50% pubchem isomer decoys
                pubchem_rate = 0.5
                formula = common.uncharged_formula(mol, mol_type='mol')
                h5obj = common.HDF5Dataset(kwargs['pubchem_map_path'])
                # output_h5 = common.HDF5Dataset(save_path)  # todo output h5
                if formula in h5obj:
                    num_pubchem = int(num_decoys * pubchem_rate)
                    num_mutation = num_decoys - num_pubchem
                else:
                    num_pubchem = 0
                    num_mutation = num_decoys
                decoy_mols = [mutate(mol, mutation_rate=1.) for _ in range(num_mutation)]
                if num_pubchem > 0:
                    cand_str = h5obj.read_str(formula)
                    smi_inchi_list = json.loads(cand_str)
                    decoy_mols += [Chem.MolFromSmiles(_[0]) for _ in random.choices(smi_inchi_list, k=num_decoys)]

                decoy_mols = np.array(common.sanitize(decoy_mols))
                decoy_inchikeys = np.array([Chem.MolToInchiKey(m) for m in decoy_mols])
                _, unique_inds = np.unique(decoy_inchikeys, return_index=True)

                decoy_smis = []
                for new_mol, new_inchikey in zip(decoy_mols[unique_inds], decoy_inchikeys[unique_inds]):
                    new_smi = Chem.MolToSmiles(new_mol)
                    if new_inchikey != inchikey:
                        decoy_smis.append(new_smi)
                for i, decoy_smi in enumerate(decoy_smis):
                    for colli_eng in collision_energies:
                        colli_eng_val = float(colli_eng.split()[0])
                        out_h5_key = f"pred_{name}/collision {colli_eng}/decoy {i}.json"
                        # if out_h5_key in output_h5:
                        #     continue  # skip entries that are already in the file
                        tup_to_process.append((decoy_smi, name + f'_decoy {i}', colli_eng_val, adduct, precursor_mz, inchi,
                                               out_h5_key))

            else:
                for colli_eng in collision_energies:
                    colli_eng_val = float(colli_eng.split()[0])
                    tup_to_process.append((smi, name, colli_eng_val, adduct, precursor_mz, inchi,
                                           f"pred_{name}_collision {colli_eng}.json"))
            return tup_to_process

        entries = [j for _, j in df.iterrows()]
        if kwargs["debug"]:
            entries = entries[:10]
            kwargs["num_workers"] = 0

        logging.info('Preparing entries')
        if kwargs["num_workers"] == 0:
            predict_entries = [prepare_entry(i) for i in tqdm(entries)]
        else:
            predict_entries = common.chunked_parallel(
                entries,
                prepare_entry,
                chunks=1000,
                max_cpu=kwargs["num_workers"],
            )
        predict_entries = [j for i in predict_entries for j in i]  # unroll
        random.shuffle(predict_entries)  # shuffle to evenly distribute graph size across batches
        logging.info(f'There are {len(predict_entries)} entries to process')

        batch_size = kwargs["batch_size"]
        all_batched_entries = [
            predict_entries[i: i + batch_size] for i in range(0, len(predict_entries), batch_size)
        ]

        def producer_func(batch):
            torch.set_num_threads(1)
            if gpu and avail_gpu_num >= 0:
                if kwargs["num_workers"] > 0:
                    worker_id = multiprocess.process.current_process()._identity[0]  # get worker id
                    gpu_id = worker_id % avail_gpu_num
                else:
                    gpu_id = 0
                device = f"cuda:{gpu_id}"
            else:
                device = "cpu"
            model.to(device)

            smi, name, colli_eng_val, adduct, precursor_mz, inchi, out_name = list(zip(*batch))
            pred = model.predict_mol(
                smi,
                precursor_mz=precursor_mz,
                collision_eng=colli_eng_val,
                adduct=adduct,
                threshold=kwargs["threshold"],
                device=device,
                max_nodes=kwargs["max_nodes"],
                decode_final_step=True,  # no parallel in model prediction
            )
            return_list = []
            for _inchi, _name, _pred, _colli_eng_val, _out_name in zip(inchi, name, pred, colli_eng_val, out_name):
                output = {
                    "root_inchi": _inchi,
                    "name": _name,
                    "frags": _pred,
                    "collision_energy": _colli_eng_val,
                }
                return_list.append((_out_name, json.dumps(output, indent=2)))
            return return_list

        def write_h5_func(out_entries):
            h5 = common.HDF5Dataset(save_path, mode='w')  # todo contrastive learning needs r+
            for out_batch in out_entries:
                for out_item in out_batch:
                    name, data = out_item
                    h5.write_str(name, data)
            h5.close()

        if kwargs["num_workers"] == 0:
            output_entries = [producer_func(batch) for batch in tqdm(all_batched_entries)]
            write_h5_func(output_entries)
        else:
            common.chunked_parallel(all_batched_entries, producer_func, output_func=write_h5_func,
                                    chunks=1000, max_cpu=kwargs["num_workers"])


if __name__ == "__main__":
    import time

    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
