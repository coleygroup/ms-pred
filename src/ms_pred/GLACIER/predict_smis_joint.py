"""predict_smis.py

Make both dag and intensity predictions jointly and revert to binned

"""
import logging
import multiprocess.process
import random
import math
import ast
from tqdm import tqdm
from datetime import datetime
import yaml
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import torch
import pytorch_lightning as pl

import ms_pred.common as common
from ms_pred.GLACIER import joint_model 

from rdkit import Chem
from rdkit import rdBase
from rdkit import RDLogger
from ms_pred.common import chem_utils

rdBase.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.*")


_WORKER_MODELS = {}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--sparse-out", default=False, action="store_true")
    parser.add_argument("--sparse-k", default=100, action="store", type=int)
    parser.add_argument("--num-gpu-workers", default=0, action="store", type=int)
    parser.add_argument("--num-cpu-workers", default=32, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_ffn_pred/")
    parser.add_argument("--out-name", default="preds.hdf5")
    parser.add_argument("--num-h5-chunks", default=1, type=int)
    parser.add_argument(
        "--checkpoint",
        help="name of checkpoint file",
        default="results/joint_train_nist20/split_1_rnd1/version_7/best.ckpt",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")
    parser.add_argument(
        "--subset-datasets",
        default="none",
        action="store",
        choices=["none", "train_only", "test_only", "debug_special"],
    )
    parser.add_argument("--binned-out", default=False, action="store_true")
    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__

    save_dir = Path(kwargs["save_dir"])
    debug = kwargs["debug"]
    common.setup_logger(save_dir, log_name="joint_pred.log", debug=debug)
    # pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(save_dir / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    data_dir = Path("")
    if kwargs.get("dataset_name") is not None:
        dataset_name = kwargs["dataset_name"]
        data_dir = Path("data/spec_datasets") / dataset_name

    labels = Path(kwargs["dataset_labels"])

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")

    if debug:
        df = df[:1000]
        kwargs["num_cpu_workers"] = 0
        kwargs["num_gpu_workers"] = 0

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
            names = splits[splits[fold_name] == "test"]["spec"].tolist()
            names = ["CCMSLIB00000001590"]
            kwargs["debug"] = True
        else:
            raise NotImplementedError()
        df = df[df["spec"].isin(names)]

    # Create model and load
    # Load from checkpoint
    checkpoint = kwargs["checkpoint"]

    gpu = kwargs["gpu"]
    avail_gpu_num = torch.cuda.device_count()
    use_gpu = gpu and avail_gpu_num > 0

    # Build joint model class

    logging.info(
        f"Loaded joint models from {checkpoint}"
    )

    model = joint_model.JointModel.load_from_checkpoint(
        checkpoint
    )

    out_name = kwargs["out_name"]
    save_path = save_dir / out_name
    save_dir.mkdir(exist_ok=True)

    with torch.inference_mode():
        model_by_device = {}

        def prepare_entry(entry):
            smi = entry["smiles"]
            adduct = entry["ionization"]
            name = entry["spec"]
            instrument = entry["instrument"] if "instrument" in entry else "Orbitrap"  # fallback to Orbitrap
            inchikey = common.inchikey_from_smiles(smi)
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    return []  # skip if rdkit can't parse the smiles
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() not in chem_utils.VALID_ATOM_NUM:
                        return []  # skip molecules with wildcard atoms
                mol=Chem.RemoveHs(mol)
                if mol.GetNumAtoms() > 100:
                    return []  # skip molecules with more than 100 heavy atoms
                smi = Chem.MolToSmiles(mol)  # canonicalize
            except:
                return []
            collision_energies = [i for i in ast.literal_eval(entry["collision_energies"])]
            tup_to_process = []

            for colli_eng in collision_energies:
                colli_eng_val = common.collision_energy_to_float(colli_eng)  # str to float
                if math.isnan(colli_eng_val):  # skip collision_energy == nan (no collision energy recorded)
                    continue
                tup_to_process.append((smi, f"pred_{name}", colli_eng_val, adduct, instrument, f"ikey {inchikey}"))
            return tup_to_process

        all_rows = [j for _, j in df.iterrows()]

        logging.info('Preparing entries')
        if kwargs["num_cpu_workers"] == 0:
            predict_entries = [prepare_entry(i) for i in tqdm(all_rows)]
        else:
            predict_entries = common.chunked_parallel(
                all_rows,
                prepare_entry,
                chunks=1000,
                max_cpu=kwargs["num_cpu_workers"],
            )
        predict_entries = [i for j in predict_entries for i in j]  # unroll
        random.shuffle(predict_entries)  # shuffle to evenly distribute graph size across batches
        logging.info(f'There are {len(predict_entries)} entries to process')

        batch_size = kwargs["batch_size"]
        all_batched_entries = [
            predict_entries[i: i + batch_size] for i in range(0, len(predict_entries), batch_size)
        ]

        def producer_func(batch):
            global _WORKER_MODELS
            torch.set_num_threads(1)
            if use_gpu:
                if kwargs["num_gpu_workers"] > 0:
                    worker_id = multiprocess.process.current_process()._identity[0]  # get worker id
                    gpu_id = worker_id % avail_gpu_num
                else:
                    gpu_id = 0
                device = f"cuda:{gpu_id}"
            else:
                device = "cpu"

            if kwargs["num_gpu_workers"] > 0:
                worker_key = f"{multiprocess.process.current_process().pid}:{device}"
                if worker_key not in _WORKER_MODELS:
                    worker_model = joint_model.JointModel.load_from_checkpoint(checkpoint)
                    worker_model.eval()
                    worker_model.freeze()
                    worker_model = worker_model.to(device)
                    _WORKER_MODELS[worker_key] = worker_model
                local_model = _WORKER_MODELS[worker_key]
            else:
                if device not in model_by_device:
                    local_model = model.to(device)
                    model_by_device[device] = local_model
                    local_model.eval()
                    local_model.freeze()
                else:
                    local_model = model_by_device[device]

            # for batch in batched_entries:
            smis, spec_names, colli_eng_vals, adducts, instruments, ikeys = list(zip(*batch))
            full_outputs = local_model.predict_mol(
                smis,
                collision_eng=colli_eng_vals,
                adduct=adducts,
                instrument=instruments,
                device=device,
            )

            return_list = []
            for output_spec, spec_name, smi, ikey, adduct, pred_frag, collision_energy in \
                    zip(full_outputs["spec"], spec_names, smis, ikeys, adducts, full_outputs["frag"], colli_eng_vals):
                assert kwargs["sparse_out"], 'sparse_out must be True'
                sparse_k = kwargs["sparse_k"]
                top_k = min(sparse_k, output_spec.shape[0])
                best_inds = torch.topk(output_spec[:, 1], k=top_k, largest=True).indices
                output_spec = output_spec.index_select(0, best_inds).cpu().numpy()
                pred_frag = pred_frag.index_select(0, best_inds).cpu().numpy()
                masses = output_spec[:, 0]
                intens = output_spec[:, 1]
                pred_ms = common.MassSpec(
                    root_canonical_smiles=smi,
                    adduct=adduct,
                    collision_energy=collision_energy,
                    masses=masses,
                    intens=intens,
                    frags=pred_frag,
                    remark=ikey,
                )
                return_list.append((spec_name, pred_ms))
            return return_list
        
        def write_h5_func(out_entries):
            specdb = common.PredSpecDB(
                h5_path=save_path, mode='w', num_h5s=kwargs["num_h5_chunks"],
                has_probs=False, has_brokens=False, has_masses=True, has_masses_no_adduct=False, has_frag_form_vecs=False,
                has_frags=True, has_intens=True, has_pulled_atoms=False)
            for out_batch in out_entries:
                for out_item in out_batch:
                    name, spec = out_item
                    specdb.write(name, spec)
            specdb.close()

        if use_gpu:
            if kwargs["num_gpu_workers"] == 0:
                output_entries = [producer_func(batch) for batch in tqdm(all_batched_entries)]
                write_h5_func(output_entries)
            else:
                common.chunked_parallel(all_batched_entries, producer_func, output_func=write_h5_func,
                                        chunks=1000, max_cpu=kwargs["num_gpu_workers"])
        else:
            if kwargs["num_cpu_workers"] == 0:
                output_entries = [producer_func(batch) for batch in tqdm(all_batched_entries)]
                write_h5_func(output_entries)
            else:
                common.chunked_parallel(all_batched_entries, producer_func, output_func=write_h5_func,
                                        chunks=1000, max_cpu=kwargs["num_cpu_workers"])


if __name__ == "__main__":
    import time

    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
