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
from ms_pred.iceberg import inten_model, gen_model, joint_model

from rdkit import Chem
from rdkit import rdBase
from rdkit import RDLogger

rdBase.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.*")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--sparse-out", default=False, action="store_true")
    parser.add_argument("--binned-out", default=False, action="store_true")
    parser.add_argument("--sparse-k", default=100, action="store", type=int)
    parser.add_argument('--adduct-shift',default=False, action="store_true")
    parser.add_argument("--num-gpu-workers", default=0, action="store", type=int)
    parser.add_argument("--num-cpu-workers", default=32, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_ffn_pred/")
    parser.add_argument("--out-name", default="preds.hdf5")
    parser.add_argument("--num-h5-chunks", default=1, type=int)
    parser.add_argument(
        "--gen-checkpoint",
        help="name of checkpoint file",
        default="results/debug_dag_nist20/split_1/ckpt/gen/best.ckpt",
    )
    parser.add_argument(
        "--inten-checkpoint",
        help="name of checkpoint file",
        default="results/debug_dag_inten/split_1/ckpt/inten/best.ckpt",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")
    parser.add_argument("--strict-sanitize", default=False, action="store_true",
                        help="sanitize input smiles using stricter rules")
    parser.add_argument("--threshold", default=0.0, action="store", type=float)
    parser.add_argument("--max-nodes", default=100, action="store", type=int)
    parser.add_argument("--upper-limit", default=1500, action="store", type=int)
    parser.add_argument("--num-bins", default=15000, action="store", type=int)
    parser.add_argument(
        "--subset-datasets",
        default="none",
        action="store",
        choices=["none", "train_only", "test_only", "debug_special"],
    )

    return parser.parse_args()


def normalize_instrument(instrument: str) -> str:
    if pd.isna(instrument) or instrument not in common.instrument2onehot_pos:
        return "Orbitrap"
    return instrument


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
        df = df[:10]
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
    gen_checkpoint = kwargs["gen_checkpoint"]
    inten_checkpoint = kwargs["inten_checkpoint"]

    gpu = kwargs["gpu"]
    inten_model_obj = inten_model.IntenGNN.load_from_checkpoint(inten_checkpoint) #, map_location="cuda" if gpu else "cpu")
    gen_model_obj = gen_model.FragGNN.load_from_checkpoint(gen_checkpoint) #, map_location="cuda" if gpu else "cpu")
    avail_gpu_num = torch.cuda.device_count()
    use_gpu = gpu and avail_gpu_num > 0

    # Build joint model class

    logging.info(
        f"Loaded gen / inten models from {gen_checkpoint} & {inten_checkpoint}"
    )

    model = joint_model.JointModel(
        gen_model_obj=gen_model_obj, inten_model_obj=inten_model_obj
    )

    out_name = kwargs["out_name"]
    save_path = save_dir / out_name
    save_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        model.eval()
        model.freeze()
        strict_sanitize = kwargs["strict_sanitize"]

        def prepare_entry(entry):
            smi = entry["smiles"]
            adduct = entry["ionization"]
            precursor_mz = entry["precursor"]
            instrument = entry["instrument"] if "instrument" in entry else "Orbitrap"  # fallback to Orbitrap
            instrument = normalize_instrument(instrument)
            name = entry["spec"]
            inchikey = common.inchikey_from_smiles(smi)
            if strict_sanitize:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    return []
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() not in common.VALID_ELEMENTS:
                        return []
                smi = Chem.MolToSmiles(mol)  # canonicalize by roundtrip from SMILES
                if '.' in smi:  # multiple molecules / salt
                    return []
            else:
                smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))  # canonicalize by roundtrip from SMILES
            collision_energies = [i for i in ast.literal_eval(entry["collision_energies"])]
            tup_to_process = []

            for colli_eng in collision_energies:
                colli_eng_val = common.collision_energy_to_float(colli_eng)  # str to float
                if math.isnan(colli_eng_val):  # skip collision_energy == nan (no collision energy recorded)
                    continue
                tup_to_process.append((smi, f"pred_{name}", colli_eng_val, adduct, instrument, precursor_mz, f"ikey {inchikey}"))
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
            torch.set_num_threads(1)
            if use_gpu:
                if kwargs["num_gpu_workers"] > 0:
                    worker_identity = multiprocess.process.current_process()._identity
                    worker_id = worker_identity[0] - 1 if worker_identity else 0
                    gpu_id = worker_id % avail_gpu_num
                else:
                    gpu_id = 0
                device = f"cuda:{gpu_id}"
                # Avoids error in pe_embedding under multiprocessing.
                torch.cuda.set_device(gpu_id)
            else:
                device = "cpu"
            model.to(device)

            # for batch in batched_entries:
            smis, spec_names, colli_eng_vals, adducts, instruments, precursor_mzs, ikeys = list(zip(*batch))
            try:
                full_outputs = model.predict_mol(
                    smis,
                    precursor_mz=precursor_mzs,
                    collision_eng=colli_eng_vals,
                    adduct=adducts,
                    instrument=instruments,
                    threshold=kwargs["threshold"],
                    device=device,
                    max_nodes=kwargs["max_nodes"],
                    adduct_shift=kwargs["adduct_shift"],
                    canonical_root_smi=True,
                    binned_out=kwargs["binned_out"],
                )
            except:
                logging.error(
                    f'Prediction failed, SMILES: {smis}; colli_eng_vals: {colli_eng_vals}; adducts: {adducts}'
                )
                raise
            return_list = []
            for output_ind, (output_spec, spec_name, smi, ikey, adduct, pred_frag, collision_energy) in enumerate(
                    zip(full_outputs["spec"], spec_names, smis, ikeys, adducts, full_outputs["frag"], colli_eng_vals)):
                assert kwargs["sparse_out"], 'sparse_out must be True'
                output_spec = output_spec.cpu().numpy()
                pred_frag = pred_frag.cpu().numpy()
                sparse_k = kwargs["sparse_k"]
                best_inds = np.argsort(output_spec[:, 1], -1)[::-1][:sparse_k]
                output_spec = output_spec[best_inds, :]
                pred_frag = np.array(pred_frag)[best_inds]
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
                    binned_spec=full_outputs["binned_spec"][output_ind] if kwargs["binned_out"] else None,
                    num_bins=kwargs["num_bins"] if kwargs["binned_out"] else None,
                    upper_limit=kwargs["upper_limit"] if kwargs["binned_out"] else None,
                )
                return_list.append((spec_name, pred_ms))
            return return_list
        
        def write_h5_func(out_entries):
            specdb = common.PredSpecDB(
                h5_path=save_path, mode='w', num_h5s=kwargs["num_h5_chunks"],
                has_probs=False, has_brokens=False, has_masses=True, has_masses_no_adduct=False, has_frag_form_vecs=False,
                has_frags=True, has_intens=True, has_pulled_atoms=False,
                has_binned_spec=kwargs["binned_out"])
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
