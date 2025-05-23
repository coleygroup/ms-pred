"""predict_smis.py

Make both dag and intensity predictions jointly and revert to binned

"""
import logging
import multiprocess.process
import random
import math
import ast
import json
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
from ms_pred.dag_pred import inten_model, gen_model, joint_model


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
    parser.add_argument("--sparse-k", default=100, action="store", type=int)
    parser.add_argument("--binned-out", default=False, action="store_true")
    parser.add_argument('--adduct-shift',default=False, action="store_true")
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_ffn_pred/")
    parser.add_argument(
        "--gen-checkpoint",
        help="name of checkpoint file",
        default="results/2022_06_22_pretrain/version_3/epoch=99-val_loss=0.87.ckpt",
    )
    parser.add_argument(
        "--inten-checkpoint",
        help="name of checkpoint file",
        default="results/2022_06_22_pretrain/version_3/epoch=99-val_loss=0.87.ckpt",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")
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


def predict():
    args = get_args()
    kwargs = args.__dict__

    save_dir = Path(kwargs["save_dir"])
    debug = kwargs["debug"]
    common.setup_logger(save_dir, log_name="joint_pred.log", debug=debug)
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

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

    if kwargs["debug"]:
        df = df[:10]

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

    # Build joint model class

    logging.info(
        f"Loaded gen / inten models from {gen_checkpoint} & {inten_checkpoint}"
    )

    model = joint_model.JointModel(
        gen_model_obj=gen_model_obj, inten_model_obj=inten_model_obj
    )

    with torch.no_grad():
        model.eval()
        model.freeze()

        binned_out = kwargs["binned_out"]

        def prepare_entry(entry):
            smi = entry["smiles"]
            adduct = entry["ionization"]
            precursor_mz = entry["precursor"]
            name = entry["spec"]
            inchikey = common.inchikey_from_smiles(smi)
            collision_energies = [i for i in ast.literal_eval(entry["collision_energies"])]
            tup_to_process = []

            for colli_eng in collision_energies:
                colli_eng_val = common.collision_energy_to_float(colli_eng)  # str to float
                if math.isnan(colli_eng_val):  # skip collision_energy == nan (no collision energy recorded)
                    continue
                tup_to_process.append((smi, name, colli_eng_val, adduct, precursor_mz,
                                       f"pred_{name}/ikey {inchikey}/collision {colli_eng}"))
            return tup_to_process

        all_rows = [j for _, j in df.iterrows()]

        logging.info('Preparing entries')
        if kwargs["num_workers"] == 0:
            predict_entries = [prepare_entry(i) for i in tqdm(all_rows)]
        else:
            predict_entries = common.chunked_parallel(
                all_rows,
                prepare_entry,
                chunks=1000,
                max_cpu=kwargs["num_workers"],
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

            # for batch in batched_entries:
            smis, spec_names, colli_eng_vals, adducts, precursor_mzs, h5_names = list(zip(*batch))
            full_outputs = model.predict_mol(
                smis,
                precursor_mz=precursor_mzs,
                collision_eng=colli_eng_vals,
                adduct=adducts,
                threshold=kwargs["threshold"],
                device=device,
                max_nodes=kwargs["max_nodes"],
                binned_out=binned_out,
                adduct_shift=kwargs["adduct_shift"],
            )
            return_list = []
            if binned_out:
                for output_spec, spec_name, smi, h5_name in \
                        zip(full_outputs["spec"], spec_names, smis, h5_names):
                    if kwargs["sparse_out"]:
                        sparse_k = kwargs["sparse_k"]
                        best_inds = np.argsort(output_spec, -1)[::-1][:sparse_k]
                        best_intens = np.take_along_axis(output_spec, best_inds, -1)
                        output_spec = np.stack([best_inds, best_intens], -1)

                    inchikey = common.inchikey_from_smiles(smi)
                    return_list.append((h5_name, spec_name, smi, inchikey, output_spec, None))
            else:
                for output_spec, spec_name, smi, h5_name, pred_frag in \
                        zip(full_outputs["spec"], spec_names, smis, h5_names, full_outputs["frag"]):
                    assert kwargs["sparse_out"], 'sparse_out must be True for non-binned output'
                    sparse_k = kwargs["sparse_k"]
                    best_inds = np.argsort(output_spec[:, 1], -1)[::-1][:sparse_k]
                    output_spec = output_spec[best_inds, :]
                    pred_frag = np.array(pred_frag)[best_inds]
                    inchikey = common.inchikey_from_smiles(smi)
                    return_list.append((h5_name, spec_name, smi, inchikey, output_spec, pred_frag))
            return return_list

        if binned_out:
            out_name = "binned_preds.hdf5"
        else:
            out_name = "preds.hdf5"

        def write_h5_func(out_entries):
            h5 = common.HDF5Dataset(save_dir / out_name, mode='w')
            h5.attrs['num_bins'] = 15000
            h5.attrs['upper_limit'] = 1500
            h5.attrs['sparse_out'] = kwargs["sparse_out"]
            for out_batch in out_entries:
                for out_item in out_batch:
                    h5_name, spec_name, smi, inchikey, output_spec, pred_frag = out_item
                    h5.write_data(h5_name + '/spec', output_spec)
                    if pred_frag is not None:
                        h5.write_str(h5_name + '/frag', json.dumps(pred_frag.tolist()))  # save as string avoids overflow
                    h5.update_attr(h5_name, {'smiles': smi, 'ikey': inchikey, 'spec_name': spec_name})
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
