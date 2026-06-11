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
from rdkit import Chem


import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything



import ms_pred.common as common
from ms_pred.marason import inten_model, gen_model, joint_model, dag_data

import scipy.sparse as sparse
from sklearn.metrics import pairwise_distances

from rdkit import rdBase
from rdkit import RDLogger

rdBase.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.*")


def build_gen_magma_map(magma_tree_path: Path, strip_pred_prefix: bool = False):
    predspec_db = common.PredSpecDB(magma_tree_path)
    name_to_entry = {}
    for name in predspec_db.get_all_names():
        map_name = name[5:] if strip_pred_prefix and name.startswith("pred_") else name
        ces, remarks = predspec_db.get_entries(name)
        for ce, remark in zip(ces, remarks):
            name_to_entry[f"{map_name}_collision {ce}"] = (name, ce, remark)
    if len(name_to_entry) > 0:
        return name_to_entry

    legacy_h5 = common.HDF5Dataset(magma_tree_path)
    return {
        Path(name).stem.replace("pred_", "") if strip_pred_prefix else Path(name).stem: name
        for name in legacy_h5.get_all_names()
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--sparse-out", default=False, action="store_true")
    parser.add_argument("--sparse-k", default=100, action="store", type=int)
    parser.add_argument("--binned-out", default=False, action="store_true")
    parser.add_argument('--adduct-shift',default=False, action="store_true")
    parser.add_argument("--num-gpu-workers", default=0, action="store", type=int)
    parser.add_argument("--num-cpu-workers", default=32, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_ffn_pred/")
    parser.add_argument("--ref-dir", default=None)
    parser.add_argument(
        "--gen-checkpoint",
        help="name of checkpoint file",
        default="results/debug_marason/split_1/ckpt/gen/best.ckpt",
    )
    parser.add_argument(
        "--inten-checkpoint",
        help="name of checkpoint file",
        default="results/debug_marason_inten/split_1/ckpt/inten/best.ckpt",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")
    parser.add_argument("--threshold", default=0.0, action="store", type=float)
    parser.add_argument("--max-nodes", default=100, action="store", type=int)
    parser.add_argument("--upper-limit", default=1500, action="store", type=int)
    parser.add_argument("--num-bins", default=15000, action="store", type=int)
    parser.add_argument("--keep", default=False, action="store_true")
    parser.add_argument(
        "--subset-datasets",
        default="none",
        action="store",
        choices=["none", "train_only", "test_only", "debug_special"],
    )
    parser.add_argument(
        "--magma-dag-folder",
        default="data/spec_datasets/gnps2015_debug/magma_outputs/magma_tree",
        help="Folder to have outputs",
    )
    parser.add_argument(
        "--root-encode",
        default="gnn",
        action="store",
        choices=["gnn", "fp"],
        help="How to encode root of trees",
    )
    parser.add_argument("--pe-embed-k", default=0, action="store", type=int)
    parser.add_argument("--max-ref-count", default=10, action="store", type=int)
    parser.add_argument("--binned-targs", default=True, action="store_true")
    parser.add_argument("--embed-elem-group", default=False, action="store_true")
    parser.add_argument("--add-hs", default=True, action="store_true")
    parser.add_argument("--add-ref", default=False, action="store_true")
    parser.add_argument("--filter", default=False, action="store_true")
    parser.add_argument("--mol-threshold", default=0.5, action="store")

    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__

    save_dir = Path(kwargs["save_dir"])
    debug = kwargs["debug"]
    common.setup_logger(save_dir, log_name="joint_pred.log", debug=debug)
    if pl.__version__.startswith("1"):
        pl.utilities.seed.seed_everything(kwargs.get("seed"))
    else: # simpler import...?
        pl.seed_everything(kwargs.get("seed"))

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
    df = df.dropna(subset=["smiles"])

    if kwargs["debug"]:
        #df = df[:10]
        df = df[:10000]
        kwargs["num_cpu_workers"] = 0
        kwargs["num_gpu_workers"] = 0
        print(df.spec.value_counts())
        # print(df.spec.unique()) # should be 1 if there are 256 entries / decoys per actual test entry.. 

    use_gpu = kwargs["gpu"] and torch.cuda.device_count() > 0

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
    if "instrument" not in df:
        df["instrument"] = "Orbitrap"

    # Create model and load
    # Load from checkpoint
    gen_checkpoint = kwargs["gen_checkpoint"]
    inten_checkpoint = kwargs["inten_checkpoint"]

    gpu = kwargs["gpu"]
    inten_model_obj = inten_model.IntenGNN.load_from_checkpoint(inten_checkpoint) #, map_location="cuda" if gpu else "cpu")
    gen_model_obj = gen_model.FragGNN.load_from_checkpoint(gen_checkpoint) #, map_location="cuda" if gpu else "cpu")
    avail_gpu_num = torch.cuda.device_count()

    logging.info(
        f"Loaded gen / inten models from {gen_checkpoint} & {inten_checkpoint}"
    )

    if kwargs["filter"]:
        dist_df = pd.read_csv("data/msg/closest_neighbors/infinite/test.csv")
        dist_df = dist_df[dist_df["distance"] < kwargs["mol_threshold"]]
        valid_spec = set(dist_df["names_ori"].tolist())
        def filter_map(r):
            return r["spec"] in valid_spec
        df["filter_condition"] = df.apply(filter_map, axis=1)
        df=df[df["filter_condition"]]

    root_morgans = db_engs = db_specs = data_adducts = ref_dataset = None
    if kwargs["ref_dir"] is not None and kwargs["add_ref"]:
        num_workers = kwargs.get("num_workers", 0)
        magma_dag_folder = Path(kwargs["magma_dag_folder"])
        name_to_json = build_gen_magma_map(magma_dag_folder, strip_pred_prefix=True)

        pe_embed_k = kwargs["pe_embed_k"]
        root_encode = kwargs["root_encode"]
        binned_targs = kwargs["binned_targs"]
        embed_elem_group = kwargs["embed_elem_group"]
        add_hs = kwargs["add_hs"]

        tree_processor = dag_data.TreeProcessor(
            pe_embed_k=pe_embed_k,
            root_encode=root_encode,
            binned_targs=binned_targs,
            add_hs=add_hs,
            embed_elem_group = embed_elem_group,
        )
        ref_dir = Path(kwargs["ref_dir"])
        ref_df = pd.read_csv(ref_dir/"train_subset.csv") 
        if kwargs["debug"]:
            ref_df = ref_df[:100]
            db_specs = sparse.load_npz(ref_dir/"db_debug_specs.npz")
        else:
            db_specs = sparse.load_npz(ref_dir/"db_specs.npz")
        ref_dataset = dag_data.IntenDataset(
            ref_df,
            magma_h5=magma_dag_folder,
            magma_map=name_to_json,
            num_workers=16,
            tree_processor=tree_processor,
            specs=db_specs
        )

        def load_infos(name):
            tree = ref_dataset.load_tree(name)
            if isinstance(tree, dict):
                smi = tree["root_canonical_smiles"]
                colli_eng = float(tree["collision_energy"])
                adduct = tree["adduct"]
                instrument = tree.get("instrument", "Orbitrap")
            else:
                smi = tree.root_canonical_smiles
                colli_eng = float(tree.collision_energy)
                adduct = tree.adduct
                instrument = tree.info.get("instrument", "Orbitrap")
            morgan_fp = common.get_morgan_fp_smi(smi, isbool=True)
            adducts = common.ion2onehot_pos[adduct]
            instrument = common.instrument2onehot_pos[instrument]
            return [morgan_fp, colli_eng, adducts, instrument] 
        db_infos = common.chunked_parallel(ref_dataset.spec_names,load_infos, max_cpu=64)
        db_engs_array, root_morgans_array, data_adducts_array, db_instruments = [], [], [], []
        for db_info in db_infos:
            root_morgans_array.append(db_info[0])
            db_engs_array.append(db_info[1])
            data_adducts_array.append(db_info[2])
            db_instruments.append(db_info[3])
        root_morgans = np.stack(root_morgans_array)
        data_adducts = np.stack(data_adducts_array)
        db_engs = np.stack(db_engs_array).astype(np.float32)
        db_instruments = np.stack(db_instruments).astype(np.float32)

    def find_ref(adduct, morgan, instrument, keep=False):
        train_distance = pairwise_distances(morgan[None, :], root_morgans, metric = "jaccard")[0, :]
        adduct = float(common.ion2onehot_pos[adduct])
        instrument = float(common.instrument2onehot_pos[instrument])
        distance = np.where(data_adducts == adduct, train_distance, 1)
        distance = np.where(db_instruments == instrument, distance, 1)
        if not keep:
            distance = np.where(np.isclose(distance, 0), 1, distance)
        ranks = np.argsort(distance)
        min_distance = np.min(distance).item()
        valid_ref_count = 1 if min_distance == 1 else np.count_nonzero(min_distance == distance)

        closest = np.copy(ranks[:valid_ref_count])
        return (min_distance, valid_ref_count, closest)
    
    # Build joint model class  
    model = joint_model.JointModel(
        gen_model_obj=gen_model_obj, inten_model_obj=inten_model_obj,
        db=ref_dataset, ref_engs=db_engs, 
        ref_specs=db_specs, add_ref=kwargs["add_ref"],
        max_ref_count=kwargs["max_ref_count"]
    )

    with torch.no_grad():
        model.eval()
        model.freeze()

        binned_out = kwargs["binned_out"]

        def prepare_entry(entry):
            smi = entry["smiles"]
            adduct = entry["ionization"]
            precursor_mz = entry["precursor"]
            instrument = entry["instrument"]
            name = entry["spec"]
            if type(smi) is not str:
                logging.error(smi, type(smi))
                logging.error(f"SMILES is not a string: {smi}")
                logging.error(f"Possible inchikey: {entry.get('inchikey', None)}")
                return []
            try:
                inchikey = common.inchikey_from_smiles(smi)
            except Exception as e:
                logging.error(f"Could not get inchikey for {smi}: {e}")
                inchikey = entry.get("inchikey", None)
            min_distance = valid_ref_count = closest = None
            if kwargs["add_ref"]:
                smi_no_stereo = common.rm_stereo(smi)
                mol = common.smi_inchi_round_mol(smi_no_stereo)
                canonical_smi = Chem.MolToSmiles(mol)  # canonical smiles
                morgan = common.get_morgan_fp_smi(canonical_smi, isbool=True)
                min_distance, valid_ref_count, closest = find_ref(adduct, morgan, instrument, keep=kwargs["keep"])

            collision_energies = [i for i in ast.literal_eval(entry["collision_energies"])]
            tup_to_process = []

            for colli_eng in collision_energies:
                colli_eng_val = common.collision_energy_to_float(colli_eng)  # str to float
                if math.isnan(colli_eng_val):  # skip collision_energy == nan (no collision energy recorded)
                    continue
                if kwargs["add_ref"]:
                    closest_engs = np.take(db_engs, closest)
                    closest_engs_diff = np.abs(closest_engs - colli_eng_val)
                    closest_engs_rank = np.argsort(closest_engs_diff)
                    closest = np.take(closest, closest_engs_rank)

                tup_to_process.append(
                    (
                        smi,
                        f"pred_{name}",
                        colli_eng_val,
                        adduct,
                        instrument,
                        precursor_mz,
                        f"ikey {inchikey}",
                        min_distance,
                        valid_ref_count,
                        closest,
                    )
                )
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

        out_name = "binned_preds.hdf5" if binned_out else "preds.hdf5"
        save_path = save_dir / out_name

        def producer_func(batch):
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
            model.to(device)
            if use_gpu:
                torch.cuda.set_device(gpu_id)

            smis, spec_names, colli_eng_vals, adducts, instruments, precursor_mzs, remarks, min_distances, valid_ref_counts, closests = list(zip(*batch))
            full_outputs = model.predict_mol(
                smis,
                precursor_mz=precursor_mzs,
                collision_eng=colli_eng_vals,
                adduct=adducts,
                instrument=instruments,
                threshold=kwargs["threshold"],
                device=device,
                max_nodes=kwargs["max_nodes"],
                binned_out=binned_out,
                adduct_shift=kwargs["adduct_shift"],
                min_distances = min_distances,
                valid_ref_counts = valid_ref_counts,
                closests = closests,
                name = spec_names,
            )
            return_list = []
            for output_ind, (output_spec, spec_name, smi, remark, adduct, pred_frag, collision_energy) in enumerate(
                zip(
                    full_outputs["spec"],
                    spec_names,
                    smis,
                    remarks,
                    adducts,
                    full_outputs["frag"],
                    colli_eng_vals,
                )
            ):
                if kwargs["sparse_out"]:
                    sparse_k = kwargs["sparse_k"]
                    best_inds = np.argsort(output_spec[:, 1], -1)[::-1][:sparse_k]
                    output_spec = output_spec[best_inds, :]
                    pred_frag = pred_frag[best_inds]

                pred_ms = common.MassSpec(
                    root_canonical_smiles=smi,
                    adduct=adduct,
                    collision_energy=collision_energy,
                    masses=output_spec[:, 0],
                    intens=output_spec[:, 1],
                    frags=pred_frag,
                    remark=remark,
                    binned_spec=full_outputs["binned_spec"][output_ind] if kwargs["binned_out"] else None,
                    num_bins=kwargs["num_bins"] if kwargs["binned_out"] else None,
                    upper_limit=kwargs["upper_limit"] if kwargs["binned_out"] else None,
                )
                return_list.append((spec_name, pred_ms))
            return return_list

        def write_h5_func(out_entries):
            specdb = common.PredSpecDB(
                h5_path=save_path,
                mode='w',
                has_probs=False,
                has_brokens=False,
                has_masses=True,
                has_masses_no_adduct=False,
                has_frag_form_vecs=False,
                has_frags=True,
                has_intens=True,
                has_pulled_atoms=False,
                has_binned_spec=kwargs["binned_out"],
            )
            for out_batch in out_entries:
                for name, spec in out_batch:
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
