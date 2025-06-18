"""train_inten.py

Train model to predict emit intensities for each fragment

"""
import logging
import yaml
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import ms_pred.common as common
from ms_pred.marason import dag_data, inten_model
from sklearn.metrics import pairwise_distances
import torch
import numpy as np
import scipy.sparse as sparse
import warnings 
import os
import json
warnings.filterwarnings('ignore') 

torch.multiprocessing.set_sharing_strategy('file_system')


def add_frag_train_args(parser):
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debug-overfit", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_tree_pred/")

    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument(
        "--magma-dag-folder",
        default="data/spec_datasets/gnps2015_debug/magma_outputs/magma_tree",
        help="Folder to have outputs",
    )
    parser.add_argument("--split-name", default="split_1.tsv")
    parser.add_argument("--reference-dir", default="data/msg/closest_neighbors/10eV")
    parser.add_argument("--batch-size", default=3, action="store", type=int)
    parser.add_argument("--max-epochs", default=100, action="store", type=int)
    parser.add_argument("--min-epochs", default=0, action="store", type=int)
    parser.add_argument("--learning-rate", default=7e-4, action="store", type=float)
    parser.add_argument("--lr-decay-rate", default=1.0, action="store", type=float)
    parser.add_argument("--weight-decay", default=0, action="store", type=float)
    parser.add_argument("--test-checkpoint", default="", action="store", type=str)

    # Fix model params
    parser.add_argument("--gnn-layers", default=3, action="store", type=int)
    parser.add_argument("--mlp-layers", default=2, action="store", type=int)
    parser.add_argument("--frag-set-layers", default=2, action="store", type=int)
    parser.add_argument("--inten-layers", default=2, action="store", type=int)

    parser.add_argument("--set-layers", default=1, action="store", type=int)
    parser.add_argument("--pe-embed-k", default=0, action="store", type=int)
    parser.add_argument("--dropout", default=0, action="store", type=float)
    parser.add_argument("--hidden-size", default=256, action="store", type=int)
    parser.add_argument("--pool-op", default="avg", action="store")
    parser.add_argument("--grad-accumulate", default=1, type=int, action="store")
    parser.add_argument("--sk-tau", default=0.01, action="store", type=float)
    parser.add_argument("--softmax-tau", default=0.01, action="store", type=float)
    parser.add_argument("--ppm-tol", default=20, action="store", type=float)
    parser.add_argument("--eng-threshold", default=float("inf"), action="store", type=float)
    parser.add_argument("--mol-threshold", default=0.5, action="store", type=float)
    parser.add_argument("--max-ref-count", default=1, action="store", type=int)
    parser.add_argument("--node-matching-weight", default=1, action="store", type=float)
    parser.add_argument("--embed-instrument", default=False, action="store_true")
    parser.add_argument(
        "--mpnn-type", default="GGNN", action="store", choices=["GGNN", "GINE", "PNA"]
    )
    parser.add_argument("--matching-method", default="softmax", action="store", choices = ["sinkhorn", "hungarian", "rrwm", "softmax", "none"])
    parser.add_argument(
        "--loss-fn",
        default="cosine",
        action="store",
        choices=["cosine", "entropy", "weighted_entropy"],
    )
    parser.add_argument(
        "--root-encode",
        default="gnn",
        action="store",
        choices=["gnn", "fp"],
        help="How to encode root of trees",
    )
    parser.add_argument("--logger", 
                        default="dag_inten_train",  
                        action="store")
    parser.add_argument("--inject-early", default=False, action="store_true")
    parser.add_argument("--include-unshifted-mz", default=False, action="store_true")
    parser.add_argument("--binned-targs", default=False, action="store_true")
    parser.add_argument("--embed-adduct", default=False, action="store_true")
    parser.add_argument("--embed-collision", default=False, action="store_true")
    parser.add_argument("--embed-elem-group", default=False, action="store_true")
    parser.add_argument("--encode-forms", default=False, action="store_true")
    parser.add_argument("--add-hs", default=False, action="store_true")
    parser.add_argument("--multi-gnn", default=False, action="store_true")
    parser.add_argument("--multi-mlp", default=False, action="store_true")
    parser.add_argument("--add-reference", default=False, action="store_true")
    parser.add_argument("--load-reference", default=False, action="store_true")
    parser.add_argument("--save-reference", default=False, action="store_true")
    parser.add_argument("--filter", default=False, action="store_true")
    parser.add_argument("--filter-valid-test", default=False, action="store_true")
    parser.add_argument("--single-ref", default=False, action="store_true")

    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_frag_train_args(parser)
    return parser.parse_args()

def train_model():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    logger_name = kwargs["logger"] + "hidden_size_" + str(kwargs["hidden_size"]) + "add_ref_" + str(kwargs["add_reference"]) + "_lr_" + str(kwargs["learning_rate"]) + "node_weight" + str(kwargs["node_matching_weight"]) + "filter_valid_test" + str(kwargs["filter_valid_test"]) + ".log"
    
    common.setup_logger(save_dir, log_name=logger_name, debug=kwargs["debug"])
    pl.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)
        

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = common.get_data_dir(dataset_name)
    labels = data_dir / kwargs["dataset_labels"]
    split_file = data_dir / "splits" / kwargs["split_name"]
    add_hs = kwargs["add_hs"]

    add_ref = kwargs['add_reference']
    load_ref = kwargs['load_reference']
    save_ref = kwargs['save_reference']


    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    if "instrument" not in df:
        df["instrument"] = "Orbitrap"
    if kwargs["debug"]:
        df = df[:1000]

    spec_names = df["spec"].values
    if kwargs["debug_overfit"]:
        train_inds, val_inds, test_inds = common.get_splits(
            spec_names, split_file
        )
        train_inds = train_inds[:1000]
    else:
        train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]

    ref_dir = Path(kwargs["reference_dir"])

    if kwargs["debug"]:
        train_df.to_csv(ref_dir/"train_subset_debug.csv")
        val_df.to_csv(ref_dir/"val_subset_debug.csv")
        test_df.to_csv(ref_dir/"test_subset_debug.csv") 
    else:
        train_df.to_csv(ref_dir/"train_subset.csv")
        val_df.to_csv(ref_dir/"val_subset.csv")
        test_df.to_csv(ref_dir/"test_subset.csv")


    num_workers = kwargs.get("num_workers", 0)
    magma_dag_folder = Path(kwargs["magma_dag_folder"])
    magma_tree_h5 = common.HDF5Dataset(magma_dag_folder)
    name_to_json = {Path(i).stem.replace("pred_", ""): i for i in magma_tree_h5.get_all_names()}


    pe_embed_k = kwargs["pe_embed_k"]
    root_encode = kwargs["root_encode"]
    binned_targs = kwargs["binned_targs"]
    embed_elem_group = kwargs["embed_elem_group"]
    if kwargs["matching_method"] == 'rrwm' or kwargs["matching_method"] == 'hungarian':
        include_frag_morgans = True
    else:
        include_frag_morgans = False
    tree_processor = dag_data.TreeProcessor(
        pe_embed_k=pe_embed_k,
        root_encode=root_encode,
        binned_targs=binned_targs,
        add_hs=add_hs,
        embed_elem_group = embed_elem_group,
        include_frag_morgan=include_frag_morgans
    )
    db_engs = db_specs = database_set = None
    train_closest = test_closest = val_closest = train_distances = val_distances = test_distances = None
    train_valid_ref_counts = test_valid_ref_counts = val_valid_ref_counts = None
    # ordered_magma_tree_h5_folder = magma_dag_folder.parent
    # if kwargs["debug"]:
    #     ordered_magma_tree_h5_folder = ordered_magma_tree_h5_folder/"debug"
    database_df = train_df.copy(deep=True)
    db_specs, db_engs = None, None
    if add_ref or kwargs["filter"] or kwargs["filter_valid_test"]:
        database_set = dag_data.IntenDataset(
            database_df,
            magma_h5=magma_dag_folder,
            magma_map=name_to_json,
            num_workers=num_workers,
            tree_processor=tree_processor,
        )

        if load_ref:
            if kwargs["debug"]:
                train_closest_df = pd.read_csv(ref_dir / "train_debug.csv")
                val_closest_df = pd.read_csv(ref_dir / "val_debug.csv")
                test_closest_df = pd.read_csv(ref_dir / "test_debug.csv")
                train_distances = train_closest_df["distance"]
                val_distances = val_closest_df["distance"]
                test_distances = test_closest_df["distance"]
                train_valid_ref_counts = train_closest_df["valid_ref_count"]
                val_valid_ref_counts = val_closest_df["valid_ref_count"]
                test_valid_ref_counts = test_closest_df["valid_ref_count"]
                train_closest = np.load(ref_dir / "train_debug_ref.npy")
                val_closest = np.load(ref_dir / "val_debug_ref.npy")
                test_closest = np.load(ref_dir / "test_debug_ref.npy")
                db_specs = sparse.load_npz(ref_dir / "db_debug_specs.npz")
                db_engs = np.load(ref_dir / "db_debug_engs.npy")
            else:
                train_closest_df = pd.read_csv(ref_dir / "train.csv")
                val_closest_df = pd.read_csv(ref_dir / "val.csv")
                test_closest_df = pd.read_csv(ref_dir / "test.csv")
                train_closest = np.load(ref_dir / "train_ref.npy")
                val_closest =  np.load(ref_dir / "val_ref.npy")
                test_closest =  np.load(ref_dir / "test_ref.npy")
                train_valid_ref_counts = train_closest_df["valid_ref_count"]
                val_valid_ref_counts = val_closest_df["valid_ref_count"]
                test_valid_ref_counts = test_closest_df["valid_ref_count"]
                train_distances = train_closest_df["distance"]
                val_distances = val_closest_df["distance"]
                test_distances = test_closest_df["distance"]
                db_specs = sparse.load_npz(ref_dir / "db_specs.npz")
                db_engs = np.load(ref_dir / "db_engs.npy")
        else:
            train_dataset = dag_data.IntenDataset(
                train_df,
                magma_h5=magma_dag_folder,
                magma_map=name_to_json,
                num_workers=num_workers,
                tree_processor=tree_processor,
            )
            val_dataset = dag_data.IntenDataset(
                val_df,
                magma_h5=magma_dag_folder,
                magma_map=name_to_json,
                num_workers=num_workers,
                tree_processor=tree_processor,
            )
            test_dataset = dag_data.IntenDataset(
                test_df,
                magma_h5=magma_dag_folder,
                magma_map=name_to_json,
                num_workers=num_workers,
                tree_processor=tree_processor,
            )

            def load_infos(name):
                tree_h5 = common.HDF5Dataset(magma_dag_folder)
                tree = json.loads(tree_h5.read_str(f"{name}.json"))
                smi = tree['root_canonical_smiles']
                morgan_fp = common.get_morgan_fp_smi(smi, isbool=True)
                colli_eng = float(common.get_collision_energy(name))
                specs = np.array(tree["raw_spec"], dtype=np.float32)
                bin_posts = np.clip(np.digitize(specs[:, 0], tree_processor.bins), 0, len(tree_processor.bins) - 1)
                new_out = np.zeros_like(tree_processor.bins, dtype=np.float32)
                for bin_post, inten in zip(bin_posts, specs[:, 1]):
                    new_out[bin_post] = max(new_out[bin_post], inten)
                adducts = common.ion2onehot_pos[tree["adduct"]]
                tree_h5.close()
                instrument = common.instrument2onehot_pos[tree["instrument"]]
                return [morgan_fp, colli_eng, adducts, new_out, instrument] 
            pad_list = []
            db_infos = common.chunked_parallel(train_dataset.spec_names,load_infos, max_cpu=64)
            db_specs, db_engs_array, root_morgans_array, data_adducts_array, db_instruments = [], [], [], [], []
            for db_info in db_infos:
                root_morgans_array.append(db_info[0])
                db_engs_array.append(db_info[1])
                data_adducts_array.append(db_info[2])
                db_specs.append(db_info[3])
                db_instruments.append(db_info[4])
            root_morgans = np.stack(root_morgans_array)
            data_adducts = np.stack(data_adducts_array)
            db_engs = np.stack(db_engs_array).astype(np.float32)
            db_specs = sparse.csr_array(np.stack(db_specs))
            db_instruments = np.stack(db_instruments).astype(np.float32)

            def find_ref(morgans_with_info):                
                morgans, adduct, collision_eng, instrument = morgans_with_info
                eng_threshold = kwargs['eng_threshold']
                morgans = morgans[None, :]
                train_distance = pairwise_distances(morgans, root_morgans, metric = "jaccard")[0]
                distance = np.where(data_adducts == adduct, np.where(np.isclose(train_distance, 0), 1, train_distance), 1)
                distance = np.where(db_instruments == instrument, distance, 1)
                diffs = np.abs(db_engs - collision_eng)
                distance = np.where(diffs < eng_threshold, distance, 1)

                ranks = np.lexsort((diffs, distance))
                min_distances = np.min(distance)
                valid_ref_count = np.count_nonzero(min_distances == distance)
                valid_ref_count = np.where(min_distances == 1, 1, valid_ref_count)
                closest = np.copy(ranks[:np.max(valid_ref_count)])
                return (min_distances, valid_ref_count, closest)
            def pad(v):
                return np.pad(v, ((0, pad_list[-1] - v.shape[0])))

            train_root_morgans = root_morgans_array
            train_adducts = data_adducts_array
            train_engs = db_engs_array
            train_specs = db_specs
            train_instruments = db_instruments
            train_morgan_with_info = list(zip(train_root_morgans, train_adducts, train_engs, train_instruments))
            infos = common.chunked_parallel(train_morgan_with_info, find_ref, max_cpu=64)
            train_valid_ref_counts, train_closest, train_distances = [], [], []
            for info in infos:
                train_distances.append(info[0])
                train_valid_ref_counts.append(info[1])
                train_closest.append(info[2])
            pad_list.append(max((closest.shape[0] for closest in train_closest)))

            train_closest = np.stack(common.chunked_parallel(train_closest, pad, max_cpu=64))
            train_valid_ref_counts = np.stack(train_valid_ref_counts)
            train_distances = np.stack(train_distances)

            val_infos = common.chunked_parallel(val_dataset.spec_names,load_infos, max_cpu=64)
            val_specs, val_engs, val_root_morgans, val_adducts, val_instruments = [], [], [], [], []
            for val_info in val_infos:
                val_root_morgans.append(val_info[0])
                val_engs.append(val_info[1])
                val_adducts.append(val_info[2])
                val_specs.append(val_info[3])
                val_instruments.append(val_info[4])
            val_specs = sparse.csr_array(np.stack(val_specs))

            val_morgan_with_info = list(zip(val_root_morgans, val_adducts, val_engs, val_instruments))
            infos = common.chunked_parallel(val_morgan_with_info, find_ref, max_cpu=64)
            val_valid_ref_counts, val_closest, val_distances = [], [], []
            for info in infos:
                val_distances.append(info[0])
                val_valid_ref_counts.append(info[1])
                val_closest.append(info[2])
            pad_list.append(max((closest.shape[0] for closest in val_closest)))

            val_closest = np.stack(common.chunked_parallel(val_closest, pad, max_cpu=64))
            val_valid_ref_counts = np.stack(val_valid_ref_counts)
            val_distances = np.stack(val_distances)

            test_infos = common.chunked_parallel(test_dataset.spec_names,load_infos, max_cpu=64)
            test_specs, test_engs, test_root_morgans, test_adducts, test_instruments = [], [], [], [], []
            for test_info in test_infos:
                test_root_morgans.append(test_info[0])
                test_engs.append(test_info[1])
                test_adducts.append(test_info[2])
                test_specs.append(test_info[3])
                test_instruments.append(test_info[4])
            test_specs = sparse.csr_array(np.stack(test_specs))

            test_morgan_with_info = list(zip(test_root_morgans, test_adducts, test_engs, test_instruments))
            infos = common.chunked_parallel(test_morgan_with_info, find_ref, max_cpu=64)
            test_valid_ref_counts, test_closest, test_distances = [], [], []
            for info in infos:
                test_distances.append(info[0])
                test_valid_ref_counts.append(info[1])
                test_closest.append(info[2])
            pad_list.append(max((closest.shape[0] for closest in test_closest)))

            test_closest = np.stack(common.chunked_parallel(test_closest, pad, max_cpu=64))
            test_valid_ref_counts = np.stack(test_valid_ref_counts)
            test_distances = np.stack(test_distances)

            del train_dataset
            del val_dataset
            del test_dataset

    persistent_workers = kwargs["num_workers"] > 0
    mp_contex = 'spawn' if num_workers > 0 else None

    # Define dataloaders
    if save_ref:
        train_ref_df = pd.DataFrame({"valid_ref_count":train_valid_ref_counts, "distance":train_distances})
        test_ref_df = pd.DataFrame({"valid_ref_count":test_valid_ref_counts, "distance":test_distances})
        val_ref_df = pd.DataFrame({"valid_ref_count":val_valid_ref_counts, "distance":val_distances})
        if kwargs["debug"]:
            train_ref_df.to_csv(ref_dir/"train_debug.csv")
            test_ref_df.to_csv(ref_dir/"test_debug.csv")
            val_ref_df.to_csv(ref_dir/"val_debug.csv")
            np.save(ref_dir/"train_debug_ref.npy", train_closest)
            np.save(ref_dir/"test_debug_ref.npy", test_closest)
            np.save(ref_dir/"val_debug_ref.npy", val_closest)
            sparse.save_npz(ref_dir/"db_debug_specs.npz", db_specs)
            sparse.save_npz(ref_dir/"train_debug_specs.npz", train_specs)
            sparse.save_npz(ref_dir/"test_debug_specs.npz", test_specs)
            sparse.save_npz(ref_dir/"val_debug_specs.npz", val_specs)

            np.save(ref_dir/"db_debug_engs.npy", db_engs)
        else:    
            train_ref_df.to_csv(ref_dir/"train.csv")
            test_ref_df.to_csv(ref_dir/"test.csv")
            val_ref_df.to_csv(ref_dir/"val.csv")
            np.save(ref_dir/"train_ref.npy", train_closest)
            np.save(ref_dir/"test_ref.npy", test_closest)
            np.save(ref_dir/"val_ref.npy", val_closest)
            sparse.save_npz(ref_dir/"db_specs.npz", db_specs)
            sparse.save_npz(ref_dir/"train_specs.npz", train_specs)
            sparse.save_npz(ref_dir/"test_specs.npz", test_specs)
            sparse.save_npz(ref_dir/"val_specs.npz", val_specs)
            np.save(ref_dir/"db_engs.npy", db_engs)

    if database_set is not None:
        db_spec_names=database_set.spec_names
        del database_set
    else:
        db_spec_names=None
    
    try:
        if kwargs["debug"]:
            train_specs = sparse.load_npz(ref_dir / "train_debug_specs.npz")
            val_specs = sparse.load_npz(ref_dir / "val_debug_specs.npz")
            test_specs = sparse.load_npz(ref_dir / "test_debug_specs.npz")
        else:
            train_specs = sparse.load_npz(ref_dir / "train_specs.npz")
            val_specs = sparse.load_npz(ref_dir / "val_specs.npz")
            test_specs = sparse.load_npz(ref_dir / "test_specs.npz")
    except:
        train_specs=val_specs=test_specs=None

    max_ref_count = 1 if kwargs["single_ref"] else kwargs["max_ref_count"]
    train_dataset = dag_data.IntenDataset(
        train_df,
        magma_h5=magma_dag_folder,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
        ref_spec_names=db_spec_names,
        engs_db = db_engs,
        specs_db = db_specs,
        closest=train_closest,
        closest_distances = train_distances,
        valid_ref_count = train_valid_ref_counts,
        max_ref_count = max_ref_count,
        specs=train_specs,
    )
    val_dataset = dag_data.IntenDataset(
        val_df,
        magma_h5=magma_dag_folder,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
        ref_spec_names=db_spec_names,    
        engs_db = db_engs,
        specs_db = db_specs,
        closest=val_closest,
        closest_distances = val_distances,
        valid_ref_count = val_valid_ref_counts,
        max_ref_count = max_ref_count,
        specs=val_specs,
    )
    test_dataset = dag_data.IntenDataset(
        test_df,
        magma_h5=magma_dag_folder,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
        ref_spec_names=db_spec_names,
        engs_db = db_engs,
        specs_db = db_specs,
        closest=test_closest,
        closest_distances = test_distances,
        valid_ref_count = test_valid_ref_counts,
        max_ref_count = max_ref_count,
        specs=test_specs,
    )

    

    collate_fn = train_dataset.get_collate_fn()
    node_feats=train_dataset.get_node_feats()
    if  kwargs["filter"]:
        train_indices = np.arange(len(train_dataset))
        test_indices = np.arange(len(test_dataset))
        val_indices = np.arange(len(val_dataset))
        train_indices = train_indices[(train_distances < kwargs["mol_threshold"])]
        val_indices = val_indices[(val_distances < kwargs["mol_threshold"])]
        test_indices = test_indices[(test_distances < kwargs["mol_threshold"])]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    elif kwargs["filter_valid_test"]:
        test_indices = np.arange(len(test_dataset))
        val_indices = np.arange(len(val_dataset))
        val_indices = val_indices[(val_distances < kwargs["mol_threshold"])]
        test_indices = test_indices[(test_distances < kwargs["mol_threshold"])]
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)


    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
        pin_memory=kwargs["gpu"],
        multiprocessing_context=mp_contex,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
        pin_memory=kwargs["gpu"],
        multiprocessing_context=mp_contex,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
        pin_memory=kwargs["gpu"],
        multiprocessing_context=mp_contex,
    )
    # Define model
    model = inten_model.IntenGNN(
        hidden_size=kwargs["hidden_size"],
        mlp_layers=kwargs["mlp_layers"],
        gnn_layers=kwargs["gnn_layers"],
        set_layers=kwargs["set_layers"],
        frag_set_layers=kwargs["frag_set_layers"],
        inten_layers=kwargs["inten_layers"], 
        dropout=kwargs["dropout"],
        mpnn_type=kwargs["mpnn_type"],
        learning_rate=kwargs["learning_rate"],
        lr_decay_rate=kwargs["lr_decay_rate"],
        weight_decay=kwargs["weight_decay"],
        node_feats=node_feats,
        pe_embed_k=kwargs["pe_embed_k"],
        pool_op=kwargs["pool_op"],
        loss_fn=kwargs["loss_fn"],
        root_encode=kwargs["root_encode"],
        inject_early=kwargs["inject_early"],
        embed_adduct=kwargs["embed_adduct"],
        embed_collision=kwargs["embed_collision"],
        embed_instrument=kwargs["embed_instrument"],
        embed_elem_group=kwargs["embed_elem_group"],
        include_unshifted_mz=kwargs["include_unshifted_mz"],
        binned_targs=binned_targs,
        encode_forms=kwargs["encode_forms"],
        add_hs=add_hs,
        sk_tau=kwargs["sk_tau"],
        softmax_tau = kwargs["softmax_tau"],
        single_ref = kwargs["single_ref"],
        ppm_tol=kwargs["ppm_tol"],
        add_ref=add_ref,
        matching_method = kwargs["matching_method"],
        max_ref_count = kwargs["max_ref_count"],
        node_weight = kwargs["node_matching_weight"],
        multi_gnn = kwargs["multi_gnn"],
    )

    # Create trainer
    monitor = "val_loss"
    if kwargs["debug"]:
        kwargs["max_epochs"] = 2

    if kwargs["debug_overfit"]:
        kwargs["min_epochs"] = 1000
        kwargs["max_epochs"] = kwargs["min_epochs"]
        kwargs["no_monitor"] = True
        monitor = "train_loss"

    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    console_logger = common.ConsoleLogger()

    tb_path = tb_logger.log_dir
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=tb_path,
        filename="best",  # "{epoch}-{val_loss:.2f}",
        save_weights_only=False,
    )
    earlystop_callback = EarlyStopping(monitor=monitor, patience=5)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [earlystop_callback, checkpoint_callback, lr_monitor]

    trainer = pl.Trainer(
        logger=[tb_logger, console_logger],
        accelerator="gpu" if kwargs["gpu"] else "cpu",
        strategy='ddp',
        devices=torch.cuda.device_count() if kwargs["gpu"] else 0,
        callbacks=callbacks,
        gradient_clip_val=5,
        min_epochs=kwargs["min_epochs"],
        max_epochs=kwargs["max_epochs"],
        gradient_clip_algorithm="value",
        accumulate_grad_batches=kwargs["grad_accumulate"],
        num_sanity_val_steps=2 if kwargs["debug"] else 0,
    )

    if not kwargs["test_checkpoint"]:
        if kwargs["debug_overfit"]:
            trainer.fit(model, train_loader)
        else:
            trainer.fit(model, train_loader, val_loader)

        checkpoint_callback = trainer.checkpoint_callback
        test_checkpoint = checkpoint_callback.best_model_path
        test_checkpoint_score = checkpoint_callback.best_model_score.item()
    else:
        test_checkpoint = kwargs["test_checkpoint"]
        test_checkpoint_score = "[unknown]"

    # Load from checkpoint
    model = inten_model.IntenGNN.load_from_checkpoint(test_checkpoint)
    logging.info(
        f"Loaded model with from {test_checkpoint} with val loss of {test_checkpoint_score}"
    )

    model.eval()
    trainer.test(model=model, dataloaders=test_loader)

if __name__ == "__main__":
    import time
    torch.cuda.empty_cache()
    start_time = time.time()
    train_model()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")