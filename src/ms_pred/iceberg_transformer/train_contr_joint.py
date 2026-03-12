""" train.py

PyTorch Lightning training script for the JointModel using the merged IntenDataset.

"""
import os
import logging
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import json

from ms_pred.iceberg_transformer import joint_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import ms_pred.common as common
from ms_pred.iceberg_transformer.dataset import IntenContrDataset, TreeProcessor
from ms_pred.iceberg_transformer.joint_model import JointModel
from pytorch_lightning.strategies import DDPStrategy
from rdkit import Chem





def add_joint_train_args(parser):
    """Add training arguments."""
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debug-overfit", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-epochs", default=200, type=int)
    parser.add_argument("--min-epochs", default=0, action="store", type=int)
    parser.add_argument("--ppm-tol", default=20, action="store", type=float)

    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_inten_pred/")

    parser.add_argument("--dataset-name", default="nist20")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--magma-folder", default="magma_outputs")
    parser.add_argument("--split-name", default="split_1.tsv")

    parser.add_argument("--learning-rate", default=0.000996, type=float)
    parser.add_argument("--lr-decay-rate", default=0.7214, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--test-checkpoint", default="", action="store", type=str)
    parser.add_argument("--train-checkpoint", default="", action="store", type=str)

    # Model params
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--graphormer-layers", default=6, type=int)
    parser.add_argument("--graphormer-dropout", default=0.05, type=float)
    parser.add_argument("--pe-embed-k", default=14, type=int)
    parser.add_argument("--inten-dropout", default=0.2, type=float)
    parser.add_argument("--frag-dropout", default=0.2, type=float)
    parser.add_argument("--embed-elem-group", default=True, action="store_true")
    parser.add_argument("--add-hs", default=True, action="store_true")
    parser.add_argument("--embed-adduct", default=True, action="store_true")
    parser.add_argument("--embed-collision", default=True, action="store_true")
    parser.add_argument("--embed-instrument", default=False, action="store_true")
    parser.add_argument("--encode-forms", default=True, action="store_true")
    parser.add_argument("--inten-decoder-layers", default=6, type=int)
    parser.add_argument("--inten-encoder-layers", default=6, type=int)
    parser.add_argument("--frag-encoder-layers", default=0, type=int)
    parser.add_argument("--frag-decoder-layers", default=3, type=int)
    parser.add_argument("--max-broken-bonds", default=6, type=int)
    parser.add_argument("--max-breakpoints", default=50, type=int)
    parser.add_argument("--multi-hop-max-dist", default=5, type=int)
    parser.add_argument("--num-edge-dis", default=10, type=int)
    parser.add_argument("--enable-aux-loss", default=False, action="store_true")
    parser.add_argument("--enable-decoder-norm", default=False, action="store_true")

    parser.add_argument("--sk-tau", default=0.05, type=float)
    parser.add_argument("--linsat-tau", default=0.01, type=float)
    
    # Other params
    parser.add_argument("--inten-weight", default=1, type=float)
    parser.add_argument("--frag-weight", default=0.1, type=float)
    parser.add_argument("--magma-warmup-steps", default=100, type=int)
    parser.add_argument("--magma-decay-rate", default=0.1, type=float)
    parser.add_argument("--magma-decay-steps", default=5000, type=int)

    parser.add_argument("--warmup", default=1000, action="store", type=int)
    parser.add_argument("--binned-targs", default=False, action="store_true")
    parser.add_argument(
        "--inten-loss-fn",
        default="cosine",
        action="store",
        choices=["cosine", "entropy", "weighted_entropy"],
    )
    parser.add_argument(
        "--contr-loss-fn",
        default="cosine",
        action="store",
        choices=["cosine", "entropy", "weighted_entropy"],
    )
    parser.add_argument(
        "--contr-weight", default=1, type=float
    )
    parser.add_argument(
        "--contr-threshold", default=0.5, type=float
    )
    parser.add_argument("--pubchem-map-path", default='data/pubchem/pubchem_formulae_inchikey.hdf5')
    parser.add_argument("--num-decoys", default=3, type=int)
    parser.add_argument("--num-fixed-decoys", default=7, action="store", type=int) 
    parser.add_argument("--all-decoy-nums", default=10, type=int)
    parser.add_argument("--grad-accumulate", default=1, type=int, action="store")

    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_joint_train_args(parser)
    return parser.parse_args()


def train_model():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name=f"joint_contr_finetune_decoy{kwargs['num_decoys']}_fixed_decoy{kwargs['num_fixed_decoys']}_contr_threshold_{kwargs['contr_threshold']}.log", debug=kwargs["debug"])
    pl.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    dataset_name = kwargs["dataset_name"]
    data_dir = common.get_data_dir(dataset_name)
    labels = data_dir / kwargs["dataset_labels"]
    split_file = data_dir / "splits" / kwargs["split_name"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    if kwargs["debug"]:
        df = df[:1000]

    spec_names = df["spec"].values
    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]


    magma_folder = kwargs["magma_folder"]
    num_workers = kwargs.get("num_workers", 0)
    magma_h5_path = data_dir / f"{magma_folder}/magma_tree_with_inten.hdf5"
    magma_h5 = common.HDF5Dataset(magma_h5_path)
    name_to_json = {Path(i).stem: i for i in magma_h5.get_all_names()}

    # Dataset parameters
    pe_embed_k = kwargs["pe_embed_k"]
    embed_elem_group = kwargs["embed_elem_group"]
    binned_targs = kwargs["binned_targs"]
    multi_hop_max_dist = kwargs["multi_hop_max_dist"]
    num_decoys = kwargs["num_decoys"]
    num_fixed_decoys = kwargs["num_fixed_decoys"]
    pubchem_path = kwargs["pubchem_map_path"]
    all_decoy_nums = kwargs["all_decoy_nums"]

    # Processor and datasets
    tree_processor = TreeProcessor(
        pe_embed_k=pe_embed_k,
        root_encode="graphormer",
        embed_elem_group=embed_elem_group,
        multi_hop_max_dist=kwargs["multi_hop_max_dist"],
    )

    train_dataset = IntenContrDataset(
        train_df,
        magma_h5=magma_h5_path,
        magma_map=name_to_json,
        num_workers=num_workers,
        root_encode="graphormer",
        embed_elem_group=embed_elem_group,
        tree_processor=tree_processor,
        datatype="HDF5",
        num_decoys=num_decoys,
        pubchem_path=pubchem_path,
        all_decoy_nums=all_decoy_nums,
    )
    val_dataset = IntenContrDataset(
        val_df,
        magma_h5=magma_h5_path,
        magma_map=name_to_json,
        num_workers=num_workers,
        root_encode="graphormer",
        embed_elem_group=embed_elem_group,
        tree_processor=tree_processor,
        datatype="HDF5",
        num_decoys=num_fixed_decoys,
        pubchem_path=pubchem_path,
        all_decoy_nums=all_decoy_nums,
        fix_decoys=True,  # Ensure val decoys are the same across epochs for consistent validation performance
    )
    test_dataset = IntenContrDataset(
        test_df,
        magma_h5=magma_h5_path,
        magma_map=name_to_json,
        num_workers=num_workers,
        root_encode="graphormer",
        embed_elem_group=embed_elem_group,
        tree_processor=tree_processor,
        datatype="HDF5",
        num_decoys=num_fixed_decoys,
        pubchem_path=pubchem_path,
        all_decoy_nums=all_decoy_nums,
        fix_decoys=True,  # Ensure test decoys are the same across epochs for consistent test performance
    )

    # Define dataloaders
    collate_fn = train_dataset.get_collate_fn()
    persistent_workers = kwargs["num_workers"] > 0
    mp_context = 'spawn' if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
        multiprocessing_context=mp_context,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
        multiprocessing_context=mp_context,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
        multiprocessing_context=mp_context,
    )

    # Define model
    model = JointModel(
        hidden_size = kwargs["hidden_size"],
        graphormer_layers=kwargs["graphormer_layers"],
        graphormer_dropout = kwargs["graphormer_dropout"],
        frag_decoder_layers=kwargs["frag_decoder_layers"],
        frag_encoder_layers=kwargs["frag_encoder_layers"],
        frag_dropout=kwargs["frag_dropout"],
        node_feats = train_dataset.get_node_feats(),
        edge_feats = tree_processor.get_edge_feats(),
        multi_hop_max_dist=multi_hop_max_dist,
        num_edge_dis=kwargs["num_edge_dis"],
        max_breakpoints=kwargs["max_breakpoints"],
        embed_adduct=kwargs["embed_adduct"],
        embed_collision=kwargs["embed_collision"],
        embed_elem_group=embed_elem_group,
        embed_instrument=kwargs["embed_instrument"],
        encode_forms=kwargs["encode_forms"],
        linsat_tau=kwargs["linsat_tau"],
        max_broken_bonds=kwargs["max_broken_bonds"],
        pe_embed_k=pe_embed_k,
        enable_aux_loss=kwargs["enable_aux_loss"],
        enable_decoder_norm=kwargs["enable_decoder_norm"],
        inten_decoder_layers=kwargs["inten_decoder_layers"],
        inten_encoder_layers=kwargs["inten_encoder_layers"],
        inten_dropout=kwargs["inten_dropout"],
        inten_loss_fn=kwargs["inten_loss_fn"],
        binned_targs=binned_targs,
        sk_tau=kwargs["sk_tau"],
        ppm_tol=kwargs["ppm_tol"],
        contr_weight=kwargs["contr_weight"],
        contr_threshold=kwargs["contr_threshold"],
        contr_loss_fn=kwargs["contr_loss_fn"],
        inten_weight=kwargs["inten_weight"],
        frag_weight = kwargs["frag_weight"],
        magma_warmup_steps=kwargs["magma_warmup_steps"],
        magma_decay_rate=kwargs["magma_decay_rate"],
        magma_decay_steps=kwargs["magma_decay_steps"],
        lr=kwargs["learning_rate"],
        lr_decay_rate=kwargs["lr_decay_rate"],
        weight_decay=kwargs["weight_decay"],
        warmup=kwargs["warmup"]
    )

    # Create trainer
    monitor = "val_loss"
    if kwargs["debug"]:
        kwargs["max_epochs"] = 5

    if kwargs["debug_overfit"]:
        kwargs["min_epochs"] = 2000
        kwargs["max_epochs"] = None
        monitor = "train_loss"

    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    console_logger = common.ConsoleLogger()

    tb_path = tb_logger.log_dir
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=tb_path,
        filename="best",
        save_weights_only=False,
    )
    earlystop_callback = EarlyStopping(monitor=monitor, patience=3)
    callbacks = [earlystop_callback, checkpoint_callback]

    trainer = pl.Trainer(
        logger=[tb_logger, console_logger],
        accelerator="gpu" if kwargs["gpu"] else "cpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        devices=torch.cuda.device_count() if kwargs["gpu"] else 0,
        callbacks=callbacks,
        gradient_clip_val=5,
        min_epochs=kwargs["min_epochs"],
        max_epochs=kwargs["max_epochs"],
        gradient_clip_algorithm="value",
        accumulate_grad_batches=kwargs["grad_accumulate"],
        num_sanity_val_steps=2 if kwargs["debug"] else 0,
    )

    if kwargs["train_checkpoint"]:
        train_checkpoint = kwargs["train_checkpoint"]
        model = joint_model.JointModel.load_from_checkpoint(train_checkpoint)
        logging.info(
            f"Loaded model with from {train_checkpoint}"
        )
        # Force contrastive learning and magma params
        model.sk_tau = kwargs["sk_tau"]
        model.contr_weight = kwargs["contr_weight"]
        model.inten_weight = kwargs["inten_weight"]
        model.frag_weight = kwargs["frag_weight"]
        model.contr_threshold = kwargs["contr_threshold"]
        model.magma_warmup_steps = kwargs["magma_warmup_steps"]
        model.magma_decay_rate = kwargs["magma_decay_rate"]
        model.magma_decay_steps = kwargs["magma_decay_steps"]

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
    model = JointModel.load_from_checkpoint(test_checkpoint)
    logging.info(
        f"Loaded model from {test_checkpoint} with val loss of {test_checkpoint_score}"
    )

    model.eval()
    trainer.test(model=model, dataloaders=test_loader)


def main():
    """Main function for backwards compatibility."""
    train_model()


if __name__ == "__main__":
    import time

    start_time = time.time()
    train_model()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
