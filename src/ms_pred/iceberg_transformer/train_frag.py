""" train_frag.py

PyTorch Lightning training script for the fragmentation-only model using the merged OneStepDataset.
Matches the structure and conveniences of train.py.
"""
import os
import logging
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader

import ms_pred.common as common
from ms_pred.iceberg_transformer.dataset import IntenDataset, TreeProcessor
from ms_pred.iceberg_transformer.frag_model import FragOnlyModel


def add_frag_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debug-overfit", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-epochs", default=100, type=int)
    parser.add_argument("--min-epochs", default=0, type=int)

    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_frag_only/")

    parser.add_argument("--dataset-name", default="nist20")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--magma-folder", default="magma_outputs")
    parser.add_argument("--split-name", default="split_1.tsv")

    parser.add_argument("--learning-rate", default=7e-4, type=float)
    parser.add_argument("--lr-decay-rate", default=1.0, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--test-checkpoint", default="", type=str)
    parser.add_argument("--embed-adduct", default=False, action="store_true")
    parser.add_argument("--embed-collision", default=False, action="store_true")
    parser.add_argument("--embed-elem-group", default=False, action="store_true")

    # Model params
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--root-encode", default="graphormer", choices=["gnn", "graphormer"], type=str)
    parser.add_argument("--add-hs", default=True, action="store_true")
    parser.add_argument("--pe-embed-k", default=10, type=int)
    parser.add_argument("--layers", default=6, type=int)
    parser.add_argument("--warmup", default=1000, type=int)
    parser.add_argument("--max-frags", default=100, type=int)
    parser.add_argument("--multi-hop-max-dist", default=5, type=int)
    parser.add_argument("--num-edge-dis", default=10, type=int)
    parser.add_argument("--sk-tau", default=0.05, type=float)
    parser.add_argument("--linsat-tau", default=0.01, type=float)

    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_frag_train_args(parser)
    return parser.parse_args()


def train_model():
    args = get_args()
    kwargs: Dict[str, Any] = args.__dict__

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="frag_train.log", debug=kwargs["debug"])
    pl.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Dataset
    dataset_name = kwargs["dataset_name"]
    data_dir = common.get_data_dir(dataset_name)
    labels = data_dir / kwargs["dataset_labels"]
    split_file = data_dir / "splits" / kwargs["split_name"]

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

    # Processor and datasets
    tree_processor = TreeProcessor(
        pe_embed_k=kwargs["pe_embed_k"],
        root_encode=kwargs["root_encode"],
        binned_targs=False,
        add_hs=kwargs["add_hs"],
        embed_elem_group=kwargs["embed_elem_group"],
        multi_hop_max_dist=kwargs["multi_hop_max_dist"],
    )

    train_dataset = IntenDataset(
        train_df,
        magma_h5=magma_h5_path,
        magma_map=name_to_json,
        num_workers=num_workers,
        root_encode=kwargs["root_encode"],
        binned_targs=False,
        add_hs=kwargs["add_hs"],
        embed_elem_group=kwargs["embed_elem_group"],
        tree_processor=tree_processor,
        datatype="HDF5",
    )
    val_dataset = IntenDataset(
        val_df,
        magma_h5=magma_h5_path,
        magma_map=name_to_json,
        num_workers=num_workers,
        root_encode=kwargs["root_encode"],
        binned_targs=False,
        add_hs=kwargs["add_hs"],
        embed_elem_group=kwargs["embed_elem_group"],
        tree_processor=tree_processor,
        datatype="HDF5",
    )
    test_dataset = IntenDataset(
        test_df,
        magma_h5=magma_h5_path,
        magma_map=name_to_json,
        num_workers=num_workers,
        root_encode=kwargs["root_encode"],
        binned_targs=False,
        add_hs=kwargs["add_hs"],
        embed_elem_group=kwargs["embed_elem_group"],
        tree_processor=tree_processor,
        datatype="HDF5",
    )

    # Dataloaders
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

    # Model
    model = FragOnlyModel(
        hidden_size=kwargs["hidden_size"],
        layers=kwargs["layers"],
        dropout=kwargs["dropout"],
        learning_rate=kwargs["learning_rate"],
        lr_decay_rate=kwargs["lr_decay_rate"],
        weight_decay=kwargs["weight_decay"],
        warmup=kwargs["warmup"],
        root_encode=kwargs["root_encode"],
        node_feats=train_dataset.get_node_feats(),
        edge_feats=tree_processor.get_edge_feats(),
        max_frags=kwargs["max_frags"],
        multi_hop_max_dist=kwargs["multi_hop_max_dist"],
        num_edge_dis=kwargs["num_edge_dis"],
        embed_adduct=kwargs["embed_adduct"],
        embed_collision=kwargs["embed_collision"],
        embed_elem_group=kwargs["embed_elem_group"],
        sk_tau=kwargs["sk_tau"],
        linsat_tau=kwargs["linsat_tau"],
    )

    # Trainer
    monitor = "val_loss"
    if kwargs["debug"]:
        kwargs["max_epochs"] = 5
    if kwargs["debug_overfit"]:
        kwargs["min_epochs"] = 2000
        kwargs["max_epochs"] = None
        monitor = "train_loss"

    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    console_logger = common.ConsoleLogger()
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=tb_logger.log_dir,
        filename="best",
        save_weights_only=False,
    )
    earlystop_callback = EarlyStopping(monitor=monitor, patience=5)
    callbacks = [earlystop_callback, checkpoint_callback]

    trainer = pl.Trainer(
        logger=[tb_logger, console_logger],
        accelerator="gpu" if kwargs["gpu"] else "cpu",
        devices=1 if kwargs["gpu"] else 0,
        callbacks=callbacks,
        gradient_clip_val=5,
        min_epochs=kwargs["min_epochs"],
        max_epochs=kwargs["max_epochs"],
        gradient_clip_algorithm="value",
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
    model = FragOnlyModel.load_from_checkpoint(test_checkpoint)
    logging.info(
        f"Loaded model from {test_checkpoint} with val loss of {test_checkpoint_score}"
    )
    model.eval()
    trainer.test(model=model, dataloaders=test_loader)


def main():
    train_model()


if __name__ == "__main__":
    import time
    start_time = time.time()
    train_model()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
