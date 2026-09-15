"""graff_ms_hyperopt.py

Hyperopt parameters for graff ms model

"""
import os
import copy
import logging
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Dict

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ray import tune

import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
import ms_pred.graff_ms.train as graff_ms_train
import ms_pred.graff_ms.graff_ms_data as graff_ms_data
import ms_pred.graff_ms.graff_ms_model as graff_ms_model


def score_function(config, base_args, orig_dir=""):
    """score_function.

    Args:
        config: All configs passed by hyperoptimizer
        base_args: Base arguments
        orig_dir: ""
    """
    # tunedir = tune.get_trial_dir()
    # Switch s.t. we can use relative data structures
    os.chdir(orig_dir)

    kwargs = copy.deepcopy(base_args)
    kwargs.update(config)
    pl.seed_everything(kwargs.get("seed"))

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = Path("data/spec_datasets") / dataset_name
    labels = data_dir / kwargs["dataset_labels"]
    split_file = data_dir / "splits" / kwargs["split_name"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    spec_names = df["spec"].values

    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]

    num_bins = kwargs.get("num_bins")
    num_workers = kwargs.get("num_workers", 0)
    num_workers = 0
    upper_limit = kwargs.get("upper_limit", 1500)

    graph_featurizer = nn_utils.MolDGLGraph(pe_embed_k=kwargs["pe_embed_k"])
    atom_feats = graph_featurizer.atom_feats
    bond_feats = graph_featurizer.bond_feats

    subform_stem = kwargs["form_dir_name"]
    subformula_folder = Path(data_dir) / "subformulae" / subform_stem
    form_map = {i.stem: Path(i) for i in subformula_folder.glob("*.json")}
    graph_featurizer = nn_utils.MolDGLGraph(pe_embed_k=kwargs["pe_embed_k"])
    atom_feats = graph_featurizer.atom_feats
    bond_feats = graph_featurizer.bond_feats
    train_dataset = graff_ms_data.BinnedDataset(
        train_df,
        form_map=form_map,
        data_dir=data_dir,
        num_bins=num_bins,
        # num_workers=num_workers,
        upper_limit=upper_limit,
        graph_featurizer=graph_featurizer,
    )
    val_dataset = graff_ms_data.BinnedDataset(
        val_df,
        form_map=form_map,
        data_dir=data_dir,
        num_bins=num_bins,
        # num_workers=num_workers,
        upper_limit=upper_limit,
        graph_featurizer=graph_featurizer,
    )
    logging.info("Constructing form dict")
    # Losses will be negatives and others will be positives
    num_fixed_forms = kwargs["num_fixed_forms"]
    top_forms = train_dataset.get_top_forms()
    num_fixed_forms = min(num_fixed_forms, len(top_forms["forms"]))
    fixed_forms = top_forms["forms"][:num_fixed_forms]
    logging.info("Done constructing form dict")

    # Define dataloaders
    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=kwargs["batch_size"],
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    # Define model
    # test_batch = next(iter(train_loader))
    logging.info("Building model")
    model = graff_ms_model.GraffGNN(
        hidden_size=kwargs["hidden_size"],
        layers=kwargs["layers"],
        set_layers=kwargs["set_layers"],
        mpnn_type=kwargs["mpnn_type"],
        dropout=kwargs["dropout"],
        output_dim=num_bins,
        learning_rate=kwargs["learning_rate"],
        weight_decay=kwargs["weight_decay"],
        upper_limit=upper_limit,
        loss_fn=kwargs["loss_fn"],
        atom_feats=atom_feats,
        bond_feats=bond_feats,
        pe_embed_k=kwargs["pe_embed_k"],
        pool_op=kwargs["pool_op"],
        num_atom_feats=graph_featurizer.num_atom_feats,
        num_bond_feats=graph_featurizer.num_bond_feats,
        lr_decay_rate=kwargs["lr_decay_rate"],
        embed_adduct=kwargs["embed_adduct"],
        num_fixed_forms=num_fixed_forms,
    )
    model.set_fixed_forms(fixed_forms)

    # outputs = model(test_batch['fps'])
    # Create trainer
    tb_logger = pl_loggers.TensorBoardLogger(tune.get_trial_dir(), "", ".")

    # Replace with custom callback that utilizes maximum loss during train
    tune_callback = nn_utils.TuneReportCallback(["val_loss"])

    val_check_interval = None
    check_val_every_n_epoch = 1
    monitor = "val_loss"

    # tb_path = tb_logger.log_dir
    earlystop_callback = EarlyStopping(monitor=monitor, patience=10)
    callbacks = [earlystop_callback, tune_callback]
    logging.info("Starting train")
    trainer = pl.Trainer(
        logger=[tb_logger],
        accelerator="gpu" if kwargs["gpu"] else None,
        devices=1 if kwargs["gpu"] else None,
        callbacks=callbacks,
        gradient_clip_val=5,
        max_epochs=kwargs["max_epochs"],
        gradient_clip_algorithm="value",
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader, val_loader)


def get_args():
    parser = argparse.ArgumentParser()
    graff_ms_train.add_graff_ms_train_args(parser)
    nn_utils.add_hyperopt_args(parser)
    return parser.parse_args()


def get_param_space(trial):
    """get_param_space.

    Use optuna to define this dynamically

    """
    trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    trial.suggest_float("lr_decay_rate", 0.7, 1.0, log=True)
    trial.suggest_categorical("weight_decay", [1e-6, 1e-7, 0])
    trial.suggest_categorical("batch_size", [16, 32, 64])
    trial.suggest_int("pe_embed_k", 0, 20)

    trial.suggest_categorical("use_reverse", [False])
    trial.suggest_float("dropout", 0, 0.3, step=0.1)
    trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    trial.suggest_int("layers", 1, 6)

    # trial.suggest_int("set_layers", 0, 3)
    trial.suggest_int("set_layers", 0, 0)

    trial.suggest_categorical("pool_op", ["avg", "attn"])
    trial.suggest_categorical("mpnn_type", ["GGNN"])
    trial.suggest_categorical("embed_adduct", [True])
    trial.suggest_categorical("num_fixed_forms", [1000, 5000, 10000, 25000, 50000])


def get_initial_points() -> List[Dict]:
    """get_intiial_points.

    Create dictionaries defining initial configurations to test

    """
    init_base = {
        "learning_rate": 0.00086,
        "lr_decay_rate": 0.95,
        "batch_size": 64,
        "pe_embed_k": 15,
        "weight_decay": 0,
        "dropout": 0.2,
        "hidden_size": 64,
        "pool_op": "avg",
        "layers": 3,
        # "set_layers": 2,
        "set_layers": 0,
        "use_reverse": True,
        "mpnn_type": "GGNN",  # "GINE",
        "embed_adduct": True,
        "num_fixed_forms": 10000,
    }
    return [init_base]


def run_hyperopt():
    args = get_args()
    kwargs = args.__dict__
    nn_utils.run_hyperopt(
        kwargs=kwargs,
        score_function=score_function,
        param_space_function=get_param_space,
        initial_points=get_initial_points(),
    )


if __name__ == "__main__":
    import time

    start_time = time.time()
    run_hyperopt()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
