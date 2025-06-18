"""inten_hyperopt.py

Hyperopt parameters for frag tree generation model

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
import ms_pred.marason.dag_data as dag_data
import ms_pred.marason.inten_model as inten_model
import ms_pred.marason.train_inten as inten_train


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
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = common.get_data_dir(dataset_name)
    labels = data_dir / kwargs["dataset_labels"]
    split_file = data_dir / "splits" / kwargs["split_name"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    spec_names = df["spec"].values
    if kwargs["debug_overfit"]:
        train_inds, val_inds, test_inds = common.get_splits(
            spec_names, split_file, val_frac=0
        )
        # train_inds = train_inds[:6]
    else:
        train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]

    num_workers = kwargs.get("num_workers", 0)
    magma_dag_folder = Path(kwargs["magma_dag_folder"])
    all_json_pths = [Path(i) for i in magma_dag_folder.glob("*.json")]
    name_to_json = {i.stem.replace("pred_", ""): i for i in all_json_pths}

    pe_embed_k = kwargs["pe_embed_k"]
    root_encode = kwargs["root_encode"]
    binned_targs = kwargs["binned_targs"]
    tree_processor = dag_data.TreeProcessor(
        pe_embed_k=pe_embed_k, root_encode=root_encode, binned_targs=binned_targs
    )

    # Build out frag datasets
    train_dataset = dag_data.IntenDataset(
        train_df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
        use_ray=True,
    )
    val_dataset = dag_data.IntenDataset(
        val_df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
        use_ray=True,
    )

    persistent_workers = kwargs["num_workers"] > 0

    # Define dataloaders
    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        num_workers=min(kwargs["num_workers"], kwargs["batch_size"]),
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=min(kwargs["num_workers"], kwargs["batch_size"]),
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
    )

    # Define model
    model = inten_model.IntenGNN(
        hidden_size=kwargs["hidden_size"],
        mlp_layers=kwargs["mlp_layers"],
        gnn_layers=kwargs["gnn_layers"],
        set_layers=kwargs["set_layers"],
        frag_set_layers=kwargs["frag_set_layers"],
        dropout=kwargs["dropout"],
        mpnn_type=kwargs["mpnn_type"],
        learning_rate=kwargs["learning_rate"],
        lr_decay_rate=kwargs["lr_decay_rate"],
        weight_decay=kwargs["weight_decay"],
        node_feats=train_dataset.get_node_feats(),
        pe_embed_k=kwargs["pe_embed_k"],
        pool_op=kwargs["pool_op"],
        loss_fn=kwargs["loss_fn"],
        root_encode=kwargs["root_encode"],
        inject_early=kwargs["inject_early"],
        embed_adduct=kwargs["embed_adduct"],
        binned_targs=binned_targs,
        encode_forms=kwargs["encode_forms"],
    )

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
    inten_train.add_frag_train_args(parser)
    nn_utils.add_hyperopt_args(parser)
    return parser.parse_args()


def get_param_space(trial):
    """get_param_space.

    Use optuna to define this dynamically

    """
    trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    trial.suggest_float("lr_decay_rate", 0.7, 1.0, log=True)
    trial.suggest_categorical("weight_decay", [1e-6, 1e-7, 0])
    trial.suggest_int("gnn_layers", 1, 6)
    trial.suggest_int("mlp_layers", 0, 3)
    trial.suggest_int("set_layers", 0, 0)
    trial.suggest_int("frag_set_layers", 0, 3)

    trial.suggest_int("pe_embed_k", 0, 0)
    trial.suggest_float("dropout", 0, 0.3, step=0.1)

    trial.suggest_categorical("hidden_size", [128, 256, 512])
    trial.suggest_categorical("batch_size", [8, 16, 32])

    # trial.suggest_categorical("mpnn_type", ["GINE"])
    trial.suggest_categorical("mpnn_type", ["GGNN"])
    trial.suggest_categorical("pool_op", ["avg", "attn"])
    trial.suggest_categorical("embed_adduct", [True])
    trial.suggest_categorical("binned_target", [True])


def get_initial_points() -> List[Dict]:
    """get_intiial_points.

    Create dictionaries defining initial configurations to test

    """
    init_base = {
        "learning_rate": 0.0002,
        "lr_decay_rate": 0.85,
        "weight_decay": 1.0e-7,
        "dropout": 0.1,
        "gnn_layers": 5,
        "set_layers": 0,
        "frag_set_layers": 2,
        "mlp_layers": 2,
        "batch_size": 32,
        "pe_embed_k": 0,
        "hidden_size": 512,
        "mpnn_type": "GGNN",
        "pool_op": "attn",
        "embed_adduct": True,
        "binned_target": True,
    }
    return [init_base]


def run_hyperopt():
    """run_hyperopt."""
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
