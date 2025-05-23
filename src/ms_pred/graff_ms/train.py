"""train.py

Train graff ms to predict binned specs

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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import ms_pred.common as common
from ms_pred.graff_ms import graff_ms_data, graff_ms_model
import ms_pred.nn_utils as nn_utils


def add_graff_ms_train_args(parser):
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debug-overfit", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=128, action="store", type=int)
    parser.add_argument("--max-epochs", default=100, action="store", type=int)
    parser.add_argument("--min-epochs", default=0, action="store", type=int)

    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_gnn/")

    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")

    parser.add_argument("--learning-rate", default=4e-7, action="store", type=float)
    parser.add_argument("--lr-decay-rate", default=1.0, action="store", type=float)
    parser.add_argument("--weight-decay", default=0, action="store", type=float)

    parser.add_argument("--num-bins", default=15000, action="store", type=int)
    parser.add_argument("--layers", default=3, action="store", type=int)
    parser.add_argument("--set-layers", default=2, action="store", type=int)
    parser.add_argument("--pe-embed-k", default=0, action="store", type=int)
    parser.add_argument("--pool-op", default="avg", action="store")
    parser.add_argument(
        "--mpnn-type", default="GGNN", action="store", choices=["GGNN", "PNA", "GINE"]
    )
    parser.add_argument(
        "--loss-fn",
        default="cosine",
        action="store",
        choices=["mse", "hurdle", "cosine"],
    )
    parser.add_argument("--dropout", default=0.1, action="store", type=float)
    parser.add_argument("--hidden-size", default=256, action="store", type=int)
    parser.add_argument(
        "--form-dir-name", default="magma_subform_50_with_raw", action="store"
    )
    parser.add_argument("--embed-adduct", default=False, action="store_true")
    parser.add_argument("--embed-collision-energy", default=False, action="store_true")
    parser.add_argument("--num-fixed-forms", default=10000, action="store", type=int)
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_graff_ms_train_args(parser)
    return parser.parse_args()


def train_model():
    args = get_args()
    kwargs = args.__dict__
    upper_limit = 1500

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="graff_train.log", debug=kwargs["debug"])
    pl.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = Path("data/spec_datasets") / dataset_name
    labels = data_dir / kwargs["dataset_labels"]
    split_file = data_dir / "splits" / kwargs["split_name"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    if args.debug:
        df = df[:100]

    spec_names = df["spec"].values

    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]

    subform_stem = kwargs["form_dir_name"]
    subformula_h5_path = Path(data_dir) / "subformulae" / subform_stem
    subformula_h5 = common.HDF5Dataset(subformula_h5_path)
    form_map = {Path(i).stem: i for i in subformula_h5.get_all_names()}
    graph_featurizer = nn_utils.MolDGLGraph(pe_embed_k=kwargs["pe_embed_k"])
    atom_feats = graph_featurizer.atom_feats
    bond_feats = graph_featurizer.bond_feats

    num_bins = kwargs.get("num_bins")
    num_workers = kwargs.get("num_workers", 0)
    train_dataset = graff_ms_data.BinnedDataset(
        train_df,
        form_map=form_map,
        form_h5=subformula_h5_path,
        data_dir=data_dir,
        num_bins=num_bins,
        # num_workers=num_workers,
        upper_limit=upper_limit,
        graph_featurizer=graph_featurizer,
    )
    val_dataset = graff_ms_data.BinnedDataset(
        val_df,
        form_map=form_map,
        form_h5=subformula_h5_path,
        data_dir=data_dir,
        num_bins=num_bins,
        # num_workers=num_workers,
        upper_limit=upper_limit,
        graph_featurizer=graph_featurizer,
    )
    test_dataset = graff_ms_data.BinnedDataset(
        test_df,
        form_map=form_map,
        form_h5=subformula_h5_path,
        data_dir=data_dir,
        num_bins=num_bins,
        # num_workers=num_workers,
        upper_limit=upper_limit,
        graph_featurizer=graph_featurizer,
    )
    new_entry = train_dataset[0]

    # Losses will be negatives and others will be positives
    num_fixed_forms = kwargs["num_fixed_forms"]
    top_forms = train_dataset.get_top_forms()
    logging.info(f"Found {top_forms['forms'].shape[0]} formulae")
    num_fixed_forms = min(num_fixed_forms, len(top_forms["forms"]))
    fixed_forms = top_forms["forms"][:num_fixed_forms]
    logging.info(f"Selected {fixed_forms.shape[0]} formulae")

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
    test_loader = DataLoader(
        test_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    # Define model
    # test_batch = next(iter(train_loader))
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
        embed_collision_energy=kwargs["embed_collision_energy"],
        num_fixed_forms=num_fixed_forms,
    )
    model.set_fixed_forms(fixed_forms)

    # outputs = model(test_batch['fps'])
    monitor = "val_loss"
    if kwargs["debug"]:
        kwargs["max_epochs"] = 2

    if kwargs["debug_overfit"]:
        kwargs["min_epochs"] = 2000
        kwargs["max_epochs"] = None
        kwargs["no_monitor"] = True
        kwargs["warmup"] = 0
        monitor = "train_loss"

    # Create trainer
    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    console_logger = common.ConsoleLogger()

    tb_path = tb_logger.log_dir
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=tb_path,
        filename="best",
        save_weights_only=False,
    )
    earlystop_callback = EarlyStopping(monitor=monitor, patience=20)
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
    )

    if kwargs["debug_overfit"]:
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, train_loader, val_loader)

    checkpoint_callback = trainer.checkpoint_callback
    best_checkpoint = checkpoint_callback.best_model_path
    best_checkpoint_score = checkpoint_callback.best_model_score.item()

    # Load from checkpoint
    model = graff_ms_model.GraffGNN.load_from_checkpoint(best_checkpoint)
    logging.info(
        f"Loaded model with from {best_checkpoint} with val loss of {best_checkpoint_score}"
    )
    model.eval()
    trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    import time

    start_time = time.time()
    train_model()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
