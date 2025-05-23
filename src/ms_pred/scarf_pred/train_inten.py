"""train_inten.py

Train model to predict emit intensities for each mol formula

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
import ms_pred.scarf_pred.scarf_data as scarf_data
import ms_pred.scarf_pred.scarf_model as scarf_model

import ms_pred.nn_utils as nn_utils


def add_inten_train_args(parser):
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
        "--formula-folder", default="subform_50", help="stem of formula folder"
    )
    parser.add_argument("--split-name", default="split_22.tsv")

    parser.add_argument("--batch-size", default=3, action="store", type=int)
    parser.add_argument("--max-epochs", default=100, action="store", type=int)
    parser.add_argument("--min-epochs", default=0, action="store", type=int)
    parser.add_argument("--learning-rate", default=7e-4, action="store", type=float)
    parser.add_argument("--lr-decay-rate", default=1.0, action="store", type=float)
    parser.add_argument("--weight-decay", default=0, action="store", type=float)
    parser.add_argument(
        "--loss-fn", default="cosine", action="store", choices=["cosine"]
    )
    parser.add_argument("--gnn-layers", default=5, action="store", type=int)
    parser.add_argument("--mlp-layers", default=2, action="store", type=int)
    parser.add_argument("--set-layers", default=1, action="store", type=int)
    parser.add_argument("--form-set-layers", default=1, action="store", type=int)
    parser.add_argument("--dropout", default=0, action="store", type=float)
    parser.add_argument("--hidden-size", default=256, action="store", type=int)
    parser.add_argument("--pe-embed-k", default=20, action="store", type=int)
    parser.add_argument("--pool-op", default="avg", action="store")
    parser.add_argument(
        "--mpnn-type", default="GGNN", action="store", choices=["GGNN", "PNA", "GINE"]
    )
    parser.add_argument("--embedder", default="abs-sines", type=str)
    parser.add_argument("--info-join", default="concat", action="store")
    parser.add_argument("--root-embedder", default="gnn", type=str)
    parser.add_argument("--binned-targs", default=False, action="store_true")
    parser.add_argument("--embed-adduct", default=False, action="store_true")

    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_inten_train_args(parser)
    return parser.parse_args()


def train_model():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="dag_inten_train.log", debug=kwargs["debug"])
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

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    if kwargs["debug"]:
        df = df[:100]
        kwargs["num_workers"] = 0

    spec_names = df["spec"].values
    if kwargs["debug_overfit"]:
        kwargs["warmup"] = 0
        train_inds, val_inds, test_inds = common.get_splits(
            spec_names, split_file, val_frac=0
        )
        train_inds = train_inds[:100]
    else:
        train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]

    num_workers = kwargs.get("num_workers", 0)
    form_dag_folder = Path(kwargs["formula_folder"])
    all_json_pths = [Path(i) for i in form_dag_folder.glob("*.json")]
    name_to_json = {i.stem.replace("pred_", ""): i for i in all_json_pths}
    graph_featurizer = nn_utils.MolDGLGraph(pe_embed_k=kwargs["pe_embed_k"])
    atom_feats = graph_featurizer.atom_feats
    bond_feats = graph_featurizer.bond_feats

    # Build out frag datasets
    train_dataset = scarf_data.IntenDataset(
        train_df,
        data_dir=data_dir,
        graph_featurizer=graph_featurizer,
        form_map=name_to_json,
        num_workers=num_workers,
        root_embedder=kwargs["root_embedder"],
        binned_targs=kwargs["binned_targs"],
    )
    val_dataset = scarf_data.IntenDataset(
        val_df,
        data_dir=data_dir,
        graph_featurizer=graph_featurizer,
        form_map=name_to_json,
        num_workers=num_workers,
        root_embedder=kwargs["root_embedder"],
        binned_targs=kwargs["binned_targs"],
    )

    test_dataset = scarf_data.IntenDataset(
        test_df,
        data_dir=data_dir,
        graph_featurizer=graph_featurizer,
        form_map=name_to_json,
        num_workers=num_workers,
        root_embedder=kwargs["root_embedder"],
        binned_targs=kwargs["binned_targs"],
    )
    ex = train_dataset[0]

    persistent_workers = kwargs["num_workers"] > 0

    # Define dataloaders
    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
    )

    # Define model
    model = scarf_model.ScarfIntenNet(
        hidden_size=kwargs["hidden_size"],
        gnn_layers=kwargs["gnn_layers"],
        mlp_layers=kwargs["mlp_layers"],
        set_layers=kwargs["set_layers"],
        form_set_layers=kwargs["form_set_layers"],
        formula_dim=common.NORM_VEC.shape[0],
        mpnn_type=kwargs["mpnn_type"],
        dropout=kwargs["dropout"],
        learning_rate=kwargs["learning_rate"],
        weight_decay=kwargs["weight_decay"],
        atom_feats=atom_feats,
        bond_feats=bond_feats,
        pe_embed_k=kwargs["pe_embed_k"],
        pool_op=kwargs["pool_op"],
        num_atom_feats=graph_featurizer.num_atom_feats,
        num_bond_feats=graph_featurizer.num_bond_feats,
        lr_decay_rate=kwargs["lr_decay_rate"],
        loss_fn=kwargs["loss_fn"],
        warmup=kwargs.get("warmup", 1000),
        info_join=kwargs["info_join"],
        root_embedder=kwargs["root_embedder"],
        embedder=kwargs["embedder"],
        binned_targs=kwargs["binned_targs"],
        embed_adduct=kwargs["embed_adduct"],
    )

    # test_batch = next(iter(train_loader))
    # outputs = model(
    #    graphs=test_batch['graphs'],
    #    formulae=test_batch['formulae'],
    #    diffs=test_batch['diffs'],
    #    num_forms=test_batch['num_forms'],
    # )

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
    earlystop_callback = EarlyStopping(monitor=monitor, patience=20)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [earlystop_callback, checkpoint_callback, lr_monitor]

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
    model = scarf_model.ScarfIntenNet.load_from_checkpoint(best_checkpoint)
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
