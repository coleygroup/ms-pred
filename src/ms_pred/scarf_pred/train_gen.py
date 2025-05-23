"""train.py

Train scarf model to predict binned specs

"""
import logging
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import ms_pred.common as common
from ms_pred.scarf_pred import scarf_data, scarf_model
import ms_pred.nn_utils as nn_utils


def add_scarf_train_args(parser):
    """add_scarf_train_args.

    Args:
        parser:
    """
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--use-reverse", default=False, action="store_true")
    parser.add_argument("--no-forward", default=False, action="store_true")
    parser.add_argument("--debug-overfit", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=128, action="store", type=int)
    parser.add_argument("--max-epochs", default=100, action="store", type=int)
    parser.add_argument("--min-epochs", default=0, action="store", type=int)

    parser.add_argument(
        "--formula-folder", default="subform_20", help="stem of formula folder"
    )

    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_scarf_gen/")

    parser.add_argument("--dataset-name", default="canopus_train_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_1.tsv")

    parser.add_argument("--learning-rate", default=7e-4, action="store", type=float)
    parser.add_argument("--lr-decay-rate", default=1.0, action="store", type=float)
    parser.add_argument("--weight-decay", default=0, action="store", type=float)

    # Fix model params
    parser.add_argument("--gnn-layers", default=5, action="store", type=int)
    parser.add_argument("--mlp-layers", default=2, action="store", type=int)
    parser.add_argument("--set-layers", default=1, action="store", type=int)
    parser.add_argument("--dropout", default=0, action="store", type=float)
    parser.add_argument("--hidden-size", default=256, action="store", type=int)
    parser.add_argument("--pe-embed-k", default=20, action="store", type=int)
    parser.add_argument("--pool-op", default="avg", action="store")
    parser.add_argument(
        "--mpnn-type", default="GGNN", action="store", choices=["GGNN", "PNA", "GINE"]
    )
    parser.add_argument("--loss-fn", default="cosine", action="store", choices=["bce"])
    parser.add_argument("--embedder", default="abs-sines", type=str)
    parser.add_argument("--use-tbc", default=False, action="store_true")
    parser.add_argument("--info-join", default="concat", action="store")
    parser.add_argument("--root-embedder", default="gnn", type=str)
    parser.add_argument("--embed-adduct", default=False, action="store_true")
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_scarf_train_args(parser)
    return parser.parse_args()


def train_model():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="scarf_train.log", debug=kwargs["debug"])
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
        train_inds, val_inds, test_inds = common.get_splits(
            spec_names, split_file, val_frac=0
        )

        # Test specific debug overfit
        # Get 2
        interest_ind = np.argwhere("CCMSLIB00000001568" == spec_names).flatten()[0]

        # train_inds = np.array([interest_ind], dtype=np.int64)
        val_inds = np.array([1])
        test_inds = np.array([1])

        train_inds = train_inds[:1]
        kwargs["warmup"] = 0
    else:
        train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)

    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]

    num_workers = kwargs.get("num_workers", 0)
    subform_stem = kwargs.get("formula_folder", 0)
    subformula_folder = data_dir / "subformulae" / subform_stem
    subform_map = {i.stem: Path(i) for i in subformula_folder.glob("*.json")}
    graph_featurizer = nn_utils.MolDGLGraph(pe_embed_k=kwargs["pe_embed_k"])
    atom_feats = graph_featurizer.atom_feats
    bond_feats = graph_featurizer.bond_feats

    train_dataset = scarf_data.ScarfDataset(
        df=train_df,
        data_dir=data_dir,
        file_map=subform_map,
        graph_featurizer=graph_featurizer,
        use_ray=False,
        root_embedder=kwargs["root_embedder"],
        num_workers=num_workers,
    )

    val_dataset = scarf_data.ScarfDataset(
        df=val_df,
        data_dir=data_dir,
        graph_featurizer=graph_featurizer,
        file_map=subform_map,
        use_ray=False,
        root_embedder=kwargs["root_embedder"],
        num_workers=num_workers,
    )

    test_dataset = scarf_data.ScarfDataset(
        df=test_df,
        data_dir=data_dir,
        graph_featurizer=graph_featurizer,
        root_embedder=kwargs["root_embedder"],
        file_map=subform_map,
        use_ray=False,
        num_workers=num_workers,
    )

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
    # graphs, formula_tensor, options, option_len,
    # dict_keys(['names', 'graphs', 'formula_tensors', 'options', 'option_len',
    #           'targ_inten', 'atom_inds', 'num_atom_types', 'mol_inds'])
    test_batch = next(iter(train_loader))

    model = scarf_model.ScarfNet(
        hidden_size=kwargs["hidden_size"],
        gnn_layers=kwargs["gnn_layers"],
        mlp_layers=kwargs["mlp_layers"],
        set_layers=kwargs["set_layers"],
        use_reverse=kwargs["use_reverse"],
        no_forward=kwargs["no_forward"],
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
        embedder=kwargs.get("embedder"),
        use_tbc=kwargs.get("use_tbc"),
        info_join=kwargs.get("info_join"),
        root_embedder=kwargs["root_embedder"],
        embed_adduct=kwargs["embed_adduct"],
    )

    # Debug
    # outputs = model(test_batch['graphs'],
    #                test_batch['formula_tensors'].float(),
    #                test_batch['options'].float(),
    #                test_batch['diffs'].float(),
    #                test_batch['option_len'],
    #                test_batch['mol_inds'].long(),
    #                test_batch['atom_inds'].float())

    # Create trainer
    monitor = "val_loss"
    if kwargs["debug"]:
        kwargs["max_epochs"] = 2

    if kwargs["debug_overfit"]:
        kwargs["min_epochs"] = 2000
        kwargs["max_epochs"] = None
        kwargs["no_monitor"] = True
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
    model = scarf_model.ScarfNet.load_from_checkpoint(best_checkpoint)
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
