"""train.py

Train model to predict DAG breakages

"""
import logging
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import ms_pred.common as common
from ms_pred.dag_pred import dag_data, gen_model


def add_frag_train_args(parser):
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debug-overfit", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=128, action="store", type=int)
    parser.add_argument("--max-epochs", default=100, action="store", type=int)
    parser.add_argument("--min-epochs", default=0, action="store", type=int)

    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_tree_pred/")

    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument(
        "--magma-folder", default="magma_outputs", help="stem of magma folder"
    )
    parser.add_argument("--split-name", default="split_1.tsv")

    parser.add_argument("--learning-rate", default=7e-4, action="store", type=float)
    parser.add_argument("--lr-decay-rate", default=1.0, action="store", type=float)
    parser.add_argument("--weight-decay", default=0, action="store", type=float)
    parser.add_argument("--test-checkpoint", default="", action="store", type=str)

    # Fix model params
    parser.add_argument("--layers", default=3, action="store", type=int)
    parser.add_argument("--pe-embed-k", default=0, action="store", type=int)
    parser.add_argument("--set-layers", default=1, action="store", type=int)
    parser.add_argument("--dropout", default=0, action="store", type=float)
    parser.add_argument("--hidden-size", default=256, action="store", type=int)
    parser.add_argument(
        "--mpnn-type", default="GGNN", action="store", choices=["GGNN", "GINE", "PNA"]
    )
    parser.add_argument("--pool-op", default="avg", action="store")
    parser.add_argument(
        "--root-encode",
        default="gnn",
        action="store",
        choices=["gnn", "fp"],
        help="How to encode root of trees",
    )
    parser.add_argument("--inject-early", default=False, action="store_true")
    parser.add_argument("--embed-adduct", default=False, action="store_true")
    parser.add_argument("--embed-collision", default=False, action="store_true")
    parser.add_argument("--embed-elem-group", default=False, action="store_true")
    parser.add_argument("--encode-forms", default=False, action="store_true")
    parser.add_argument("--add-hs", default=False, action="store_true")

    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_frag_train_args(parser)
    return parser.parse_args()


def train_model():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="dag_gen_train.log", debug=kwargs["debug"])
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

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    if kwargs["debug"]:
        df = df[:100]

    spec_names = df["spec"].values
    if kwargs["debug_overfit"]:
        train_inds, val_inds, test_inds = common.get_splits(
            spec_names, split_file, val_frac=0
        )

        # Test specific debug overfit
        # Get 2
        interest_ind = np.argwhere("CCMSLIB00000577858" == spec_names).flatten()[0]

        train_inds = np.array([interest_ind], dtype=np.int64)
        val_inds = np.array([1])
        test_inds = np.array([1])

        # train_inds = train_inds[:6]
    else:
        train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]

    magma_folder = kwargs["magma_folder"]
    num_workers = kwargs.get("num_workers", 0)
    magma_tree_h5 = common.HDF5Dataset(data_dir / f"{magma_folder}/magma_tree.hdf5")
    name_to_json = {Path(i).stem: i for i in magma_tree_h5.get_all_names()}

    pe_embed_k = kwargs["pe_embed_k"]
    root_encode = kwargs["root_encode"]
    embed_elem_group = kwargs["embed_elem_group"]
    tree_processor = dag_data.TreeProcessor(
        pe_embed_k=pe_embed_k, root_encode=root_encode, add_hs=add_hs, embed_elem_group=embed_elem_group,
    )
    # Build out frag datasets
    train_dataset = dag_data.GenDataset(
        train_df,
        magma_h5=data_dir / f"{magma_folder}/magma_tree.hdf5",
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
    )
    val_dataset = dag_data.GenDataset(
        val_df,
        magma_h5=data_dir / f"{magma_folder}/magma_tree.hdf5",
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
    )

    test_dataset = dag_data.GenDataset(
        test_df,
        magma_h5=data_dir / f"{magma_folder}/magma_tree.hdf5",
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
    )

    # Define dataloaders
    collate_fn = train_dataset.get_collate_fn()
    persistent_workers = kwargs["num_workers"] > 0
    mp_contex = 'spawn' if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
        multiprocessing_context=mp_contex,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
        multiprocessing_context=mp_contex,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
        multiprocessing_context=mp_contex,
    )

    # Define model
    model = gen_model.FragGNN(
        hidden_size=kwargs["hidden_size"],
        layers=kwargs["layers"],
        dropout=kwargs["dropout"],
        mpnn_type=kwargs["mpnn_type"],
        set_layers=kwargs["set_layers"],
        learning_rate=kwargs["learning_rate"],
        lr_decay_rate=kwargs["lr_decay_rate"],
        weight_decay=kwargs["weight_decay"],
        node_feats=train_dataset.get_node_feats(),
        pe_embed_k=kwargs["pe_embed_k"],
        pool_op=kwargs["pool_op"],
        root_encode=kwargs["root_encode"],
        inject_early=kwargs["inject_early"],
        embed_adduct=kwargs["embed_adduct"],
        embed_collision=kwargs["embed_collision"],
        embed_elem_group=kwargs["embed_elem_group"],
        encode_forms=kwargs["encode_forms"],
        add_hs=add_hs,
    )

    # test_batch = next(iter(train_loader))
    # outputs = model(test_batch['frag_graphs'], test_batch['root_graphs'],
    #                test_batch['inds'])

    # Create trainer
    monitor = "val_loss"
    if kwargs["debug"]:
        kwargs["max_epochs"] = 2

    if kwargs["debug_overfit"]:
        kwargs["min_epochs"] = 2000
        kwargs["max_epochs"] = None
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
    model = gen_model.FragGNN.load_from_checkpoint(test_checkpoint)
    logging.info(
        f"Loaded model with from {test_checkpoint} with val loss of {test_checkpoint_score}"
    )

    model.eval()
    trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    import time

    start_time = time.time()
    train_model()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
