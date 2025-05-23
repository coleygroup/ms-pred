"""train.py

Train Massformer to predict binned specs

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
from ms_pred.massformer_pred import massformer_data, massformer_model
import ms_pred.nn_utils as nn_utils


def add_massformer_train_args(parser):
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debug-overfit", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=128, action="store", type=int)
    parser.add_argument("--max-epochs", default=100, action="store", type=int)
    parser.add_argument("--min-epochs", default=0, action="store", type=int)

    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_mf/")

    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")

    parser.add_argument("--learning-rate", default=4e-7, action="store", type=float)
    parser.add_argument("--lr-decay-rate", default=1.0, action="store", type=float)
    parser.add_argument("--weight-decay", default=0, action="store", type=float)

    parser.add_argument("--num-bins", default=15000, action="store", type=int)

    parser.add_argument(
        "--loss-fn",
        default="cosine",
        action="store",
        choices=["mse", "hurdle", "cosine"],
    )
    parser.add_argument("--form-dir-name", default="subform_20", action="store")

    # embed adduct
    parser.add_argument("--embed-adduct", default=False, action="store_true")
    parser.add_argument("--embed-collision-energy", default=False, action="store_true")
    parser.add_argument("--use-reverse", default=False, action="store_true")

    parser.add_argument("--mf-num-ff-num-layers", type=int)
    parser.add_argument("--mf-ff-h-dim", type=int)
    parser.add_argument("--mf-ff-skip", action="store_true")
    parser.add_argument("--mf-layer-type", default="neims")
    parser.add_argument("--mf-dropout", type=float)

    parser.add_argument("--gf-model-name", default="graphormer_base")
    parser.add_argument("--gf-pretrain-name", default="pcqm4mv2_graphormer_base")
    parser.add_argument("--gf-fix-num-pt-layers", type=int, default=0)
    parser.add_argument("--gf-reinit-num-pt-layers", type=int, default=-1)
    parser.add_argument("--gf-reinit-layernorm", action="store_true")

    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_massformer_train_args(parser)
    return parser.parse_args()


def test_model():
    args = get_args()
    kwargs = args.__dict__
    upper_limit = 1500

    save_dir = kwargs["save_dir"]
    common.setup_logger(
        save_dir, log_name="massformer_train.log", debug=kwargs["debug"]
    )
    pl.seed_everything(kwargs.get("seed"))

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
    test_df = df.iloc[test_inds]

    num_bins = kwargs.get("num_bins")
    num_workers = kwargs.get("num_workers", 0)
    test_dataset = massformer_data.BinnedDataset(
        test_df,
        data_dir=data_dir,
        num_bins=num_bins,
        num_workers=num_workers,
        upper_limit=upper_limit,
        form_dir_name=kwargs["form_dir_name"],
    )

    # Define dataloaders
    collate_fn = test_dataset.get_collate_fn()
    test_loader = DataLoader(
        test_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

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

    # Load from checkpoint
    best_checkpoint = Path(save_dir) / 'version_0/best.ckpt'
    model = massformer_model.MassFormer.load_from_checkpoint(best_checkpoint)
    logging.info(
        f"Loaded model with from {best_checkpoint}"
    )
    model.eval()
    trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    import time

    start_time = time.time()
    test_model()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
