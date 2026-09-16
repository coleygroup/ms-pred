"""predict.py

Make predictions with trained model

"""

import logging
from datetime import datetime
import yaml
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
from ms_pred.molnetms import molnetms_data, molnetms_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--sparse-out", default=False, action="store_true")
    parser.add_argument("--sparse-k", default=100, action="store", type=int)
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_molnetms_pred/")
    parser.add_argument(
        "--checkpoint-pth",
        help="name of checkpoint file",
        default="results/2022_06_22_pretrain/version_3/epoch=99-val_loss=0.87.ckpt",
    )
    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")
    parser.add_argument(
        "--subset-datasets",
        default="none",
        action="store",
        choices=["none", "train_only", "test_only"],
    )
    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__
    sparse_out = kwargs["sparse_out"]
    sparse_k = kwargs["sparse_k"]

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="molnetms_pred.log", debug=kwargs["debug"])
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

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")

    if kwargs["subset_datasets"] != "none":
        splits = pd.read_csv(data_dir / "splits" / kwargs["split_name"], sep="\t")
        folds = set(splits.keys())
        folds.remove("spec")
        fold_name = list(folds)[0]
        if kwargs["subset_datasets"] == "train_only":
            names = splits[splits[fold_name] == "train"]["spec"].tolist()
        elif kwargs["subset_datasets"] == "test_only":
            names = splits[splits[fold_name] == "test"]["spec"].tolist()
        else:
            raise NotImplementedError()
        df = df[df["spec"].isin(names)]

    num_workers = kwargs.get("num_workers", 0)

    # Create model and load
    # Load from checkpoint
    best_checkpoint = kwargs["checkpoint_pth"]
    model = molnetms_model.MolNetMS.load_from_checkpoint(best_checkpoint)
    logging.info(f"Loaded model with from {best_checkpoint}")

    graph_featurizer = molnetms_data.MolMSFeaturizer()
    pred_dataset = molnetms_data.MolDataset(
        df, num_workers=num_workers, graph_featurizer=graph_featurizer
    )

    # Define dataloaders
    collate_fn = pred_dataset.get_collate_fn()
    pred_loader = DataLoader(
        pred_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    model.eval()
    gpu = kwargs["gpu"]
    if gpu:
        model = model.cuda()

    spec_names_ar, smiles_ar, preds = [], [], []
    with torch.no_grad():
        for batch in tqdm(pred_loader):
            graphs, smiles, weights, adducts = (
                batch["graphs"],
                batch["names"],
                batch["full_weight"],
                batch["adducts"],
            )
            spec_names = batch["spec_names"]
            if gpu:
                graphs = graphs.to("cuda")
                weights = weights.to("cuda")
                adducts = adducts.to("cuda")

            output = model.predict(graphs, weights, adducts)
            output_spec = output["spec"].cpu().detach().numpy()
            smiles_ar.append(smiles)
            spec_names_ar.append(spec_names)
            # Shrink it to only top k, ordering inds, intens

            if sparse_out:
                best_inds = np.argsort(output_spec, -1)[:, ::-1][:, :sparse_k]
                best_intens = np.take_along_axis(output_spec, best_inds, -1)
                output_spec = np.stack([best_inds, best_intens], -1)

            preds.append(output_spec)

        spec_names_ar = [j for i in spec_names_ar for j in i]
        smiles_ar = [j for i in smiles_ar for j in i]
        inchikeys = [common.inchikey_from_smiles(i) for i in smiles_ar]
        preds = np.vstack(preds)

        output = {
            "preds": preds,
            "smiles": smiles_ar,
            "ikeys": inchikeys,
            "spec_names": spec_names_ar,
            "num_bins": model.output_dim,
            "upper_limit": model.upper_limit,
            "sparse_out": sparse_out,
        }

        out_file = Path(kwargs["save_dir"]) / "binned_preds.p"
        with open(out_file, "wb") as fp:
            pickle.dump(output, fp)


if __name__ == "__main__":
    import time

    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
