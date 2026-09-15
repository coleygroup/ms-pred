"""predict.py

Make predictions with trained model

"""
import logging
from datetime import datetime
import yaml
import argparse
import pickle
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
from ms_pred.scarf_pred import scarf_data, scarf_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=1, action="store", type=int)
    parser.add_argument("--max-nodes", default=200, action="store", type=int)
    parser.add_argument("--threshold", default=0, action="store", type=float)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_scarf_pred/")
    parser.add_argument(
        "--checkpoint-pth",
        help="name of checkpoint file",
        default="results/2022_06_22_pretrain/version_3/epoch=99-val_loss=0.87.ckpt",
    )
    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_1.tsv")
    parser.add_argument(
        "--subset-datasets",
        default="none",
        action="store",
        choices=["none", "train_only", "test_only", "debug_special"],
    )
    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="scarf_pred.log", debug=kwargs["debug"])
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
        elif kwargs["subset_datasets"] == "debug_special":

            names = splits[splits[fold_name] == "train"]["spec"].tolist()
            names = names[:6]
            # names = ["CCMSLIB00000001568"]
        else:
            raise NotImplementedError()
        df = df[df["spec"].isin(names)]

    num_workers = kwargs.get("num_workers", 0)

    # Create model and load
    # Load from checkpoint
    best_checkpoint = kwargs["checkpoint_pth"]
    model = scarf_model.ScarfNet.load_from_checkpoint(best_checkpoint)
    logging.info(f"Loaded model with from {best_checkpoint}")
    graph_featurizer = nn_utils.MolDGLGraph(
        atom_feats=model.atom_feats,
        bond_feats=model.bond_feats,
        pe_embed_k=model.pe_embed_k,
    )

    pred_dataset = scarf_data.MolDataset(
        df,
        num_workers=num_workers,
        graph_featurizer=graph_featurizer,
        root_embedder=model.root_embedder,
    )

    # Define dataloaders
    collate_fn = pred_dataset.get_collate_fn()
    # Require batch size 1
    # assert(kwargs['batch_size'] == 1)
    pred_loader = DataLoader(
        pred_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    # Create model and load
    best_checkpoint = kwargs["checkpoint_pth"]

    # Load from checkpoint
    model = scarf_model.ScarfNet.load_from_checkpoint(best_checkpoint)
    logging.info(f"Loaded model with from {best_checkpoint}")

    # Debug
    # batch = next(iter(pred_loader))
    # formula_tensors, graphs = batch['formula_tensors'], batch['mol_graphs']
    # outputs, intens = model.make_prediction(formula_tensors=formula_tensors,
    #                                        mol_graphs=graphs,)
    model.eval()
    gpu = kwargs["gpu"]
    if gpu:
        model = model.cuda()

    save_path = Path(kwargs["save_dir"]) / "form_preds"
    save_path.mkdir(exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(pred_loader):

            graphs, batch_names, formula_tensors, adducts = (
                batch["mol_graphs"],
                batch["names"],
                batch["formula_tensors"],
                batch["adducts"],
            )
            spec_names = batch["spec_names"]

            if gpu:
                graphs = graphs.to("cuda")
                formula_tensors = formula_tensors.cuda()

            outputs, intens = model.make_prediction(
                formula_tensors=formula_tensors,
                mol_graphs=graphs,
                threshold=kwargs["threshold"],
                max_nodes=kwargs["max_nodes"],
                adducts=adducts,
            )

            for spec_name, smi_name, forms, inten in zip(
                spec_names, batch_names, outputs, intens
            ):
                out_file = save_path / f"pred_{spec_name}.json"
                inten_export = [float(np_int) for np_int in inten]
                inten_logit = [float(np.log(np_int)) for np_int in inten]
                form_masses = [common.formula_mass(i) for i in forms]
                orig_form = common.form_from_smi(smi_name)

                output_tbl = {
                    "ms2_inten": inten_export,
                    "rel_inten": inten_export,
                    "log_prob": inten_logit,
                    "formula": forms,
                    "formula_mass_no_adduct": form_masses,
                }

                json_out = {
                    "smiles": smi_name,
                    "spec_name": spec_name,
                    "output_tbl": output_tbl,
                    "cand_form": orig_form,
                }
                with open(out_file, "w") as fp:
                    json.dump(json_out, fp, indent=2)


if __name__ == "__main__":
    import time

    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
