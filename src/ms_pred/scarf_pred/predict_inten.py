"""predict_inten.py

Make intensity predictions with trained model

"""

import logging
from datetime import datetime
import yaml
import argparse
import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import ms_pred.common as common
import ms_pred.scarf_pred.scarf_data as scarf_data
import ms_pred.scarf_pred.scarf_model as scarf_model

import ms_pred.nn_utils as nn_utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--binned-out", default=False, action="store_true")
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_ffn_pred/")
    parser.add_argument(
        "--checkpoint-pth",
        help="name of checkpoint file",
        default="results/2022_06_22_pretrain/version_3/epoch=99-val_loss=0.87.ckpt",
    )
    parser.add_argument(
        "--formula-folder", default="subform_50", help="stem of formula folder"
    )
    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")
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
    common.setup_logger(save_dir, log_name="inten_pred.log", debug=kwargs["debug"])
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
            names = splits[splits[fold_name] == "test"]["spec"].tolist()
            names = names[:5]
        else:
            raise NotImplementedError()
        df = df[df["spec"].isin(names)]

    # Create model and load
    # Load from checkpoint
    best_checkpoint = kwargs["checkpoint_pth"]
    model = scarf_model.ScarfIntenNet.load_from_checkpoint(best_checkpoint)
    logging.info(f"Loaded model with from {best_checkpoint}")

    num_workers = kwargs.get("num_workers", 0)
    form_dag_folder = Path(kwargs["formula_folder"])
    all_json_pths = [Path(i) for i in form_dag_folder.glob("*.json")]
    name_to_json = {i.stem.replace("pred_", ""): i for i in all_json_pths}

    graph_featurizer = nn_utils.MolDGLGraph(
        atom_feats=model.atom_feats,
        bond_feats=model.bond_feats,
        pe_embed_k=model.pe_embed_k,
    )
    pred_dataset = scarf_data.IntenDataset(
        df,
        num_workers=num_workers,
        data_dir=data_dir,
        form_map=name_to_json,
        graph_featurizer=graph_featurizer,
        root_embedder=model.root_embedder,
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
    device = "cuda" if gpu else "cpu"

    # Step 1: Get all predictions
    pred_list = []
    binned_out = kwargs["binned_out"]
    with torch.no_grad():
        for batch in tqdm(pred_loader):

            # IDs to use to recapitulate
            spec_names = batch["names"]
            form_strs = batch["form_strs"]
            graphs = batch["graphs"].to(device)
            formulae = batch["formulae"].to(device)
            diffs = batch["diffs"].to(device)
            num_forms = batch["num_forms"].to(device)
            adducts = batch["adducts"].to(device)

            outputs = model.predict(
                graphs=graphs,
                full_formula=formulae,
                diffs=diffs,
                num_forms=num_forms,
                adducts=adducts,
                binned_out=binned_out,
            )

            output_specs = outputs["spec"]
            for spec, form_str, output_spec in zip(spec_names, form_strs, output_specs):
                output_obj = {
                    "spec_name": spec,
                    "forms": form_str,
                    "form_masses": [common.formula_mass(i) for i in form_str],
                    "output_spec": output_spec,
                    "smiles": pred_dataset.name_to_smiles[spec],
                    "root_form": pred_dataset.name_to_root_form[spec],
                }
                pred_list.append(output_obj)

    # Export pred objects
    if binned_out:
        spec_names_ar = [str(i["spec_name"]) for i in pred_list]
        smiles_ar = [str(i["smiles"]) for i in pred_list]
        inchikeys = [common.inchikey_from_smiles(i) for i in smiles_ar]
        preds = np.vstack([i["output_spec"] for i in pred_list])
        output = {
            "preds": preds,
            "smiles": smiles_ar,
            "ikeys": inchikeys,
            "spec_names": spec_names_ar,
            "num_bins": model.inten_buckets.shape[-1],
            "upper_limit": 1500,
            "sparse_out": False,
        }
        out_file = Path(kwargs["save_dir"]) / "binned_preds.p"
        with open(out_file, "wb") as fp:
            pickle.dump(output, fp)
    else:
        for pred_obj in pred_list:
            mz = pred_obj["form_masses"]
            intens = [float(i) for i in pred_obj["output_spec"]]
            cand_form = pred_obj["root_form"]
            smiles = pred_obj["smiles"]
            form_list = pred_obj["forms"]
            spec_name = pred_obj["spec_name"]
            tbl = {
                "mz": mz,
                "ms2_inten": intens,
                "rel_inten": intens,
                "mono_mass": mz,
                "formula_mass_no_adduct": mz,
                "mass_diff": [0] * len(mz),
                "formula": form_list,
                "ions": ["H+"] * len(mz),
            }
            new_form = {
                "cand_form": cand_form,
                "spec_name": spec_name,
                "cand_ion": "H+",
                "output_tbl": tbl,
                "smiles": smiles,
            }
            save_path = Path(kwargs["save_dir"]) / "tree_preds_inten"
            save_path.mkdir(exist_ok=True)
            out_file = save_path / f"pred_{spec_name}.json"
            with open(out_file, "w") as fp:
                json.dump(new_form, fp, indent=2)


if __name__ == "__main__":
    import time

    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
