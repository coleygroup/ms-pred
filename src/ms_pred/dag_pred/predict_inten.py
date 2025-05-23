"""predict_inten.py

Make intensity predictions with trained model

"""

import logging
from datetime import datetime
import yaml
import argparse
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import ms_pred.common as common
from ms_pred.dag_pred import dag_data, inten_model


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
        "--magma-dag-folder",
        help="Folder to have outputs",
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
    pl.utilities.seed.seed_everything(kwargs.get("seed"))
    binned_out = kwargs["binned_out"]

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
            names = ["mona_1118"]
            # names = splits[splits[fold_name] == "test"]["spec"].tolist()
            names = ["CCMSLIB00001058857"]
            names = ["CCMSLIB00001058185"]
            # names = names[:5]
        else:
            raise NotImplementedError()
        df = df[df["spec"].isin(names)]

    # Create model and load
    # Load from checkpoint
    best_checkpoint = kwargs["checkpoint_pth"]
    model = inten_model.IntenGNN.load_from_checkpoint(best_checkpoint)
    logging.info(f"Loaded model with from {best_checkpoint}")

    pe_embed_k = model.pe_embed_k
    root_encode = model.root_encode
    add_hs = model.add_hs
    embed_elem_group = model.embed_elem_group
    magma_dag_folder = Path(kwargs["magma_dag_folder"])
    magma_tree_h5 = common.HDF5Dataset(magma_dag_folder)
    name_to_json = {Path(i).stem.replace("pred_", ""): i for i in magma_tree_h5.get_all_names()}
    num_workers = kwargs.get("num_workers", 0)

    tree_processor = dag_data.TreeProcessor(
        pe_embed_k=pe_embed_k, root_encode=root_encode, add_hs=add_hs, embed_elem_group=embed_elem_group,
    )
    pred_dataset = dag_data.IntenPredDataset(
        df,
        tree_processor=tree_processor,
        num_workers=num_workers,
        data_dir=data_dir,
        magma_h5=magma_dag_folder,
        magma_map=name_to_json,
    )
    # Define dataloaders
    collate_fn = pred_dataset.get_collate_fn()
    mp_contex = 'spawn' if num_workers > 0 else None
    pred_loader = DataLoader(
        pred_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        multiprocessing_context=mp_contex,
    )

    model.eval()
    gpu = kwargs["gpu"]
    if gpu:
        model = model.cuda()

    device = torch.device("cuda") if gpu else torch.device("cpu")
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(pred_loader):

            frag_graphs = batch["frag_graphs"].to(device)
            root_reprs = batch["root_reprs"].to(device)
            ind_maps = batch["inds"].to(device)
            num_frags = batch["num_frags"].to(device)
            broken_bonds = batch["broken_bonds"].to(device)
            max_remove_hs = batch["max_remove_hs"].to(device)
            max_add_hs = batch["max_add_hs"].to(device)

            # IDs to use to recapitulate
            spec_names = batch["names"]
            inten_frag_ids = batch["inten_frag_ids"]
            masses = batch["masses"].to(device)
            adducts = batch["adducts"].to(device)
            precursor_mzs = batch["precursor_mzs"].to(device)
            collision_energies = batch["collision_engs"].to(device)

            root_forms = batch["root_form_vecs"].to(device)
            frag_forms = batch["frag_form_vecs"].to(device)

            outputs = model.predict(
                graphs=frag_graphs,
                root_reprs=root_reprs,
                ind_maps=ind_maps,
                num_frags=num_frags,
                max_breaks=broken_bonds,
                adducts=adducts,
                precursor_mzs=precursor_mzs,
                collision_engs=collision_energies,
                max_add_hs=max_add_hs,
                max_remove_hs=max_remove_hs,
                masses=masses,
                root_forms=root_forms,
                frag_forms=frag_forms,
                binned_out=binned_out,
            )

            outputs = outputs["spec"]
            for spec, inten_frag_id, collision_energy, output_spec in zip(
                spec_names, inten_frag_ids, collision_energies, outputs
            ):
                output_obj = {
                    "spec_name": spec,
                    "frag_ids": inten_frag_id,
                    "output_spec": output_spec,
                    "smiles": pred_dataset.name_to_smiles[spec],
                    "collision_energy": collision_energy,
                }
                pred_list.append(output_obj)

    # Export pred objects
    if binned_out:
        h5 = common.HDF5Dataset(Path(kwargs["save_dir"]) / "binned_preds.hdf5", mode='w')
        h5.attrs['num_bins'] = model.inten_buckets.shape[-1]
        h5.attrs['upper_limit'] = 1500
        h5.attrs['sparse_out'] = False
        for output_obj in pred_list:
            spec_name = output_obj["spec_name"]
            smi = output_obj["smiles"]
            inchikey = common.inchikey_from_smiles(smi)
            collision_energy = output_obj["collision_energy"]
            output_spec = output_obj["output_spec"]
            h5_name = f'pred_{spec_name}/ikey {inchikey}/collision {collision_energy}'
            h5.write_data(h5_name + '/spec', output_spec)
            h5.update_attr(h5_name, {'smiles': smi, 'ikey': inchikey, 'spec_name': spec_name})
        h5.close()

        # spec_names_ar = [str(i["spec_name"]) for i in pred_list]
        # smiles_ar = [str(i["smiles"]) for i in pred_list]
        # inchikeys = [common.inchikey_from_smiles(i) for i in smiles_ar]
        # preds = np.vstack([i["output_spec"] for i in pred_list])
        # output = {
        #     "preds": preds,
        #     "smiles": smiles_ar,
        #     "ikeys": inchikeys,
        #     "spec_names": spec_names_ar,
        #     "num_bins": model.inten_buckets.shape[-1],
        #     "upper_limit": 1500,
        #     "sparse_out": False,
        # }
        # out_file = Path(kwargs["save_dir"]) / "binned_preds.p"
        # with open(out_file, "wb") as fp:
        #     pickle.dump(output, fp)
    else:
        raise NotImplementedError()
        # Process each spec
        # zip_iter = zip(spec_names, inten_frag_ids)
        # for spec_ind, (spec_name, frag_ids) in enumerate(zip_iter):

        #    # Step 1: Get tree from dataset
        #    pred_tree = pred_dataset.spec_name_to_tree[spec_name]
        #    new_tree = copy.deepcopy(pred_tree)

        #    # Extract from output dict
        #    spec_intens = outputs["spec"][spec_ind]
        #    other_keys = set(outputs.keys()).difference(["spec"])

        #    # Step 2: Add in new info to the tree
        #    for ind, frag in enumerate(frag_ids):
        #        inten_vec = spec_intens[ind]
        #        new_tree["frags"][frag]["intens"] = inten_vec.tolist()

        #        for k in other_keys:
        #            new_tree["frags"][frag][k] = outputs[k][spec_ind][ind].tolist()

        #    # Step 3: Output to file
        #    save_path = Path(kwargs["save_dir"]) / "tree_preds_inten"
        #    save_path.mkdir(exist_ok=True)
        #    out_file = save_path / f"pred_{spec_name}.json"
        #    with open(out_file, "w") as fp:
        #        json.dump(new_tree, fp, indent=2)


if __name__ == "__main__":
    import time

    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
