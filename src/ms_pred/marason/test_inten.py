import logging
import yaml
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

from torch.utils.data import DataLoader


import ms_pred.common as common
from ms_pred.marason import dag_data, inten_model
from sklearn.metrics import pairwise_distances
import torch
import numpy as np
import scipy.sparse as sparse

torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
from sklearn.exceptions import DataConversionWarning
from torch.profiler import profile, record_function, ProfilerActivity
from ms_pred.common.plot_utils import plot_compare_ref_ms_with_structures
import json

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def add_frag_train_args(parser):
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debug-overfit", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_tree_pred/")

    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument(
        "--magma-dag-folder",
        default="data/spec_datasets/gnps2015_debug/magma_outputs/magma_tree",
        help="Folder to have outputs",
    )
    parser.add_argument("--split-name", default="split_1.tsv")
    parser.add_argument("--reference-dir", default="data/closest_neighbors/10eV")
    parser.add_argument("--batch-size", default=3, action="store", type=int)

    # Fix model params
    parser.add_argument("--pe-embed-k", default=0, action="store", type=int)
    parser.add_argument("--mol-threshold", default=0.5, action="store", type=float)
    parser.add_argument("--max-ref-count", default=1, action="store", type=int)
    parser.add_argument(
        "--root-encode",
        default="gnn",
        action="store",
        choices=["gnn", "fp"],
        help="How to encode root of trees",
    )
    parser.add_argument("--logger", 
                        default="dag_inten_test",  
                        action="store")
    parser.add_argument("--inject-early", default=False, action="store_true")
    parser.add_argument("--binned-targs", default=False, action="store_true")
    parser.add_argument("--embed-elem-group", default=False, action="store_true")
    parser.add_argument("--add-hs", default=False, action="store_true")
    parser.add_argument("--add-reference", default=False, action="store_true")
    parser.add_argument("--load-reference", default=False, action="store_true")
    parser.add_argument("--filter-test", default=False, action="store_true")
    parser.add_argument("--draw", default=False, action="store_true")
    parser.add_argument("--test-checkpoint", default="results/marason_inten_nist20/split_1_rnd1/version_0/best.ckpt")
    parser.add_argument("--test-checkpoint2", default=None)
    parser.add_argument("--save-path", default="baseline.csv")
    parser.add_argument("--plot-spec", default=False, action="store_true")
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_frag_train_args(parser)
    return parser.parse_args()

def test():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = common.get_data_dir(dataset_name)
    labels = data_dir / kwargs["dataset_labels"]
    split_file = data_dir / "splits" / kwargs["split_name"]
    add_hs = kwargs["add_hs"]

    add_ref = kwargs['add_reference']
    load_ref = kwargs['load_reference']


    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    if kwargs["debug"]:
        df = df[:1000]
    if "instrument" not in df:
        df["instrument"] = "Orbitrap"

    spec_names = df["spec"].values
    if kwargs["debug_overfit"]:
        train_inds, _, test_inds = common.get_splits(
            spec_names, split_file
        )
        train_inds = train_inds[:1000]
    else:
        train_inds, _, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    test_df = df.iloc[test_inds]



    num_workers = kwargs.get("num_workers", 0)
    magma_dag_folder = Path(kwargs["magma_dag_folder"])
    magma_tree_h5 = common.HDF5Dataset(magma_dag_folder)
    name_to_json = {Path(i).stem.replace("pred_", ""): i for i in magma_tree_h5.get_all_names()}

    pe_embed_k = kwargs["pe_embed_k"]
    root_encode = kwargs["root_encode"]
    binned_targs = kwargs["binned_targs"]
    embed_elem_group = kwargs["embed_elem_group"]
    tree_processor = dag_data.TreeProcessor(
        pe_embed_k=pe_embed_k,
        root_encode=root_encode,
        binned_targs=binned_targs,
        add_hs=add_hs,
        embed_elem_group = embed_elem_group,
        include_draw_dict=kwargs["draw"]
    )
    db_engs = db_specs = None
    test_closest = test_distances = None
    test_valid_ref_counts = None
    db_specs, db_engs = None, None
    if add_ref or kwargs["filter_test"]:
        ref_dir = Path(kwargs["reference_dir"])
        if load_ref:
            if kwargs["debug"]:
                test_closest_df = pd.read_csv(ref_dir / "test_debug.csv")
                test_distances = test_closest_df["distance"]
                test_valid_ref_counts = test_closest_df["valid_ref_count"]
                test_closest = np.load(ref_dir / "test_debug_ref.npy")
                db_specs = sparse.load_npz(ref_dir / "db_debug_specs.npz")
                db_engs = np.load(ref_dir / "db_debug_engs.npy")
            else:
                test_closest_df = pd.read_csv(ref_dir / "test.csv")
                test_closest =  np.load(ref_dir / "test_ref.npy")
                test_valid_ref_counts = test_closest_df["valid_ref_count"]
                test_distances = test_closest_df["distance"]
                db_specs = sparse.load_npz(ref_dir / "db_specs.npz")
                db_engs = np.load(ref_dir / "db_engs.npy")
        else:
            raise NotImplementedError

    try:
        if kwargs["debug"]:
            test_specs = sparse.load_npz(ref_dir / "test_debug_specs.npz")
        else:
            test_specs = sparse.load_npz(ref_dir / "test_specs.npz")
    except:
        test_specs=None
        

    # kwargs['num_workers'] = 0
    persistent_workers = kwargs["num_workers"] > 0
    mp_contex = 'spawn' if num_workers > 0 else None
    # persistent_workers = False

    db_dataset = dag_data.IntenDataset(
        train_df,
        magma_h5=magma_dag_folder,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
    )
    ref_spec_names = db_dataset.spec_names

    test_dataset = dag_data.IntenDataset(
        test_df,
        magma_h5=magma_dag_folder,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
        engs_db = db_engs,
        specs_db = db_specs,
        closest=test_closest,
        closest_distances = test_distances,
        valid_ref_count = test_valid_ref_counts,
        max_ref_count = kwargs["max_ref_count"],
        specs=test_specs,
        ref_spec_names=ref_spec_names
    )
    collate_fn=test_dataset.get_collate_fn()

    if kwargs["filter_test"]:
        test_indices = np.arange(len(test_dataset))
        test_indices = test_indices[(test_distances < kwargs["mol_threshold"])]
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
        pin_memory=kwargs["gpu"],
        multiprocessing_context=mp_contex,
    )    
    test_checkpoint = kwargs["test_checkpoint"]
    test_checkpoint2 = kwargs["test_checkpoint2"] 
    device = "cuda"
    model = inten_model.IntenGNN.load_from_checkpoint(test_checkpoint).to(device)
    model1 = inten_model.IntenGNN.load_from_checkpoint(test_checkpoint2).to(device) if test_checkpoint2 else None
    losses = []
    ref_losses = []
    safe_device = lambda x: x.to(device) if x is not None else x
    test_adducts = []
    with torch.no_grad():
        for batch in test_loader:
            adducts = safe_device(batch["adducts"]).to(device)
            collision_engs = safe_device(batch["collision_engs"]).to(device)
            root_forms = safe_device(batch["root_form_vecs"])
            frag_forms = safe_device(batch["frag_form_vecs"])
            frag_morgans = safe_device(batch["frag_morgans"])
            edge_feats = safe_device(batch["edge_feats"])
            connectivities = safe_device(batch["connectivities"])
            num_edges = safe_device(batch["num_edges"])
            instruments = safe_device(batch["instruments"])

            closest_graphs = safe_device(batch["closest_frag_graphs"])
            closest_root_repr = safe_device(batch["closest_root_reprs"])
            closest_ind_maps = safe_device(batch["closest_inds"])
            closest_num_frags = safe_device(batch["closest_num_frags"])
            closest_broken = safe_device(batch["closest_broken_bonds"])
            closest_adducts = safe_device(batch["closest_adducts"])
            closest_max_remove_hs = safe_device(batch["closest_max_remove_hs"])
            closest_max_add_hs = safe_device(batch["closest_max_add_hs"])
            closest_masses = safe_device(batch["closest_masses"])
            closest_root_forms = safe_device(batch["closest_root_form_vecs"])
            closest_frag_forms = safe_device(batch["closest_frag_form_vecs"])
            closest_frag_morgans = safe_device(batch["closest_frag_morgans"])
            closest_inten_targs = safe_device(batch["closest_inten_targs"])
            closest_edge_feats = safe_device(batch["closest_edge_feats"])
            closest_connectivities = safe_device(batch["closest_connectivities"])
            closest_num_edges = safe_device(batch["closest_num_edges"])
            closest_instruments = instruments

            inten_targs = safe_device(batch["inten_targs"])
            distances = safe_device(batch["distances"])
            draw_dicts = batch["draw_dicts"]

            ref_collision_engs = safe_device(batch["ref_collision_engs"])
            ref_inten_targs = safe_device(batch["ref_inten_targs"])
            ref_counts = safe_device(batch["ref_counts"])

            frag_graphs = safe_device(batch["frag_graphs"])
            root_reprs = safe_device(batch["root_reprs"])
            ind_maps = safe_device(batch["inds"])
            num_frags = safe_device(batch["num_frags"])
            broken_bonds = safe_device(batch["broken_bonds"])
            max_remove_hs = safe_device(batch["max_remove_hs"])
            max_add_hs = safe_device(batch["max_add_hs"])
            masses = safe_device(batch["masses"])
            dag_graphs = safe_device(batch["dag_graphs"])
            closest_dag_graphs = safe_device(batch["closest_dag_graphs"])
            closest_draw_dicts = batch["closest_draw_dicts"]

            inten_preds = model.forward(
                graphs=frag_graphs,
                root_repr=root_reprs,
                ind_maps=ind_maps,
                num_frags=num_frags,
                broken=broken_bonds,
                max_add_hs=max_add_hs,
                max_remove_hs=max_remove_hs,
                masses=masses,
                root_forms=root_forms,
                frag_forms=frag_forms,
                adducts=adducts,
                collision_engs=collision_engs,
                frag_morgans=frag_morgans,
                draw_dicts=draw_dicts,
                dag_graphs=dag_graphs,
                edge_feats=edge_feats,
                connectivities=connectivities,
                num_edges=num_edges,
                instruments=instruments,
                closest_graphs = closest_graphs,
                closest_root_repr = closest_root_repr,
                closest_ind_maps = closest_ind_maps,
                closest_num_frags = closest_num_frags,
                closest_broken=closest_broken,
                closest_adducts=closest_adducts,
                closest_max_remove_hs=closest_max_remove_hs,
                closest_max_add_hs=closest_max_add_hs,
                closest_masses=closest_masses,
                closest_root_forms=closest_root_forms,
                closest_frag_forms=closest_frag_forms,
                closest_frag_morgans = closest_frag_morgans,
                closest_draw_dicts=closest_draw_dicts,
                closest_dag_graphs=closest_dag_graphs,
                closest_edge_feats=closest_edge_feats,
                closest_connectivities=closest_connectivities,
                closest_num_edges=closest_num_edges,
                closest_instruments=closest_instruments,
                distances=distances,
                ref_collision_engs=ref_collision_engs,
                ref_inten_targs=ref_inten_targs,
                ref_counts=ref_counts,
                output_matching = kwargs["draw"]
            )
            loss = model.cos_loss(inten_preds["output_binned"][:, 0, :], inten_targs)
            if kwargs["draw"]:
                return
            if kwargs["plot_spec"]:
                if model1:
                    inten_preds_extra = model1.forward(
                        graphs=frag_graphs,
                        root_repr=root_reprs,
                        ind_maps=ind_maps,
                        num_frags=num_frags,
                        broken=broken_bonds,
                        max_add_hs=max_add_hs,
                        max_remove_hs=max_remove_hs,
                        masses=masses,
                        root_forms=root_forms,
                        frag_forms=frag_forms,
                        adducts=adducts,
                        collision_engs=collision_engs,
                        frag_morgans=frag_morgans,
                        draw_dicts=draw_dicts,
                        dag_graphs=dag_graphs,
                        edge_feats=edge_feats,
                        connectivities=connectivities,
                        num_edges=num_edges,
                        instruments=instruments,
                        closest_graphs = closest_graphs,
                        closest_root_repr = closest_root_repr,
                        closest_ind_maps = closest_ind_maps,
                        closest_num_frags = closest_num_frags,
                        closest_broken=closest_broken,
                        closest_adducts=closest_adducts,
                        closest_max_remove_hs=closest_max_remove_hs,
                        closest_max_add_hs=closest_max_add_hs,
                        closest_masses=closest_masses,
                        closest_root_forms=closest_root_forms,
                        closest_frag_forms=closest_frag_forms,
                        closest_frag_morgans = closest_frag_morgans,
                        closest_draw_dicts=closest_draw_dicts,
                        closest_dag_graphs=closest_dag_graphs,
                        closest_edge_feats=closest_edge_feats,
                        closest_connectivities=closest_connectivities,
                        closest_num_edges=closest_num_edges,
                        closest_instruments=closest_instruments,
                        distances=distances,
                        ref_collision_engs=ref_collision_engs,
                        ref_inten_targs=ref_inten_targs,
                        ref_counts=ref_counts,
                        output_matching = kwargs["draw"]
                    )
                    loss_extra = model1.cos_loss(inten_preds_extra["output_binned"][:, 0, :], inten_targs)["loss"].cpu().numpy()
                else:
                    inten_preds_extra = None
                loss = loss["loss"].cpu().numpy()
                for i in range(kwargs["batch_size"]):
                    if inten_preds_extra is not None:
                        predicted_spec_extra = inten_preds_extra["output_binned"][i, 0, :].cpu().numpy()
                        predicted_pos_extra = np.array(np.nonzero(predicted_spec_extra))[0]
                        predicted_val_extra = predicted_spec_extra[predicted_pos_extra]
                        predicted_spec_extra = np.stack((predicted_pos_extra/10, predicted_val_extra), axis=-1)
                        extra_loss = loss_extra[i].item()
                    else:
                        predicted_spec_extra = None
                        extra_loss = None
                    target_file_name, ref_file_name = test_dataset.spec_names[i], ref_spec_names[test_closest[i, 0]]
                    target_spec = np.array(test_dataset.load_tree(target_file_name)["raw_spec"])
                    ref_spec = np.array(db_dataset.load_tree(ref_file_name)["raw_spec"])
                    target_smis = test_dataset.name_to_smiles[target_file_name]
                    ref_smis = db_dataset.name_to_smiles[ref_file_name]
                    target_eng, ref_eng = test_dataset[i]["collision_energy"], db_engs[test_closest[i, 0]] 
                    target_ev, ref_ev = f"{target_eng}", f"{ref_eng}"
                    predicted_spec = inten_preds["output_binned"][i, 0, :].cpu().numpy()
                    predicted_pos = np.array(np.nonzero(predicted_spec))[0]
                    predicted_val = predicted_spec[predicted_pos]
                    predicted_spec = np.stack((predicted_pos/10, predicted_val), axis=-1)

                    plot_compare_ref_ms_with_structures(target_spec, ref_spec, spec3=predicted_spec, spec4=predicted_spec_extra,
                                                        spec1_name="Experimental", spec2_name="Reference", spec3_name=f"Predicted (MARASON) Dist={round(loss[i].item(), 3)}", 
                                                        spec4_name=f"Predicted (MARASON no RAG) Dist={round(extra_loss, 3)}", spec1_smiles=target_smis, spec2_smiles=ref_smis, 
                                                        spec1_ce_label=target_ev, spec2_ce_label=ref_ev, save_path = f"MARASON spectrum visualization{i}.pdf")         
                return
            losses.append(loss["loss"].cpu())
            test_adducts.append(adducts.cpu().numpy())
            if add_ref:
                ref_loss = model.cos_loss(closest_inten_targs, inten_targs)
                ref_losses.append(ref_loss["loss"].cpu())

    test_adducts = np.concatenate(test_adducts)
    if kwargs["draw"]:
        raise ValueError("No valid matching pattern is presented.")
    losses = torch.cat(losses)
    avg_losses = torch.mean(losses).item()
    print(f"loss: {avg_losses}")
    if add_ref:
        ref_losses = torch.cat(ref_losses)
        avg_ref_losses = torch.mean(ref_losses).item()
        print(f"ref loss:{avg_ref_losses}")
    save_df = pd.DataFrame({"loss":losses.numpy(), "adducts":test_adducts})
    save_df.to_csv(kwargs["save_path"])

          
if __name__ == "__main__":
    test()


