""" train_frag.py

PyTorch Lightning training script for the fragmentation-only model using the merged OneStepDataset.
Matches the structure and conveniences of train.py.
"""
import os
import logging
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader

import ms_pred.common as common
from ms_pred.iceberg_transformer.dataset import IntenDataset, TreeProcessor
from ms_pred.iceberg_transformer.frag_model import FragOnlyModel
import ms_pred.nn_utils as nn_utils


def add_frag_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debug-overfit", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-epochs", default=100, type=int)
    parser.add_argument("--min-epochs", default=0, type=int)

    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_frag_only/")

    parser.add_argument("--dataset-name", default="nist20")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--magma-folder", default="magma_outputs")
    parser.add_argument("--split-name", default="split_1.tsv")

    parser.add_argument("--learning-rate", default=7e-4, type=float)
    parser.add_argument("--lr-decay-rate", default=1.0, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--test-checkpoint", default="", type=str)

    # Model params
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--root-encode", default="graphormer", choices=["gnn", "graphormer"], type=str)
    parser.add_argument("--add-hs", default=True, action="store_true")
    parser.add_argument("--embed-elem-group", default=True, action="store_true")
    parser.add_argument("--pe-embed-k", default=10, type=int)
    parser.add_argument("--layers", default=6, type=int)
    parser.add_argument("--warmup", default=1000, type=int)
    parser.add_argument("--max-frags", default=100, type=int)
    parser.add_argument("--multi-hop-max-dist", default=5, type=int)
    parser.add_argument("--num-edge-dis", default=10, type=int)
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_frag_train_args(parser)
    return parser.parse_args()

def connected_subgraph_mask_batch(adj_batch, masks_batch, max_iter=None):
    """
    Vectorized connectivity check for masked subgraphs in a batch of graphs.
    
    Parameters
    ----------
    adj_batch : (B, N, N) float/bool Tensor
        Batched adjacency matrices (0/1), padded with zeros.
    masks_batch : (B, M, N) float/bool Tensor
        Batched subgraph masks (0/1), padded with zeros.
    node_counts : (B,) Tensor[int], optional
        Number of valid nodes per graph. If None, assume all N are valid.
    max_iter : int, optional
        Max BFS iterations. Defaults to N (sufficient for connectivity).
    
    Returns
    -------
    connected : (B, M) BoolTensor
        Whether each subgraph is connected.
    """
    B, N, _ = adj_batch.shape
    _, M, _ = masks_batch.shape
    device = adj_batch.device
    
    if max_iter is None:
        max_iter = N
    
    # expand adjacency for (B, M, N, N)
    A = adj_batch[:, None, :, :].expand(B, M, N, N)
    
    # masks (B, M, N)
    masks = masks_batch.bool()
    sizes = masks.sum(-1)  # (B, M)
    
    # choose first node in each mask as "root"
    first_nodes = masks.float().cumsum(-1) == 1  # (B, M, N)
    
    # initialize frontier & visited
    visited = first_nodes.clone()
    frontier = first_nodes.clone()
    
    for _ in range(max_iter):
        if not frontier.any():
            break
        # multiply adjacency with frontier: (B, M, N)
        # frontier (B, M, N) -> (B,M,1,N)
        new_reach = torch.matmul(frontier.unsqueeze(-2).float(), A.float()).squeeze(-2).bool()
        new_reach = new_reach & masks & ~visited
        visited |= new_reach
        frontier = new_reach
    
    # subgraph connected if visited covers all masked nodes
    connected = (visited.sum(-1) == sizes) | (sizes <= 1)
    return connected


def test_model():
    args = get_args()
    kwargs: Dict[str, Any] = args.__dict__

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="frag_test.log", debug=kwargs["debug"])
    pl.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Dataset
    dataset_name = kwargs["dataset_name"]
    data_dir = common.get_data_dir(dataset_name)
    labels = data_dir / kwargs["dataset_labels"]
    split_file = data_dir / "splits" / kwargs["split_name"]

    df = pd.read_csv(labels, sep="\t")
    if kwargs["debug"]:
        df = df[:1000]
    spec_names = df["spec"].values
    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    test_df = df.iloc[test_inds]

    magma_folder = kwargs["magma_folder"]
    num_workers = kwargs.get("num_workers", 0)
    magma_h5_path = data_dir / f"{magma_folder}/magma_tree_with_inten.hdf5"
    magma_h5 = common.HDF5Dataset(magma_h5_path)
    name_to_json = {Path(i).stem: i for i in magma_h5.get_all_names()}

    # Processor and datasets
    tree_processor = TreeProcessor(
        pe_embed_k=kwargs["pe_embed_k"],
        root_encode=kwargs["root_encode"],
        binned_targs=False,
        add_hs=kwargs["add_hs"],
        embed_elem_group=kwargs["embed_elem_group"],
        multi_hop_max_dist=kwargs["multi_hop_max_dist"],
    )

    test_dataset = IntenDataset(
        test_df,
        magma_h5=magma_h5_path,
        magma_map=name_to_json,
        num_workers=num_workers,
        root_encode=kwargs["root_encode"],
        binned_targs=False,
        add_hs=kwargs["add_hs"],
        embed_elem_group=kwargs["embed_elem_group"],
        tree_processor=tree_processor,
        datatype="HDF5",
    )

    # Dataloaders
    collate_fn = test_dataset.get_collate_fn()
    persistent_workers = kwargs["num_workers"] > 0
    mp_context = 'spawn' if num_workers > 0 else None
    test_loader = DataLoader(
        test_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
        multiprocessing_context=mp_context,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FragOnlyModel.load_from_checkpoint(kwargs["test_checkpoint"]).to(device)
    pred_boundary = model.pred_boundary
    model.freeze()
    model.eval()
    recalls = []
    connected_counts = []
    with torch.no_grad():
        for batch in test_loader:
            for c in batch:
                if isinstance(batch[c], torch.Tensor):
                    batch[c] = batch[c].to(device)
                if c == "graphormer_input":
                    for k in batch[c]:
                        if isinstance(batch[c][k], torch.Tensor):
                            batch[c][k] = batch[c][k].to(device)
            out = model(batch)
            pattern_predicted = out["frag_predicted"] > 0.5
            adj_batch = batch["adj_matrices"] > 0.5
            frag_targs_padded = nn_utils.pad_packed_tensor(
                batch["frag_targs"], batch["num_frag_targs"], False
            )
            if pred_boundary:
                pattern_predicted, pattern_count = model.breakpoints_to_patterns(pattern_predicted, adj_batch, batch["num_atoms"])
            pattern_match_recall = model.pattern_match_recall(pattern_predicted, frag_targs_padded, batch["num_frag_targs"])
            recalls.append(pattern_match_recall)
            connected_count = connected_subgraph_mask_batch(adj_batch, pattern_predicted)
            connected_counts.append(connected_count.sum().item())
        total_connected = sum(connected_counts)
        logging.info(f"Total connected fragments ratio: {total_connected} / {len(test_dataset)*100}")
        avg_recall = torch.mean(torch.cat(recalls)).item()
        logging.info(f"Average test set fragment pattern recall: {avg_recall:.4f}")
def main():
    test_model()


if __name__ == "__main__":
    import time
    start_time = time.time()
    test_model()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")