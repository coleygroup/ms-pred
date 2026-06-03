"""predict_smis_joint.py

Make joint fragment + intensity predictions for a table of SMILES and write a
sparse PredSpecDB.

Performance design
------------------
The expensive part of inference is **featurization** -- turning each molecule
into a Graphormer input (BFS distance matrix + O(N^2 * multi_hop_max_dist)
edge-path features + random-walk positional encodings) plus the MAGMA fragment
engine / formula vectors. ``forward()`` on the GPU is comparatively cheap and,
at a reasonable batch size, saturates the GPU on its own.

Previously featurization ran inline in the GPU process (``predict_mol``), so the
GPU sat idle waiting for the CPU. Here featurization is pushed into a
``torch.utils.data.DataLoader`` with many CPU worker processes that prefetch
batches while the GPU runs ``forward`` on the previous batch. Within a batch,
identical molecules (the same SMILES across several collision energies) are
featurized only once via a per-worker LRU cache, and entries are sorted by
(heavy-atom count, SMILES) so identical molecules land in the same batch and
similarly sized molecules share a batch (less padding waste).

Optionally shards the work across multiple GPUs (one process per GPU).
"""
import logging
import os
import math
import ast
import queue
import threading
from collections import OrderedDict
from datetime import datetime
import yaml
import argparse
from pathlib import Path
import pandas as pd

from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

import ms_pred.common as common

from rdkit import Chem
from rdkit import rdBase
from rdkit import RDLogger
from ms_pred.common import chem_utils

# ms_pred.GLACIER.joint_model / dataset pull in the heavy stack (pytorch_lightning,
# dgl, LinSATNet, pygmtools). They are imported lazily (see _lazy_imports) so the
# spawn-based prepare_entry worker pool -- which re-imports this module just to run
# prepare_entry and never touches the model/dataset -- does not pay that ~30s/worker
# import cost (which otherwise balloons pool startup to minutes).
joint_model = None
glacier_dataset = None


def _lazy_imports():
    global joint_model, glacier_dataset
    if joint_model is None:
        from ms_pred.GLACIER import joint_model as _jm, dataset as _gd
        joint_model = _jm
        glacier_dataset = _gd


rdBase.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.*")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--sparse-out", default=False, action="store_true")
    parser.add_argument("--sparse-k", default=100, action="store", type=int)
    parser.add_argument(
        "--num-gpu-workers",
        default=0,
        action="store",
        type=int,
        help="Number of GPU shards (processes). 0/1 => single GPU. >1 => shard "
        "molecules across that many GPUs, each writing its own shard file.",
    )
    parser.add_argument(
        "--num-cpu-workers",
        default=32,
        action="store",
        type=int,
        help="DataLoader worker processes used for featurization (per GPU shard).",
    )
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    parser.add_argument("--prefetch-factor", default=4, action="store", type=int)
    parser.add_argument(
        "--no-dedup",
        default=False,
        action="store_true",
        help="Disable within-batch unique-molecule featurization caching/sorting. "
        "Forces every (mol, collision-energy) row to be re-featurized -- useful to "
        "stress-test featurization throughput as if every row were a distinct molecule.",
    )
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_ffn_pred/")
    parser.add_argument("--out-name", default="preds.hdf5")
    parser.add_argument("--num-h5-chunks", default=1, type=int)
    parser.add_argument(
        "--checkpoint",
        help="name of checkpoint file",
        default="results/joint_train_nist20/split_1_rnd1/version_7/best.ckpt",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")
    parser.add_argument(
        "--subset-datasets",
        default="none",
        action="store",
        choices=["none", "train_only", "test_only", "debug_special"],
    )
    parser.add_argument("--binned-out", default=False, action="store_true")
    parser.add_argument(
        "--no-write",
        default=False,
        action="store_true",
        help="Run the full GPU pipeline but skip HDF5 writing (benchmark/profiling only).",
    )
    return parser.parse_args()


def prepare_entry(entry):
    """Turn one labels row into a list of (smi, name, ce, adduct, instrument, ikey) tuples.

    Canonicalizes the SMILES and drops molecules rdkit can't parse, that contain
    wildcard atoms, or that have more than 100 heavy atoms.
    """
    smi = entry["smiles"]
    adduct = entry["ionization"]
    name = entry["spec"]
    instrument = entry["instrument"] if "instrument" in entry else "Orbitrap"
    inchikey = common.inchikey_from_smiles(smi)
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in chem_utils.VALID_ATOM_NUM:
                return []
        mol = Chem.RemoveHs(mol)
        if mol.GetNumAtoms() > 100:
            return []
        smi = Chem.MolToSmiles(mol)  # canonicalize
    except Exception:
        return []
    collision_energies = [i for i in ast.literal_eval(entry["collision_energies"])]
    tup_to_process = []
    for colli_eng in collision_energies:
        colli_eng_val = common.collision_energy_to_float(colli_eng)
        if math.isnan(colli_eng_val):
            continue
        tup_to_process.append((smi, f"pred_{name}", colli_eng_val, adduct, instrument, f"ikey {inchikey}"))
    return tup_to_process


def _prepare_entries_parallel(all_rows, num_workers):
    """Canonicalize/filter/expand label rows into (mol, collision-energy) entries.

    Uses a *fork* pool: workers inherit the parent's already-imported modules, so
    they pay no per-worker re-import. (The project-wide pool forces 'spawn' --
    ctx._force_start_method('spawn') in common.parallel_utils -- which re-imports the
    whole torch/lightning/dgl stack in every worker, turning this trivial step into a
    minutes-long warmup.) Safe because this runs before any CUDA initialization and
    the work is CPU-only.
    """
    if num_workers and num_workers > 1 and len(all_rows) > 1:
        import multiprocessing as _mp

        n = min(num_workers, len(all_rows))
        chunksize = max(1, len(all_rows) // (n * 8))
        with _mp.get_context("fork").Pool(n) as pool:
            nested = pool.map(prepare_entry, all_rows, chunksize=chunksize)
    else:
        nested = [prepare_entry(r) for r in all_rows]
    return [t for sub in nested for t in sub]


class GraphormerFeatDataset(Dataset):
    """Featurizes one prediction entry per item; intended to run in DataLoader workers.

    Each entry is ``(smi, pred_name, collision_energy, adduct, instrument, ikey)``.
    Heavy featurization (``featurize_tree`` -> ``create_graphormer_input``) is cached
    per (smiles, adduct) so repeated collision energies of the same molecule are
    featurized once. Reuses ``IntenDataset.collate_fn`` for tensor batching.
    """

    def __init__(self, entries, tree_processor, sort_for_batching=True, feat_cache_size=8192):
        _lazy_imports()
        self.tree_processor = tree_processor
        self.feat_cache_size = feat_cache_size
        self._feat_cache = OrderedDict()
        if sort_for_batching:
            n_atoms_cache = {}

            def _num_atoms(smi):
                if smi not in n_atoms_cache:
                    m = Chem.MolFromSmiles(smi)
                    n_atoms_cache[smi] = m.GetNumAtoms() if m is not None else 0
                return n_atoms_cache[smi]

            entries = sorted(entries, key=lambda e: (_num_atoms(e[0]), e[0]))
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def _featurize_cached(self, smi, adduct, ce):
        if self.feat_cache_size <= 0:  # dedup disabled (stress-test / all-unique input)
            tree = {"root_canonical_smiles": smi, "adduct": adduct, "collision_energy": ce}
            return self.tree_processor.featurize_tree(
                tree, include_inten_targs=False, include_frag_targs=False
            )
        key = (smi, adduct)
        feat = self._feat_cache.get(key)
        if feat is None:
            tree = {"root_canonical_smiles": smi, "adduct": adduct, "collision_energy": ce}
            feat = self.tree_processor.featurize_tree(
                tree, include_inten_targs=False, include_frag_targs=False
            )
            self._feat_cache[key] = feat
            if self.feat_cache_size and len(self._feat_cache) > self.feat_cache_size:
                self._feat_cache.popitem(last=False)
        else:
            self._feat_cache.move_to_end(key)
        return feat

    def __getitem__(self, idx):
        smi, pred_name, ce, adduct, instrument, ikey = self.entries[idx]
        feat = dict(self._featurize_cached(smi, adduct, ce))
        feat["collision_energy"] = ce
        instrument_pos = common.instrument2onehot_pos.get(
            instrument, common.instrument2onehot_pos.get("Orbitrap", 0)
        )
        item = {
            "name": pred_name,
            "adduct": common.ion2onehot_pos[adduct],
            "precursor": 0.0,
            "instrument": instrument_pos,
            # metadata carried through collate for writing the spectrum:
            "_pred_name": pred_name,
            "_adduct_str": adduct,
            "_ikey": ikey,
            "_ce": ce,
            "_smi": smi,
        }
        item.update(feat)
        return item


def collate_predict(batch):
    _lazy_imports()
    out = glacier_dataset.IntenDataset.collate_fn(batch)
    out["pred_names"] = [b["_pred_name"] for b in batch]
    out["adduct_strs"] = [b["_adduct_str"] for b in batch]
    out["ikeys"] = [b["_ikey"] for b in batch]
    out["ces"] = [b["_ce"] for b in batch]
    out["smis_canon"] = [b["_smi"] for b in batch]
    return out


def _batch_to_device(batch, device):
    for k in batch:
        v = batch[k]
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
        elif k == "graphormer_input" and v is not None:
            for gk in v:
                v[gk] = v[gk].to(device, non_blocking=True)
    return batch


def run_shard(entries, checkpoint, device, out_path, kwargs):
    """Featurize (parallel CPU) + forward (GPU) for one shard, writing a PredSpecDB."""
    assert kwargs["sparse_out"], "sparse_out must be True"
    _lazy_imports()
    sparse_k = kwargs["sparse_k"]
    num_workers = kwargs["num_cpu_workers"]

    # Keep the model on CPU for now -- see the iter()/`.to(device)` ordering below.
    model = joint_model.JointModel.load_from_checkpoint(checkpoint)
    model.eval()
    model.freeze()

    tree_processor = glacier_dataset.TreeProcessor(
        pe_embed_k=model.pe_embed_k,
        root_encode="graphormer",
        embed_elem_group=model.embed_elem_group,
        multi_hop_max_dist=model.multi_hop_max_dist,
    )
    dedup = not kwargs.get("no_dedup", False)
    pred_dataset = GraphormerFeatDataset(
        entries,
        tree_processor=tree_processor,
        # Always size-sort: grouping similar-sized molecules into a batch cuts
        # graphormer padding ~2x (measured 1457 vs 658 rows/s). Independent of dedup.
        sort_for_batching=True,
        # Cache reuses one featurization across a molecule's collision energies.
        feat_cache_size=8192 if dedup else 0,
    )
    loader_kwargs = dict(
        num_workers=num_workers,
        collate_fn=collate_predict,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )
    if num_workers > 0:
        # Fork (not spawn) so workers inherit the already-imported torch/dgl/rdkit/
        # ms_pred stack instead of re-importing it (which costs minutes per pool for
        # this model). Safe only because we fork BEFORE the parent touches CUDA and
        # the workers are CPU-only. pin_memory is left off so its CUDA-allocating
        # thread doesn't initialize CUDA in the parent before the workers fork.
        loader_kwargs["multiprocessing_context"] = "fork"
        loader_kwargs["prefetch_factor"] = kwargs["prefetch_factor"]
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["pin_memory"] = False
    loader = DataLoader(pred_dataset, **loader_kwargs)

    # Create the iterator (forks the CPU-only workers) BEFORE moving the model to the
    # GPU, so CUDA is initialized only in the parent and never inside a forked child.
    data_iter = iter(loader)
    model = model.to(device)

    no_write = kwargs.get("no_write", False)

    # MassSpec construction + HDF5 writing are pure CPU/IO and, done inline, serialize
    # with the GPU forward and starve it. Offload them to a single background thread fed
    # by a bounded queue, so writing batch N overlaps the forward of batch N+1.
    write_q = queue.Queue(maxsize=8)
    writer_error = []

    def writer_loop():
        specdb = None
        try:
            if not no_write:
                specdb = common.PredSpecDB(
                    h5_path=out_path, mode="w", num_h5s=kwargs["num_h5_chunks"],
                    has_probs=False, has_brokens=False, has_masses=True, has_masses_no_adduct=False,
                    has_frag_form_vecs=False, has_frags=True, has_intens=True, has_pulled_atoms=False,
                )
            while True:
                recs = write_q.get()
                if recs is None:
                    break
                if specdb is not None:
                    for name, smi, adduct_str, ce, ikey, masses, intens, frag in recs:
                        specdb.write(name, common.MassSpec(
                            root_canonical_smiles=smi, adduct=adduct_str, collision_energy=ce,
                            masses=masses, intens=intens, frags=frag, remark=ikey,
                        ))
        except Exception as e:  # surface writer failures instead of dying silently
            writer_error.append(e)
            # drain remaining items so the producer never blocks on a full queue
            while True:
                if write_q.get() is None:
                    break
        finally:
            if specdb is not None:
                specdb.close()

    writer_thread = threading.Thread(target=writer_loop, daemon=True)
    writer_thread.start()

    with torch.inference_mode():
        for batch in tqdm(data_iter, total=len(loader), desc=f"{device}"):
            batch = _batch_to_device(batch, device)
            # Vectorized forward + sparse top-k selection: all GPU work + a few host
            # transfers, no per-molecule Python/sync loop on the hot path.
            res = model.predict_inten_frag_batch_sparse(batch, sparse_k)
            masses, intens, frags, counts = res["masses"], res["intens"], res["frags"], res["counts"]
            pred_names = batch["pred_names"]
            adduct_strs = batch["adduct_strs"]
            ikeys = batch["ikeys"]
            ces = batch["ces"]
            smis = batch["smis_canon"]
            recs = []
            for i in range(len(pred_names)):
                c = int(counts[i])
                recs.append((
                    pred_names[i], smis[i], adduct_strs[i], ces[i], ikeys[i],
                    masses[i, :c], intens[i, :c], frags[i, :c],
                ))
            write_q.put(recs)
            if writer_error:
                break
    write_q.put(None)
    writer_thread.join()
    if writer_error:
        raise writer_error[0]


def predict():
    args = get_args()
    kwargs = args.__dict__

    save_dir = Path(kwargs["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    debug = kwargs["debug"]
    common.setup_logger(save_dir, log_name="joint_pred.log", debug=debug)

    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(save_dir / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    data_dir = Path("")
    if kwargs.get("dataset_name") is not None:
        data_dir = Path("data/spec_datasets") / kwargs["dataset_name"]

    labels = Path(kwargs["dataset_labels"])
    df = pd.read_csv(labels, sep="\t")

    if debug:
        df = df[:1000]
        kwargs["num_cpu_workers"] = 0

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
            names = ["CCMSLIB00000001590"]
            kwargs["debug"] = True
        else:
            raise NotImplementedError()
        df = df[df["spec"].isin(names)]

    checkpoint = kwargs["checkpoint"]
    logging.info(f"Loaded joint models from {checkpoint}")

    gpu = kwargs["gpu"]
    avail_gpu_num = torch.cuda.device_count()
    use_gpu = gpu and avail_gpu_num > 0

    # Build flat list of prediction entries (parallel CPU preprocessing).
    all_rows = [j for _, j in df.iterrows()]
    logging.info("Preparing entries")
    predict_entries = _prepare_entries_parallel(all_rows, kwargs["num_cpu_workers"])
    logging.info(f"There are {len(predict_entries)} entries to process")
    if len(predict_entries) == 0:
        logging.warning("No entries to process; exiting.")
        return

    out_name = kwargs["out_name"]
    num_shards = kwargs["num_gpu_workers"] if use_gpu else 0

    if num_shards <= 1:
        device = "cuda:0" if use_gpu else "cpu"
        run_shard(predict_entries, checkpoint, device, save_dir / out_name, kwargs)
    else:
        num_shards = min(num_shards, avail_gpu_num)
        # Shard by molecule so the per-worker featurization cache stays effective
        # (all collision energies of a molecule go to the same shard).
        by_mol = OrderedDict()
        for e in predict_entries:
            by_mol.setdefault(e[0], []).append(e)
        mol_lists = list(by_mol.values())
        shards = [[] for _ in range(num_shards)]
        # Greedy balance by molecule count.
        for i, ml in enumerate(sorted(mol_lists, key=len, reverse=True)):
            shards[i % num_shards].extend(ml)
        out_stem, out_suffix = os.path.splitext(out_name)
        shard_out_paths = [save_dir / f"{out_stem}_shard{r}{out_suffix}" for r in range(num_shards)]
        logging.info(
            f"Sharding {len(predict_entries)} entries across {num_shards} GPUs: "
            f"sizes={[len(s) for s in shards]}"
        )
        mp.spawn(
            _shard_entrypoint,
            args=(shards, checkpoint, shard_out_paths, kwargs),
            nprocs=num_shards,
            join=True,
        )


def _shard_entrypoint(rank, shards, checkpoint, shard_out_paths, kwargs):
    """torch.multiprocessing entrypoint for one GPU shard."""
    device = f"cuda:{rank}"
    common.setup_logger(
        Path(kwargs["save_dir"]), log_name=f"joint_pred_shard{rank}.log", debug=kwargs["debug"]
    )
    run_shard(shards[rank], checkpoint, device, shard_out_paths[rank], kwargs)


if __name__ == "__main__":
    import time

    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
