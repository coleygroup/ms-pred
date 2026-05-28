"""Helpers shared by Tier B synthetic forward tests for dag_pred.

Build a magma-style tree dict by hand from FragmentEngine, plus per-task
sample dicts suitable for the gen / inten collate functions. Avoids the
need for a cached magma HDF5 or a real spectrum.
"""
from typing import Optional

import numpy as np

from ms_pred.common import instrument2onehot_pos, ion2onehot_pos
from ms_pred.dag_pred.dag_data import TreeProcessor
from ms_pred.magma.fragmentation import FRAGMENT_ENGINE_PARAMS, FragmentEngine

# Small set of SMILES that exercises ethane (trivial), heteroatom +
# alcohol, and an aromatic ring with an alkyl substituent. Depth is
# capped per molecule so DAG generation stays bounded; deeper DAGs blow
# up combinatorially on rings.
TEST_MOLECULES = [
    ("CC", 3),               # ethane
    ("CCO", 3),              # ethanol (heteroatom)
    ("Cc1ccccc1", 2),        # toluene (aromatic ring + alkyl)
]


def build_tree(
    smi: str,
    adduct: str = "[M+H]+",
    max_tree_depth: Optional[int] = None,
    max_broken_bonds: Optional[int] = None,
) -> dict:
    """Synthesize the magma tree dict that _convert_to_dgl expects.

    Reuses FragmentEngine's frag_to_entry. atoms_pulled (which magma
    augmentation normally derives from peak matching) is approximated as
    the first parent_ind_removed; sufficient for a forward-only test.
    All depths are included; the processor's `last_row` flag picks what
    to keep per task.
    """
    kwargs = {}
    if max_tree_depth is not None:
        kwargs["max_tree_depth"] = max_tree_depth
    if max_broken_bonds is not None:
        kwargs["max_broken_bonds"] = max_broken_bonds
    fe = FragmentEngine(mol_str=smi, **kwargs)
    fe.generate_fragments()

    frags = {}
    for idx, entry in enumerate(fe.frag_to_entry.values()):
        parent_removed = entry.get("parent_ind_removed") or []
        atoms_pulled = [int(parent_removed[0])] if parent_removed else []
        frags[str(idx)] = {
            "max_broken": int(entry["max_broken"]),
            "tree_depth": int(entry["tree_depth"]),
            "atoms_pulled": atoms_pulled,
            "frag": int(entry["frag"]),
            "max_remove_hs": int(entry.get("max_remove_hs", 0)),
            "max_add_hs": int(entry.get("max_add_hs", 0)),
            "base_mass": float(entry.get("base_mass", 0.0)),
        }
    return {
        "root_canonical_smiles": fe.smiles,
        "adduct": adduct,
        "frags": frags,
        "collision_energy": 30.0,
    }


def _sample_meta(tree: dict, name: str, instrument: str = "Orbitrap") -> dict:
    return {
        "name": name,
        "adduct": ion2onehot_pos[tree["adduct"]],
        "precursor": 100.0,
        "instrument": instrument2onehot_pos[instrument],
    }


def build_sample_gen(
    tree: dict,
    processor: Optional[TreeProcessor] = None,
    name: str = "synthetic_gen_0",
) -> dict:
    """Mirror GenDataset.__getitem__ output for one synthetic sample."""
    proc = processor or TreeProcessor()
    dgl_tree = proc.process_tree_gen(tree)["dgl_tree"]
    sample = _sample_meta(tree, name)
    sample.update(dgl_tree)
    return sample


def build_sample_inten(
    tree: dict,
    processor: Optional[TreeProcessor] = None,
    name: str = "synthetic_inten_0",
) -> dict:
    """Mirror IntenDataset.__getitem__ output for one synthetic sample.

    Uses process_tree_inten_pred so no real spectrum / inten target is
    needed.
    """
    proc = processor or TreeProcessor()
    dgl_tree = proc.process_tree_inten_pred(tree)["dgl_tree"]
    sample = _sample_meta(tree, name)
    sample.update(dgl_tree)
    return sample
