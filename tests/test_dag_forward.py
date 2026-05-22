"""Synthetic CPU forward through dag_pred FragGNN (Tier B, gen side).

See tests/_dag_helpers.py for the tree synthesis utilities. Catches
shape / API regressions in the gen inference path without a checkpoint
or cached HDF5.
"""
import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")

from ms_pred.dag_pred.dag_data import GenDataset, TreeProcessor
from ms_pred.dag_pred.gen_model import FragGNN

from tests._dag_helpers import TEST_MOLECULES, build_sample_gen, build_tree


@pytest.mark.parametrize("smi,max_depth", TEST_MOLECULES)
def test_synthetic_tree_processes_through_tree_processor(smi, max_depth):
    tree = build_tree(smi, max_tree_depth=max_depth)
    processor = TreeProcessor()
    out = processor.process_tree_gen(tree)
    assert "dgl_tree" in out
    dgl_tree = out["dgl_tree"]
    expected_keys = {
        "root_repr",
        "dgl_frags",
        "targs",
        "max_broken",
        "form_vecs",
        "root_form_vec",
        "collision_energy",
    }
    assert expected_keys.issubset(set(dgl_tree.keys()))
    assert len(dgl_tree["dgl_frags"]) >= 1
    assert isinstance(dgl_tree["dgl_frags"][0], dgl.DGLGraph)


@pytest.mark.parametrize("smi,max_depth", TEST_MOLECULES)
def test_gen_model_forward_on_synthetic_batch(smi, max_depth):
    tree = build_tree(smi, max_tree_depth=max_depth)
    processor = TreeProcessor(pe_embed_k=0, root_encode="gnn")
    sample = build_sample_gen(tree, processor)

    batch = GenDataset.collate_fn([sample])
    assert batch["frag_graphs"].num_nodes() > 0

    model = FragGNN(hidden_size=8, layers=1, set_layers=1, pe_embed_k=0)
    model.eval()
    with torch.no_grad():
        out = model(
            graphs=batch["frag_graphs"],
            root_repr=batch["root_reprs"],
            ind_maps=batch["inds"],
            broken=batch["broken_bonds"],
            collision_engs=batch["collision_engs"],
            precursor_mzs=batch["precursor_mzs"],
            adducts=batch.get("adducts"),
            instruments=batch.get("instruments"),
            root_forms=batch["root_form_vecs"],
            frag_forms=batch["frag_form_vecs"],
        )
    if isinstance(out, dict):
        for k, v in out.items():
            if torch.is_tensor(v):
                assert torch.isfinite(v).all(), f"NaN/Inf in output[{k}]"
    elif torch.is_tensor(out):
        assert torch.isfinite(out).all()
