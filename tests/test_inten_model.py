"""Synthetic CPU forward through dag_pred IntenGNN (Tier B, inten side).

Mirror of test_dag_forward.py for the intensity-prediction half of the
joint pipeline. Uses TreeProcessor.process_tree_inten_pred so no real
spectrum (raw_spec) or inten target is needed.
"""
import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")

from ms_pred.dag_pred.dag_data import IntenDataset, TreeProcessor
from ms_pred.dag_pred.inten_model import IntenGNN

from tests._dag_helpers import TEST_MOLECULES, build_sample_inten, build_tree


@pytest.mark.parametrize("smi,max_depth", TEST_MOLECULES)
def test_synthetic_tree_processes_through_inten_processor(smi, max_depth):
    tree = build_tree(smi, max_tree_depth=max_depth)
    processor = TreeProcessor()
    out = processor.process_tree_inten_pred(tree)
    assert "dgl_tree" in out
    dgl_tree = out["dgl_tree"]
    expected_keys = {
        "root_repr",
        "dgl_frags",
        "masses",
        "max_remove_hs",
        "max_add_hs",
        "max_broken",
        "form_vecs",
        "root_form_vec",
        "collision_energy",
    }
    assert expected_keys.issubset(set(dgl_tree.keys()))
    assert len(dgl_tree["dgl_frags"]) >= 1
    assert isinstance(dgl_tree["dgl_frags"][0], dgl.DGLGraph)


@pytest.mark.parametrize("smi,max_depth", TEST_MOLECULES)
def test_inten_model_forward_on_synthetic_batch(smi, max_depth):
    tree = build_tree(smi, max_tree_depth=max_depth)
    processor = TreeProcessor(pe_embed_k=0, root_encode="gnn")
    sample = build_sample_inten(tree, processor)

    batch = IntenDataset.collate_fn([sample])
    assert batch["frag_graphs"].num_nodes() > 0

    model = IntenGNN(
        hidden_size=8,
        gnn_layers=1,
        set_layers=1,
        frag_set_layers=0,
        pe_embed_k=0,
    )
    model.eval()
    with torch.no_grad():
        out = model(
            graphs=batch["frag_graphs"],
            root_repr=batch["root_reprs"],
            ind_maps=batch["inds"],
            num_frags=batch["num_frags"],
            broken=batch["broken_bonds"],
            collision_engs=batch["collision_engs"],
            precursor_mzs=batch["precursor_mzs"],
            adducts=batch.get("adducts"),
            instruments=batch.get("instruments"),
            max_add_hs=batch["max_add_hs"],
            max_remove_hs=batch["max_remove_hs"],
            masses=batch["masses"],
            root_forms=batch["root_form_vecs"],
            frag_forms=batch["frag_form_vecs"],
        )
    if isinstance(out, dict):
        for k, v in out.items():
            if torch.is_tensor(v):
                assert torch.isfinite(v).all(), f"NaN/Inf in output[{k}]"
    elif torch.is_tensor(out):
        assert torch.isfinite(out).all()
