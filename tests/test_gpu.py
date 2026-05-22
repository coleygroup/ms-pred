"""GPU parity tests.

Marked with `@pytest.mark.gpu` so they are skipped in the default (CPU) CI
run. Invoke on an Engaging GPU node via:

    make test-gpu
    bash tests/run_gpu.sh

These confirm a CUDA device is visible to the process and that a basic
compute kernel produces the same result on GPU as on CPU within float32
tolerance. Model-level GPU parity tests can land alongside their CPU
counterparts in later PRs.
"""

import pytest


torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")
    return torch.device("cuda:0")


def test_cuda_visible(cuda_device):
    assert torch.cuda.device_count() >= 1
    name = torch.cuda.get_device_name(cuda_device)
    assert isinstance(name, str) and len(name) > 0


def test_matmul_matches_cpu(cuda_device):
    torch.manual_seed(0)
    a = torch.randn(64, 32)
    b = torch.randn(32, 16)
    cpu_out = a @ b
    gpu_out = (a.to(cuda_device) @ b.to(cuda_device)).cpu()
    torch.testing.assert_close(cpu_out, gpu_out, rtol=1e-4, atol=1e-4)


def test_tensor_roundtrip(cuda_device):
    x = torch.arange(128, dtype=torch.float32)
    y = x.to(cuda_device).contiguous().cpu()
    assert torch.equal(x, y)


def test_fourier_featurizer_gpu_matches_cpu(cuda_device):
    """Parity check: the same module forwards the same input identically
    on CPU and GPU within float32 tolerance."""
    from ms_pred.nn_utils.form_embedder import FourierFeaturizer

    model = FourierFeaturizer()
    counts = torch.tensor([0, 1, 5, 10, 100, 200], dtype=torch.long)
    cpu_out = model(counts)
    gpu_out = model.to(cuda_device)(counts.to(cuda_device)).cpu()
    torch.testing.assert_close(cpu_out, gpu_out, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(gpu_out).all()


# ---------------------------------------------------------------------------
# dag_pred GPU forward tests. Mirror tests/test_dag_forward.py and
# tests/test_inten_model.py but move the model + batch to CUDA. Catch GPU-
# specific dgl / torch-scatter regressions and any CPU<->GPU divergence
# in the joint pipeline.
# ---------------------------------------------------------------------------


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    if hasattr(obj, "to"):
        try:
            return obj.to(device)
        except (TypeError, ValueError, AttributeError):
            return obj
    return obj


def _frag_batch_for(smi: str, max_depth: int):
    from ms_pred.dag_pred.dag_data import GenDataset, TreeProcessor

    from tests._dag_helpers import build_sample_gen, build_tree

    tree = build_tree(smi, max_tree_depth=max_depth)
    processor = TreeProcessor(pe_embed_k=0, root_encode="gnn")
    sample = build_sample_gen(tree, processor)
    return GenDataset.collate_fn([sample])


def _inten_batch_for(smi: str, max_depth: int):
    from ms_pred.dag_pred.dag_data import IntenDataset, TreeProcessor

    from tests._dag_helpers import build_sample_inten, build_tree

    tree = build_tree(smi, max_tree_depth=max_depth)
    processor = TreeProcessor(pe_embed_k=0, root_encode="gnn")
    sample = build_sample_inten(tree, processor)
    return IntenDataset.collate_fn([sample])


def _frag_gnn_forward(model, batch):
    return model(
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


def _inten_gnn_forward(model, batch):
    return model(
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


def _flatten_tensors(out):
    if torch.is_tensor(out):
        yield out
    elif isinstance(out, dict):
        for v in out.values():
            yield from _flatten_tensors(v)
    elif isinstance(out, (list, tuple)):
        for v in out:
            yield from _flatten_tensors(v)


@pytest.mark.parametrize(
    "smi,max_depth",
    [
        ("CC", 3),
        ("CCO", 3),
        ("Cc1ccccc1", 2),
    ],
)
def test_frag_gnn_gpu_forward(cuda_device, smi, max_depth):
    from ms_pred.dag_pred.gen_model import FragGNN

    torch.manual_seed(0)
    model = FragGNN(hidden_size=8, layers=1, set_layers=1, pe_embed_k=0)
    model.eval().to(cuda_device)
    batch = _to_device(_frag_batch_for(smi, max_depth), cuda_device)
    with torch.no_grad():
        out = _frag_gnn_forward(model, batch)
    for t in _flatten_tensors(out):
        assert t.device.type == "cuda"
        assert torch.isfinite(t).all()


@pytest.mark.parametrize(
    "smi,max_depth",
    [
        ("CC", 3),
        ("CCO", 3),
        ("Cc1ccccc1", 2),
    ],
)
def test_inten_gnn_gpu_forward(cuda_device, smi, max_depth):
    from ms_pred.dag_pred.inten_model import IntenGNN

    torch.manual_seed(0)
    model = IntenGNN(
        hidden_size=8,
        gnn_layers=1,
        set_layers=1,
        frag_set_layers=0,
        pe_embed_k=0,
    )
    model.eval().to(cuda_device)
    batch = _to_device(_inten_batch_for(smi, max_depth), cuda_device)
    with torch.no_grad():
        out = _inten_gnn_forward(model, batch)
    for t in _flatten_tensors(out):
        assert t.device.type == "cuda"
        assert torch.isfinite(t).all()


@pytest.mark.xfail(
    strict=False,
    reason=(
        "See test_joint_model.test_joint_predict_mol_cpu_runs_end_to_end. "
        "gen.predict_mol strips random-walk PE from root_repr; "
        "joint.predict_mol doesn't re-add it before calling inten.predict."
    ),
)
@pytest.mark.parametrize(
    "smi,precursor_mz",
    [
        ("CC", 46.0),
        ("CCO", 47.0),
        ("Cc1ccccc1", 93.0),
    ],
)
def test_joint_predict_mol_gpu(cuda_device, smi, precursor_mz):
    from ms_pred.dag_pred.gen_model import FragGNN
    from ms_pred.dag_pred.inten_model import IntenGNN
    from ms_pred.dag_pred.joint_model import JointModel

    from ms_pred.common import ELEMENT_DIM, MAX_H

    # pe_embed_k>=1 + matching node_feats: predict_mol calls add_pe_embed
    # unconditionally and random_walk_pe always emits >=1 column. See
    # test_joint_model._tiny_joint for the rationale.
    pe = 1
    node_feats = ELEMENT_DIM + MAX_H + pe
    torch.manual_seed(0)
    gen = FragGNN(
        hidden_size=8,
        layers=1,
        set_layers=1,
        pe_embed_k=pe,
        node_feats=node_feats,
    )
    inten = IntenGNN(
        hidden_size=8,
        gnn_layers=1,
        set_layers=1,
        frag_set_layers=0,
        pe_embed_k=pe,
        node_feats=node_feats,
    )
    joint = JointModel(gen_model_obj=gen, inten_model_obj=inten).to(cuda_device)
    result = joint.predict_mol(
        smi=smi,
        collision_eng=30.0,
        precursor_mz=precursor_mz,
        adduct="[M+H]+",
        threshold=0.0,
        device=str(cuda_device),
        max_nodes=5,
    )
    assert isinstance(result, dict)
    assert "spec" in result
    assert "frag" in result
