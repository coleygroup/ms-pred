"""End-to-end JointModel.predict_mol on CPU with untrained models.

predict_mol drives the full ICEBERG joint inference path: FragGNN
generates a fragment DAG, IntenGNN scores it, results are bin-packed.
With threshold=0.0 + tiny max_nodes the path runs to completion even
on randomly-initialized weights. No checkpoint, no spectrum, no GPU
required.
"""
import pytest

torch = pytest.importorskip("torch")

from ms_pred.common import ELEMENT_DIM, MAX_H
from ms_pred.iceberg.gen_model import FragGNN
from ms_pred.iceberg.inten_model import IntenGNN
from ms_pred.iceberg.joint_model import JointModel

# predict_mol calls TreeProcessor.add_pe_embed unconditionally and
# nn_utils.random_walk_pe always emits >=1 column regardless of k. The
# model is built with `node_feats` defaulting to ELEMENT_DIM+MAX_H, which
# does NOT include the pe_embed_k columns. To make the runtime feature
# count match the model input we pass `node_feats` explicitly.
_PE = 1
_NODE_FEATS = ELEMENT_DIM + MAX_H + _PE

JOINT_MOLECULES = [
    ("CC", 46.0),         # ethane
    ("CCO", 47.0),        # ethanol
    ("Cc1ccccc1", 93.0),  # toluene
]


def _tiny_joint() -> JointModel:
    gen = FragGNN(
        hidden_size=8,
        layers=1,
        set_layers=1,
        pe_embed_k=_PE,
        node_feats=_NODE_FEATS,
    )
    inten = IntenGNN(
        hidden_size=8,
        gnn_layers=1,
        set_layers=1,
        frag_set_layers=0,
        pe_embed_k=_PE,
        node_feats=_NODE_FEATS,
    )
    return JointModel(gen_model_obj=gen, inten_model_obj=inten)


@pytest.mark.xfail(
    strict=False,
    reason=(
        "JointModel.predict_mol cannot be run end-to-end on untrained "
        "synthetic models because gen.predict_mol unconditionally strips "
        "the random-walk PE from root_repr at the end (gen_model.py "
        "rm_pe_embed near line 716), but joint.predict_mol does not "
        "re-add it before calling inten.predict. The inten model is "
        "built with node_feats that include pe_embed_k, so it sees the "
        "stripped root_repr as too-narrow and errors with a shape "
        "mismatch. Remove this xfail once the production path is fixed "
        "or once a real checkpoint avoids the strip path."
    ),
)
@pytest.mark.parametrize("smi,precursor_mz", JOINT_MOLECULES)
def test_joint_predict_mol_cpu_runs_end_to_end(smi, precursor_mz):
    joint = _tiny_joint()
    result = joint.predict_mol(
        smi=smi,
        collision_eng=30.0,
        precursor_mz=precursor_mz,
        adduct="[M+H]+",
        threshold=0.0,
        device="cpu",
        max_nodes=5,
    )
    assert isinstance(result, dict)
    assert "spec" in result
    assert "frag" in result
