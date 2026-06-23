"""Instantiation + CLI smoke tests for ms_pred.iceberg (ICEBERG).

These do not load any checkpoint, data files, or run a real forward pass.
They confirm that the joint-model surface assembles and that the predict
CLI scripts have a working argparse setup. Tier B (synthetic forward) and
Tier C (real-checkpoint inference) live elsewhere.
"""
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from ms_pred.iceberg.gen_model import FragGNN
from ms_pred.iceberg.inten_model import IntenGNN
from ms_pred.iceberg.joint_model import JointModel


def _tiny_gen() -> FragGNN:
    return FragGNN(hidden_size=8, layers=1, set_layers=1)


def _tiny_inten() -> IntenGNN:
    return IntenGNN(hidden_size=8, gnn_layers=1, set_layers=1, frag_set_layers=0)


def test_fraggnn_instantiates():
    model = _tiny_gen()
    assert isinstance(model, FragGNN)
    assert model.hidden_size == 8


def test_intenggnn_instantiates():
    model = _tiny_inten()
    assert isinstance(model, IntenGNN)
    assert model.hidden_size == 8


def test_joint_model_composes_gen_and_inten():
    gen = _tiny_gen()
    inten = _tiny_inten()
    joint = JointModel(gen_model_obj=gen, inten_model_obj=inten)
    assert isinstance(joint, JointModel)
    # The joint module should expose both sub-models.
    assert any(isinstance(m, FragGNN) for m in joint.modules())
    assert any(isinstance(m, IntenGNN) for m in joint.modules())


@pytest.mark.parametrize(
    "module_name",
    [
        "ms_pred.iceberg.predict_gen",
        "ms_pred.iceberg.predict_inten",
        "ms_pred.iceberg.predict_smis",
    ],
)
def test_predict_cli_help(module_name):
    """`python -m <module> --help` parses argparse without crashing."""
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"{module_name} --help exited {result.returncode}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    assert "usage" in result.stdout.lower()
