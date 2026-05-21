"""Smoke tests: confirm every subpackage's anchor module imports cleanly.

A failure here usually means a dependency rename, a top-level NameError, or
that one branch's refactor broke the import graph.
"""

import importlib

import pytest

ANCHOR_MODULES = [
    "ms_pred",
    "ms_pred.common",
    "ms_pred.common.chem_utils",
    "ms_pred.common.misc_utils",
    "ms_pred.common.fingerprint",
    "ms_pred.common.splitter",
    "ms_pred.common.parallel_utils",
    "ms_pred.common.plot_utils",
    "ms_pred.common.denoising_utils",
    "ms_pred.nn_utils",
    "ms_pred.nn_utils.form_embedder",
    "ms_pred.massformer_pred",
    "ms_pred.dag_pred.dag_data",
    "ms_pred.ffn_pred.ffn_data",
    "ms_pred.gnn_pred.gnn_data",
    "ms_pred.scarf_pred.scarf_data",
    "ms_pred.marason.dag_data",
    "ms_pred.magma.fragmentation",
    "ms_pred.retrieval.bootstrap_metrics",
    "ms_pred.autoregr_gen.autoregr_data",
    "ms_pred.graff_ms.graff_ms_data",
    "ms_pred.molnetms.molnetms_data",
]


@pytest.mark.parametrize("module_name", ANCHOR_MODULES)
def test_import_anchor_module(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None
