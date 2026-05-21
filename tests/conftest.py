"""Shared pytest fixtures and global setup for the ms-pred test suite."""

import random

import numpy as np
import pytest


def _seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


_seed_all(0)

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass


@pytest.fixture(autouse=True)
def _reseed_each_test():
    _seed_all(0)
    yield


@pytest.fixture
def known_molecules():
    """SMILES, formula, exact monoisotopic mass for a small reference set.

    Masses are RDKit ExactMolWt values rounded to 4 decimals; tests should
    compare with a tolerance, not by exact equality.
    """
    return {
        "water": ("O", "H2O", 18.0106),
        "methane": ("C", "CH4", 16.0313),
        "ethanol": ("CCO", "C2H6O", 46.0419),
        "glucose": ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "C6H12O6", 180.0634),
        "caffeine": ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "C8H10N4O2", 194.0804),
        "aspirin": ("CC(=O)OC1=CC=CC=C1C(=O)O", "C9H8O4", 180.0423),
        "fluorobenzene": ("Fc1ccccc1", "C6H5F", 96.0375),
    }


@pytest.fixture
def tiny_spectrum():
    """A 5-peak spectrum: 2-D array with columns [m/z, intensity]."""
    return np.array(
        [
            [50.0, 0.10],
            [100.0, 0.80],
            [150.0, 0.45],
            [200.0, 1.00],
            [250.0, 0.20],
        ],
        dtype=np.float64,
    )
