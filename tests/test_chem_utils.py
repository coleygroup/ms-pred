"""Unit tests for ms_pred.common.chem_utils."""
import re

import pytest
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from ms_pred.common.chem_utils import (
    form_from_smi,
    formula_difference,
    formula_mass,
    formula_to_dense,
    has_valid_els,
    inchikey_from_smiles,
    is_charged,
    mass_from_smi,
    rm_stereo,
    standardize_form,
    vec_to_formula,
)


def _counts(formula: str) -> dict:
    """Parse a chemical formula into a {element: count} dict.

    Used so tests don't depend on element ordering in the output string.
    """
    pairs = re.findall(r"([A-Z][a-z]*)([0-9]*)", formula)
    return {sym: int(n) if n else 1 for sym, n in pairs if sym}


def test_formula_mass_matches_rdkit(known_molecules):
    for _name, (smi, formula, expected) in known_molecules.items():
        internal = formula_mass(formula)
        rdkit_exact = ExactMolWt(Chem.MolFromSmiles(smi))
        assert abs(internal - rdkit_exact) < 0.01
        assert abs(internal - expected) < 0.01


def test_formula_to_dense_vec_to_formula_roundtrip():
    for formula in ["H2O", "CH4", "C6H12O6", "C8H10N4O2", "C9H8O4"]:
        roundtrip = vec_to_formula(formula_to_dense(formula))
        assert _counts(roundtrip) == _counts(formula)


def test_standardize_form_idempotent():
    for formula in ["H2O", "CCH4HO", "C6H12O6"]:
        once = standardize_form(formula)
        assert standardize_form(once) == once


def test_formula_difference_self_is_empty():
    for formula in ["H2O", "C6H12O6", "C8H10N4O2"]:
        diff = formula_difference(formula, formula)
        assert _counts(diff) == {}


def test_formula_difference_removes_subformula():
    diff = formula_difference("H2O", "H2")
    assert _counts(diff) == {"O": 1}


def test_form_from_smi_matches_rdkit_counts(known_molecules):
    for _name, (smi, formula, _mass) in known_molecules.items():
        assert _counts(form_from_smi(smi)) == _counts(formula)


def test_mass_from_smi_matches_formula_mass(known_molecules):
    for _name, (smi, _formula, _mass) in known_molecules.items():
        ours = formula_mass(form_from_smi(smi))
        rdkit_exact = mass_from_smi(smi)
        assert abs(ours - rdkit_exact) < 0.01


def test_has_valid_els():
    assert has_valid_els("C6H12O6")
    assert has_valid_els("C8H10N4O2")
    assert not has_valid_els("Xe2")
    assert not has_valid_els("Hg")


def test_is_charged():
    assert not is_charged("CCO")
    assert is_charged("[NH4+]")
    assert is_charged("[OH-]")


def test_rm_stereo_strips_chirality():
    chiral = "C[C@H](N)C(=O)O"
    stripped = rm_stereo(chiral)
    assert "@" not in stripped


def test_inchikey_from_smiles_deterministic_and_shaped():
    key = inchikey_from_smiles("O")
    assert len(key) == 27
    assert key.count("-") == 2
    # determinism across repeated calls
    assert inchikey_from_smiles("O") == key
    assert inchikey_from_smiles("O") == key


def test_inchikey_distinguishes_molecules():
    assert inchikey_from_smiles("O") != inchikey_from_smiles("CCO")
    assert inchikey_from_smiles("CCO") != inchikey_from_smiles(
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
    )


def test_inchikey_invalid_smiles_returns_empty():
    assert inchikey_from_smiles("not a smiles string") == ""
