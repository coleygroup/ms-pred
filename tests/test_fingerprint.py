"""Unit tests for ms_pred.common.fingerprint."""
import numpy as np

from ms_pred.common.fingerprint import get_morgan_fp_inchi, get_morgan_fp_smi


def test_morgan_fp_shape_and_dtype():
    fp = get_morgan_fp_smi("CCO")
    assert fp.shape == (2048,)
    assert fp.dtype == np.uint8


def test_morgan_fp_boolean_flag():
    fp = get_morgan_fp_smi("CCO", isbool=True)
    assert fp.shape == (2048,)
    assert fp.dtype == np.bool_


def test_morgan_fp_deterministic():
    fp1 = get_morgan_fp_smi("CCO")
    fp2 = get_morgan_fp_smi("CCO")
    np.testing.assert_array_equal(fp1, fp2)


def test_morgan_fp_distinct_inputs_distinct_outputs():
    fp_water = get_morgan_fp_smi("O")
    fp_caffeine = get_morgan_fp_smi("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")
    assert not np.array_equal(fp_water, fp_caffeine)


def test_morgan_fp_smi_matches_inchi_for_same_molecule():
    smi_fp = get_morgan_fp_smi("CCO")
    inchi_fp = get_morgan_fp_inchi("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
    np.testing.assert_array_equal(smi_fp, inchi_fp)
