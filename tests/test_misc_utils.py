"""Unit tests for ms_pred.common.misc_utils."""
import numpy as np
import pytest

from ms_pred.common.misc_utils import (
    batches,
    digitize_ar,
    ev_to_nce,
    is_iterable,
    max_inten_spec,
    nce_to_ev,
    norm_spectrum,
    str_to_hash,
)


def test_norm_spectrum_max_is_one():
    spec = np.array([[0.0, 2.0, 4.0], [0.0, 0.0, 0.0]])
    normed = norm_spectrum(spec)
    # Non-zero rows: max-normalized then sqrt-transformed -> max == 1.
    assert pytest.approx(normed[0].max(), abs=1e-9) == 1.0
    # All-zero rows stay zero.
    assert (normed[1] == 0).all()


def test_digitize_ar_bins_evenly_spaced():
    x = np.array([0.0, 50.0, 150.0, 1499.0])
    bins = digitize_ar(x, num_bins=15000, upper_limit=1500)
    assert bins.shape == (4,)
    assert (bins >= 0).all()
    # monotonic for sorted input
    assert (np.diff(bins) >= 0).all()


def test_batches_chunks_correctly():
    out = list(batches(range(10), 3))
    assert out == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_batches_exact_division():
    assert list(batches(range(6), 3)) == [[0, 1, 2], [3, 4, 5]]


def test_batches_empty_input():
    assert list(batches([], 3)) == []


def test_nce_ev_reversible_float():
    for nce, mz in [(20.0, 300.0), (35.0, 500.0), (60.0, 1000.0)]:
        ev = nce_to_ev(nce, mz)
        recovered = ev_to_nce(ev, mz)
        assert pytest.approx(recovered, rel=1e-6) == nce


def test_nce_ev_known_value():
    # nce_to_ev: ev = nce * mz / 500
    assert pytest.approx(nce_to_ev(20.0, 500.0), rel=1e-9) == 20.0
    assert pytest.approx(nce_to_ev(40.0, 500.0), rel=1e-9) == 40.0
    assert pytest.approx(nce_to_ev(20.0, 250.0), rel=1e-9) == 10.0


def test_str_to_hash_deterministic_and_distinct():
    a = str_to_hash("hello")
    assert a == str_to_hash("hello")
    assert a != str_to_hash("hello!")
    assert len(a) == 32  # default digest_size=16 -> 32 hex chars


def test_is_iterable():
    assert is_iterable([1, 2])
    assert is_iterable((1, 2))
    assert is_iterable(range(3))
    assert is_iterable("abc")
    assert not is_iterable(5)
    assert not is_iterable(None)


def test_max_inten_spec_keeps_top_n(tiny_spectrum):
    out = max_inten_spec(tiny_spectrum, max_num_inten=3, inten_thresh=0)
    assert out.shape == (3, 2)
    # Output sorted by intensity descending
    assert (np.diff(out[:, 1]) <= 0).all()
    # Top three intensities of tiny_spectrum are 1.00, 0.80, 0.45
    assert pytest.approx(sorted(out[:, 1].tolist(), reverse=True)[:3]) == [
        1.0,
        0.8,
        0.45,
    ]


def test_max_inten_spec_respects_threshold(tiny_spectrum):
    out = max_inten_spec(tiny_spectrum, max_num_inten=10, inten_thresh=0.5)
    assert out.shape[0] == 2  # only 1.00 and 0.80 exceed 0.5
    assert (out[:, 1] > 0.5).all()
