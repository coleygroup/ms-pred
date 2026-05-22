"""CPU forward-pass tests for ms_pred.nn_utils layers.

These do not load any checkpoint or training data. They confirm a layer
instantiates, forwards a small tensor, and returns finite values with the
expected shape.
"""
import pytest

torch = pytest.importorskip("torch")

from ms_pred.nn_utils.form_embedder import FourierFeaturizer


def test_fourier_featurizer_instantiates():
    model = FourierFeaturizer()
    # MAX_COUNT_INT = 255 -> num_freqs = ceil(log2(255)) + 2 = 10
    # embedding_dim = 2 * num_freqs = 20  (sin + cos)
    assert model.embedding_dim == 20


def test_fourier_featurizer_forward_shape_and_finiteness():
    model = FourierFeaturizer()
    x = torch.zeros(2, 5, dtype=torch.long)
    out = model(x)
    # Per forward(): reshape((*orig_shape[:-1], -1)) -> (2, 5 * 20)
    assert out.shape == (2, 5 * model.embedding_dim)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_fourier_featurizer_forward_1d_input():
    model = FourierFeaturizer()
    x = torch.tensor([0, 1, 5, 10, 100, 200], dtype=torch.long)
    out = model(x)
    assert out.shape == (6 * model.embedding_dim,)
    assert torch.isfinite(out).all()


def test_fourier_featurizer_deterministic():
    model = FourierFeaturizer()
    x = torch.tensor([0, 1, 5, 10, 100, 200], dtype=torch.long)
    out1 = model(x)
    out2 = model(x)
    torch.testing.assert_close(out1, out2)


def test_fourier_featurizer_distinct_inputs_distinct_outputs():
    model = FourierFeaturizer()
    out_small = model(torch.tensor([0, 1, 2], dtype=torch.long))
    out_large = model(torch.tensor([100, 50, 25], dtype=torch.long))
    assert not torch.allclose(out_small, out_large)
