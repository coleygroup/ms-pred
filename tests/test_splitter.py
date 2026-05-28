"""Unit tests for ms_pred.common.splitter."""
import numpy as np

from ms_pred.common.splitter import random_split


def _names(n: int) -> list:
    return [f"item_{i}" for i in range(n)]


def test_random_split_sizes_sum_to_input():
    names = _names(100)
    train, val, test = random_split(names, split_sizes=(0.8, 0.1, 0.1))
    assert len(train) + len(val) + len(test) == 100


def test_random_split_partitions_are_disjoint():
    names = _names(100)
    train, val, test = random_split(names, split_sizes=(0.8, 0.1, 0.1))
    train_set, val_set, test_set = set(train), set(val), set(test)
    assert train_set & val_set == set()
    assert train_set & test_set == set()
    assert val_set & test_set == set()


def test_random_split_proportions_within_tolerance():
    names = _names(1000)
    train, val, test = random_split(names, split_sizes=(0.8, 0.1, 0.1))
    assert abs(len(train) / 1000 - 0.8) < 0.02
    assert abs(len(val) / 1000 - 0.1) < 0.02
    assert abs(len(test) / 1000 - 0.1) < 0.02


def test_random_split_seedable_via_numpy():
    names = _names(50)
    np.random.seed(0)
    a = random_split(names, split_sizes=(0.8, 0.1, 0.1))
    np.random.seed(0)
    b = random_split(names, split_sizes=(0.8, 0.1, 0.1))
    for arr_a, arr_b in zip(a, b):
        np.testing.assert_array_equal(arr_a, arr_b)
