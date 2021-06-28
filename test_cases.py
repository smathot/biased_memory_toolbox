# coding=utf-8
import math
import pytest
import numpy as np
from datamatrix import DataMatrix
import biased_memory_toolbox as bmt

N = 1000


def test_bias():
    """Simulate one set of noisy responses with a response bias on each trial,
    and one set of noise responses without a response bias.
    """
    dm = DataMatrix(length=N)
    dm.m = np.random.randint(0, 359, N)
    dm.n = np.random.randint(-10, 10, N)
    for row in dm:
        category = bmt.category(row.m, bmt.DEFAULT_CATEGORIES)
        row.r1 = .5 * (row.m + bmt.DEFAULT_CATEGORIES[category][2]) + row.n
        row.r2 = row.m + row.n
    dm.b1 = bmt.response_bias(dm.m, dm.r1, bmt.DEFAULT_CATEGORIES)
    _, _, b1 = bmt.fit_mixture_model(dm.b1)
    assert b1 > 1  # Should have a positive response bias
    dm.b2 = bmt.response_bias(dm.m, dm.r2, bmt.DEFAULT_CATEGORIES)
    _, _, b2 = bmt.fit_mixture_model(dm.b2)
    assert math.isclose(b2, 0, abs_tol=1)  # Should have (almost) no response bias


def test_precision():
    """Simulate two sets of responses with various levels of noise."""
    dm = DataMatrix(length=N)
    dm.m = np.random.randint(0, 359, N)
    dm.r1 = dm.m + np.random.randint(-5, 5, N)
    dm.r2 = dm.m + np.random.randint(-25, 25, N)
    dm.b1 = bmt.response_bias(dm.m, dm.r1, bmt.DEFAULT_CATEGORIES)
    dm.b2 = bmt.response_bias(dm.m, dm.r2, bmt.DEFAULT_CATEGORIES)
    p1, _, _ = bmt.fit_mixture_model(dm.b1)
    p2, _, _ = bmt.fit_mixture_model(dm.b2)
    assert p2 < p1


def test_guess_rate():
    """Simulate two sets of responses with various levels of noise."""
    dm = DataMatrix(length=N)
    dm.m = np.random.randint(0, 359, N)
    dm.r1 = dm.m + np.random.randint(-10, 10, N)
    dm.r2 = dm.m + np.random.randint(-10, 10, N)
    dm.r2[:N // 2] = np.random.randint(0, 359, N // 2)
    dm.b1 = bmt.response_bias(dm.m, dm.r1, bmt.DEFAULT_CATEGORIES)
    dm.b2 = bmt.response_bias(dm.m, dm.r2, bmt.DEFAULT_CATEGORIES)
    _, gr1, _ = bmt.fit_mixture_model(dm.b1)
    assert math.isclose(gr1, 0, abs_tol=.1)
    _, gr2, _ = bmt.fit_mixture_model(dm.b2)
    assert math.isclose(gr2, .5, abs_tol=.1)
