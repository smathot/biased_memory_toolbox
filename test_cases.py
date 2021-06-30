# coding=utf-8
"""
This file is part of biased memory toolbox.

biased memory toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

biased memory toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with biased memory toolbox.  If not, see <http://www.gnu.org/licenses/>.
"""
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
    assert math.isclose(b2, 0, abs_tol=1)  # Should have no response bias


def test_precision():
    """Simulate two sets of responses with various levels of noise."""
    dm = DataMatrix(length=N)
    dm.m = np.random.randint(0, 359, N)
    dm.r1 = dm.m + np.random.randint(-5, 5, N)
    dm.r2 = dm.m + np.random.randint(-25, 25, N)
    dm.b1 = bmt.response_bias(dm.m, dm.r1)
    dm.b2 = bmt.response_bias(dm.m, dm.r2)
    p1, _, = bmt.fit_mixture_model(dm.b1, include_bias=False)
    p2, _, = bmt.fit_mixture_model(dm.b2, include_bias=False)
    assert p2 < p1


def test_guess_rate():
    """Simulate two sets of responses with various levels of noise."""
    dm = DataMatrix(length=N)
    dm.m = np.random.randint(0, 359, N)
    dm.r1 = dm.m + np.random.randint(-10, 10, N)
    dm.r2 = dm.m + np.random.randint(-10, 10, N)
    dm.r2[:N // 2] = np.random.randint(0, 359, N // 2)
    dm.b1 = bmt.response_bias(dm.m, dm.r1)
    dm.b2 = bmt.response_bias(dm.m, dm.r2)
    _, gr1 = bmt.fit_mixture_model(dm.b1, include_bias=False)
    assert math.isclose(gr1, 0, abs_tol=.1)
    _, gr2 = bmt.fit_mixture_model(dm.b2, include_bias=False)
    assert math.isclose(gr2, .5, abs_tol=.1)


def test_swap_errors():
    """Simulate three sets of responses with various levels of swap errors."""
    dm = DataMatrix(length=N)
    dm.target = np.random.randint(0, 359, N)
    dm.nontarget1 = np.random.randint(0, 359, N)
    dm.nontarget2 = np.random.randint(0, 359, N)
    dm.r1 = dm.target + np.random.randint(-10, 10, N)
    dm.r2 = dm.target + np.random.randint(-10, 10, N)
    dm.r2[:N // 8] = dm.nontarget1[:N // 8] + np.random.randint(0, 359, N // 8)
    dm.r2[N // 8:N // 4] = dm.nontarget2[N // 8:N // 4] + \
        np.random.randint(0, 359, N // 8)
    dm.r3 = dm.target + np.random.randint(-10, 10, N)
    dm.r3[:N // 4] = dm.nontarget1[:N // 4] + np.random.randint(0, 359, N // 4)
    dm.r3[N // 4:N // 2] = dm.nontarget2[N // 4:N // 2] + \
        np.random.randint(0, 359, N // 4)
    _, _, sr1 = bmt.fit_mixture_model(
        x=bmt.response_bias(dm.target, dm.r1),
        x_nontargets=[
            bmt.response_bias(dm.nontarget1, dm.r1),
            bmt.response_bias(dm.nontarget2, dm.r1)
        ],
        include_bias=False
    )
    assert math.isclose(sr1, 0, abs_tol=.1)
    _, _, sr2 = bmt.fit_mixture_model(
        x=bmt.response_bias(dm.target, dm.r2),
        x_nontargets=[
            bmt.response_bias(dm.nontarget1, dm.r2),
            bmt.response_bias(dm.nontarget2, dm.r2)
        ],
        include_bias=False
    )
    # Swap errors are underestimated, such that 25% swap errors results in an
    # swap-rate parameter of around .125.
    assert math.isclose(sr2, .125, abs_tol=.1)
    _, _, sr3 = bmt.fit_mixture_model(
        x=bmt.response_bias(dm.target, dm.r3),
        x_nontargets=[
            bmt.response_bias(dm.nontarget1, dm.r3),
            bmt.response_bias(dm.nontarget2, dm.r3)
        ],
        include_bias=False
    )
    assert math.isclose(sr3, .25, abs_tol=.1)
