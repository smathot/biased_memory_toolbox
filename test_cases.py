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
import itertools
import pytest
import numpy as np
from datamatrix import DataMatrix
import biased_memory_toolbox as bmt

N = 10000


def _precision_to_scale(precision):
    
    if precision == 500:
        return 20
    elif precision == 2000:
        return 10
    raise ValueError('Invalid precision')


def test_with_bias_and_swap_rate():

    dm = DataMatrix(length=N)
    for precision, guess_rate, bias, swap_rate in itertools.product(
        (500, 2000),
        (0, .25),
        (0, 2.5),
        (0, .25)
    ):
        dm.target = np.random.randint(0, 359, N)
        dm.nontarget1 = dm.target + 180
        dm.responses = dm.target[:]
        n_guess = int(N * guess_rate)
        n_swap = int(N * swap_rate)
        dm.responses[:n_guess] = np.random.randint(0, 359, n_guess)
        dm.responses[n_guess:n_guess +
                     n_swap] = dm.nontarget1[n_guess:n_guess + n_swap]
        dm.responses += np.random.normal(loc=0,
                                         scale=_precision_to_scale(precision),
                                         size=N)
        for row in dm:
            category = bmt.category(row.target, bmt.DEFAULT_CATEGORIES)
            lower, upper, proto = bmt.DEFAULT_CATEGORIES[category]
            if bmt._distance(row.responses, proto) > 0:
                row.responses += bias
            else:
                row.responses -= bias
        p, gr, b, sr = bmt.fit_mixture_model(
            x=bmt.response_bias(
                dm.target,
                dm.responses,
                categories=bmt.DEFAULT_CATEGORIES
            ),
            x_nontargets=[
                bmt.response_bias(
                    dm.nontarget1,
                    dm.responses,
                    categories=bmt.DEFAULT_CATEGORIES
                )
            ]
        )
        assert(math.isclose(precision, p, rel_tol=.25))
        assert(math.isclose(guess_rate, gr, abs_tol=.1))
        assert(math.isclose(bias, b, abs_tol=2))
        assert(math.isclose(swap_rate, sr, abs_tol=.1))


def test_with_bias():

    dm = DataMatrix(length=N)
    for precision, guess_rate, bias in itertools.product(
        (500, 2000),
        (0, .25),
        (0, 2.5)
    ):
        dm.target = np.random.randint(0, 359, N)
        dm.responses = dm.target[:]
        n_guess = int(N * guess_rate)
        dm.responses[:n_guess] = np.random.randint(0, 359, n_guess)
        dm.responses += np.random.normal(loc=0,
                                         scale=_precision_to_scale(precision),
                                         size=N)
        for row in dm:
            category = bmt.category(row.target, bmt.DEFAULT_CATEGORIES)
            lower, upper, proto = bmt.DEFAULT_CATEGORIES[category]
            if bmt._distance(row.responses, proto) > 0:
                row.responses += bias
            else:
                row.responses -= bias
        p, gr, b = bmt.fit_mixture_model(
            x=bmt.response_bias(
                dm.target,
                dm.responses,
                categories=bmt.DEFAULT_CATEGORIES
            )
        )
        assert(math.isclose(precision, p, rel_tol=.25))
        assert(math.isclose(guess_rate, gr, abs_tol=.1))
        assert(math.isclose(bias, b, abs_tol=2))


def test_with_swap_rate():

    dm = DataMatrix(length=N)
    for precision, guess_rate, swap_rate in itertools.product(
        (500, 2000),
        (0, .25),
        (0, .25)
    ):
        dm.target = np.random.randint(0, 359, N)
        dm.nontarget1 = dm.target + 180
        dm.responses = dm.target[:]
        n_guess = int(N * guess_rate)
        n_swap = int(N * swap_rate)
        dm.responses[:n_guess] = np.random.randint(0, 359, n_guess)
        dm.responses[n_guess:n_guess +
                     n_swap] = dm.nontarget1[n_guess:n_guess + n_swap]
        dm.responses += np.random.normal(loc=0,
                                         scale=_precision_to_scale(precision),
                                         size=N)
        p, gr, sr = bmt.fit_mixture_model(
            x=bmt.response_bias(dm.target, dm.responses),
            x_nontargets=[bmt.response_bias(dm.nontarget1, dm.responses)],
            include_bias=False
        )
        assert(math.isclose(precision, p, rel_tol=.25))
        assert(math.isclose(guess_rate, gr, abs_tol=2))
        assert(math.isclose(swap_rate, sr, abs_tol=.1))


def test_basic():

    dm = DataMatrix(length=N)
    for precision, guess_rate in itertools.product(
        (500, 2000),
        (0, .25)
    ):
        dm.target = np.random.randint(0, 359, N)
        dm.responses = dm.target[:]
        n_guess = int(N * guess_rate)
        dm.responses[:n_guess] = np.random.randint(0, 359, n_guess)
        dm.responses += np.random.normal(loc=0,
                                         scale=_precision_to_scale(precision),
                                         size=N)
        p, gr, = bmt.fit_mixture_model(
            x=bmt.response_bias(dm.target, dm.responses),
            include_bias=False
        )
        assert(math.isclose(precision, p, rel_tol=.25))
        assert(math.isclose(guess_rate, gr, abs_tol=2))
