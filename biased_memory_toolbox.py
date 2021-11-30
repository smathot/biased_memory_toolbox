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

import sys
import warnings
import random
import numpy as np
from scipy.stats import vonmises, uniform, ttest_ind
from scipy import optimize

__version__ = '1.2.3'

# These default categories have been established in a separate validation
# experiment. Each tuple indicates a start_value, end_value, and prototype.
# values are hues in 0 - 360 in HSV space such that 0 is red.
DEFAULT_CATEGORIES = {
    'red': (-19, 16, -1.5),
    'orange': (16, 47, 31),
    'yellow': (47, 75, 61),
    'green': (75, 167, 121),
    'blue': (167, 261, 214),
    'purple': (261, 295, 278),
    'pink': (295, 341, 318),
}

# Starting parameters of the fit
STARTING_PRECISION = 500
STARTING_GUESS_RATE = .1
STARTING_BIAS = 0
STARTING_SWAP_RATE = .1
# Realistic bounds for each parameter
BOUNDS_PRECISION = 0, 10000
BOUNDS_GUESS_RATE = 0, 1
BOUNDS_BIAS = -180, 180
BOUNDS_SWAP_RATE = 0, .5
BOUNDS_GUESS_RATE_WITH_SWAP = 0, .5  # guess + swap rate <= 1


def mixture_model_pdf(x, precision=STARTING_PRECISION,
                      guess_rate=STARTING_GUESS_RATE, bias=STARTING_BIAS):
    """Returns a probability density function for a mixture model.
    
    Parameters
    ----------
    x : A list (or other iterable object) of values for the x axis. For example
        `range(-180, 181)` would generate the PDF for every relevant value.
    precision: The precision (or kappa) parameter. This is inversely related to
               the standard deviation, and is a value in degrees.
    guess_rate: The proportion of guess responses (0 - 1).
    bias: The bias (or loc) parameter in degrees.
    
    Returns
    -------
    An array with probability densities for each value of x.
    """
    
    x = np.radians(x)
    pdf_vonmises = vonmises.pdf(
        x=x,
        kappa=np.radians(precision),
        loc=np.radians(bias)
    )
    pdf_uniform = uniform.pdf(x, loc=-np.pi, scale=2*np.pi)
    return pdf_vonmises * (1 - guess_rate) + pdf_uniform * guess_rate


def fit_mixture_model(x, x_nontargets=None,
                      include_bias=True, x0=None, bounds=None):
    """Fits the biased mixture model to a dataset. The input to the mixture
    model should generally be a response bias as determined by
    `response_bias()` when the bias parameter is fit, or a signed response
    error when no bias parameter is fit.
    
    Parameters
    ----------
    x : A DataMatrix column (or other iterable object) of response biases
    x_nontargets : A list of DataMatrix columns (or other iterable objects) of
                   response biases relative to non-targets. If this argument is
                   provided, a swap rate is returned as a final parameter.
    include_bias : Indicates whether the bias parameter should be fit as well.
    x0 : A list of starting values for the parameters. Order: precision, guess
         rate, bias. If no starting value is provided for a parameter, then it
         is left at the default value of `mixture_model_pdf()`.
    bounds : A list of (upper, lower) bound tuples for the parameters. If no
             value is provided, then default values are used.
    
    Returns
    -------
    A tuple with parameters. Depending on the arguments these are:
    - precision, guess rate
    - precision, guess rate, bias
    - precision, guess rate, swap rate
    - precision, guess rate, bias, swap rate
    """
    
    if x_nontargets is not None:
        return _fit_swap_model(
            x, x_nontargets, include_bias=include_bias, x0=x0, bounds=bounds)
    if x0 is None:
        x0 = [STARTING_PRECISION, STARTING_GUESS_RATE]
        if include_bias:
            x0.append(STARTING_BIAS)
    if bounds is None:
        bounds = [BOUNDS_PRECISION, BOUNDS_GUESS_RATE]
        if include_bias:
            bounds.append(BOUNDS_BIAS)
    fit = optimize.minimize(_error, x0=x0, args=x, bounds=bounds)
    return tuple(fit.x)


def category(x, categories):
    
    """Gets the category to which x belongs. For example, if x corresponds to a
    slightly orangy shade of red, then the category would be 'red'.
    
    Parameters
    ----------
    x : A value in degrees (0 - 360)
    categories : See reponse_bias()
    
    Returns
    -------
    A category description.
    """
    
    for category, (minval, maxval, proto) in categories.items():
        if minval <= x < maxval:
            return category
        if minval <= x - 360 < maxval:
            return category
        if minval <= x + 360 < maxval:
            return category
    raise ValueError('{} has no category'.format(x))


def prototype(x, categories):
    
    """Gets the prototype for the category to which x belongs. For example, if
    x corresponds to a slightly orangy shade of red, then the prototype would
    be the hue of a prototypical shade of red.
    
    Parameters
    ----------
    x : A value in degrees (0 - 360)
    categories : See reponse_bias()
    
    Returns
    -------
    A prototype value in degrees (0 360).
    """
       
    return categories[category(x, categories)][2]
    
    
def response_bias(memoranda, responses, categories=None):
    
    """Calculates the response bias, which is the error between a response and
    a memorandum in the direction of the prototype for the category to which
    the memorandum belongs. For example, if the memorandum was an orangy shade
    of red, then a positive value would indicate an error towards a
    prototypical red, and a negative value would indicate an error towards the
    yellow category.
    
    Parameters
    ----------
    memoranda : A DataMatrix column with memoranda values in degrees (0 - 360)
    responses : A DataMatrix column with response values in degrees (0 - 360)
    categories : A dict that defines the categories. Keys are names of
                 categories and values are (start_value, end_value, prototype) 
                 values that indicate where categories begin and end, and what
                 the prototypical value is. The start_value and prototpe can be
                 negative and should be smaller than the end value.

    Returns
    -------
    A list of response_bias values.
    """
    
    errors = _distance(memoranda, responses)
    if categories is None:
        return errors
    cache = {}
    protos = np.empty(len(memoranda))
    for i, memorandum in enumerate(memoranda):
        if memorandum not in cache:
            cache[memorandum] = prototype(memorandum, categories)
        protos[i] = cache[memorandum]
    proto_dists = _distance(memoranda, protos)
    bias = []
    for error, proto_dist in zip(errors, proto_dists):
        if error * proto_dist < 0:
            bias.append(-abs(error))
        else:
            bias.append(abs(error))
    return bias


def test_chance_performance(memoranda, responses):
    
    """Tests whether responses are above chance. This is done by first
    determining the real error and the memoranda, and then determinining the
    shuffled error between the memoranda and the shuffled responses. Finally,
    an independent t-test is done to compare the real and shuffled error. The
    exact values will vary because the shuffling is random.
    
    Parameters
    ----------
    memoranda : A DataMatrix column with memoranda values in degrees (0 - 360)
    responses : A DataMatrix column with response values in degrees (0 - 360)
    
    Returns
    -------
    A (t_value, p_value) tuple.
    """
    
    real_errors = np.abs(_distance(memoranda, responses))
    shuffled_responses = responses[:]
    random.shuffle(shuffled_responses)
    fake_errors = np.abs(_distance(memoranda, shuffled_responses))
    return ttest_ind(real_errors, fake_errors)


def _distance(x, y):
    
    """A helper function that determines the rotational distance between x and
    y."""
    
    d = y - x
    if isinstance(d, (int, float)):
        if d >= 180:
            return d - 360
        if d < -180:
            return d + 360
        return d
    d[d >= 180] -= 360
    d[d < -180] += 360
    return d


def _error(args, x):
    
    """A helper function used for maximum likelihood estimation. This gives the
    error that should be minimized.
    """
    
    return -np.sum(np.log(mixture_model_pdf(x, *args)))


def aic(args, x):
    
    """A helper function used for Akaike information criterion."""
    
    pdf = mixture_model_pdf(x, *args)
    if hasattr(np, 'longdouble'):
        dtype = np.longdouble
    else:
        dtype = np.float64
    prod = np.prod(pdf, dtype=dtype)
    if prod == 0:
        warnings.warn('overflow in np.prod(), usin smallest possible float')
        prod = sys.float_info.min
    return 2 * len(args) - 2 * np.log(prod)


def _swap_pdf(x_target, x_nontargets, precision=STARTING_PRECISION,
              guess_rate=STARTING_GUESS_RATE, swap_rate=STARTING_SWAP_RATE,
              bias=STARTING_BIAS):
    x_target = np.radians(x_target)
    pdf_vonmises_target = vonmises.pdf(
        x=x_target,
        kappa=np.radians(precision),
        loc=np.radians(bias)
    )
    pdf_vonmises_non_targets = [vonmises.pdf(
        x=np.radians(x_nontarget),
        kappa=np.radians(precision),
        loc=np.radians(bias)
    ) for x_nontarget in x_nontargets]
    pdf_uniform = uniform.pdf(x_target, loc=-np.pi, scale=2 * np.pi)
    return (
        pdf_vonmises_target * (1 - guess_rate - swap_rate)
        + swap_rate * sum(pdf_vonmises_non_targets) / len(pdf_vonmises_non_targets)
        + pdf_uniform * guess_rate
    )


def _swap_error(args, x):
    
    return -np.sum(np.log(_swap_pdf(x[0], x[1], *args)))


def _fit_swap_model(x, x_nontargets=None,
                    include_bias=True, x0=None, bounds=None):
    
    if x0 is None:
        x0 = [STARTING_PRECISION, STARTING_GUESS_RATE]
        x0.append(STARTING_SWAP_RATE)
        if include_bias:
            x0.append(STARTING_BIAS)
    if bounds is None:
        bounds = [BOUNDS_PRECISION, BOUNDS_GUESS_RATE_WITH_SWAP]
        bounds.append(BOUNDS_SWAP_RATE)
        if include_bias:
            bounds.append(BOUNDS_BIAS)
    fit = optimize.minimize(
        _swap_error,
        args=[
            x,
            x_nontargets],
        x0=x0,
        bounds=bounds)
    if include_bias:
        return fit.x[0], fit.x[1], fit.x[3], fit.x[2]
    return fit.x
