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

import numpy as np
from scipy.stats import vonmises, uniform, ttest_ind
from scipy import optimize

__version__ = '1.0.0'

# These default categories have been established in a separate validation
# experiment. Each tuple indicates a start_value, end_value, and prototype.
# values are hues in 0 - 360 in HSV space such that 0 is red.
DEFAULT_CATEGORIES = {
    'red': (-26.2058823530101, 33.676470588249, 5.999999999993606),
    'pink': (275.52941176473, 333.79411764698995, 281.6806722688825),
    'blue': (163.50000000003, 275.52941176473, 229.89915966385712),
    'green': (69.88235294115, 163.50000000003, 121.86554621849697),
    'yellow': (33.676470588249, 69.88235294115, 57.91596638651193)
}

# Starting parameters of the fit
X0 = [500, .2, 0]
# Realistic bounds for each parameter
BOUNDS = [
    (0, 10000),  # precision
    (0, 1),  # guess rate
    (-180, 180)  # bias
]


def mixture_model_pdf(x, precision=100, guess_rate=.2, bias=0):
    
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


def fit_mixture_model(x, x0=X0, bounds=BOUNDS):
    
    """Fits the biased mixture model to a dataset. The input to the mixture
    model should generally be a response bias as determined by
    `response_bias()`.
    
    Parameters
    ----------
    x : A DataMatrix column (or other iterable object) of response biases.
    
    Returns
    -------
    A (precision, guess_rate, bias) tuple.
    """        
    
    fit = optimize.minimize(_error, x0=x0, args=x, bounds=bounds)
    return fit.x


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
    A protopy value in degrees (0 360).
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
    protos = memoranda @ (lambda x: prototype(x, categories))
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
    np.random.shuffle(shuffled_responses)
    fake_errors = np.abs(_distance(memoranda, shuffled_responses))
    return ttest_ind(real_errors, fake_errors)


def _distance(x, y):
    
    """A helper function that determines the rotational distance between x and
    y."""
    
    d = y - x
    d[d > 180] -= 360
    d[d < -180] += 360
    return d


def _error(args, x):
    
    """A helper function used for maximum likelihood estimation. This gives the
    error that should be minimized.
    """
    
    return -np.sum(np.log(mixture_model_pdf(x, *args)))


def aic(args, x):
    
    """A helper function used for Akaike information criterion."""
    
    return 2 * len(args) - 2 * np.log(np.prod(mixture_model_pdf(x, *args)))
