# coding=utf-8

import numpy as np
from scipy.stats import vonmises, uniform
from scipy import optimize


X0 = [500, .2, 0]
BOUNDS = [
    (0, 10000),
    (0, 1),
    (-180, 180)
]
CATEGORIES = {
    'red': (-26.2058823530101, 33.676470588249, 5.999999999993606),
    'pink': (275.52941176473, 333.79411764698995, 281.6806722688825),
    'blue': (163.50000000003, 275.52941176473, 229.89915966385712),
    'green': (69.88235294115, 163.50000000003, 121.86554621849697),
    'yellow': (33.676470588249, 69.88235294115, 57.91596638651193)
}


def distance(x, y):
    
    d = y - x
    d[d > 180] -= 360
    d[d < -180] += 360
    return d


def standard_mixture_model_pdf(x, precision=100, guess_rate=.2, bias=0):
    
    x = np.radians(x)
    pdf_vonmises = vonmises.pdf(
        x=x,
        kappa=np.radians(precision),
        loc=np.radians(bias)
    )
    pdf_uniform = uniform.pdf(x, loc=-np.pi, scale=2*np.pi)
    return pdf_vonmises * (1 - guess_rate) + pdf_uniform * guess_rate


def _error(args, x):
    
    return -np.sum(np.log(standard_mixture_model_pdf(x, *args)))
    
    
def fit_standard_mixture_model(x):
    
    fit = optimize.minimize(_error, x0=X0, args=x, bounds=BOUNDS)
    precision, guess_rate, bias = fit.x
    return precision, guess_rate, bias

def category(x, categories):
    
    for category, (minval, maxval, proto) in categories.items():
        if minval <= x < maxval:
            return category
        if minval <= x - 360 < maxval:
            return category
        if minval <= x + 360 < maxval:
            return category
    raise ValueError('{} has no category'.format(x))  


def prototype(x, categories):
       
    return categories[category(x, categories)][2]
    
    
def response_bias(memoranda, responses, categories):
    
    errors = distance(memoranda, responses)
    protos = memoranda @ (lambda x: prototype(x, categories))
    proto_dists = distance(memoranda, protos)
    bias = []
    for error, proto_dist in zip(errors, proto_dists):
        if error * proto_dist < 0:
            bias.append(-abs(error))
        else:
            bias.append(abs(error))
    return bias
