from numpy import dot, multiply, exp, sqrt, size
from numpy import random, dot, multiply
import numpy as np

from lib import util

"""
Plot and prep data.

"""


def gen(num_points, noise=False, noise_level=0.01):
    inputs = dot(2, random.rand(num_points, 2)) - 1
    x1 = inputs[:, 0]
    x2 = inputs[:, 1]
    y = surface_function(x1, x2)
    y = y.reshape((y.shape[0], 1))
    if noise:
        noise = dot(sqrt(noise_level), random.normal(size=size(y)))
        y += noise
    # TODO: Convert to sklearn dataset class to standardize.
    return util.add_column(inputs, y)

def surface_function(x1, x2):
    """
    Returns surface function used in AIS Spring 2017
    :param x1: x inputs.
    :type x1: np.ndarray
    :type x2: np.ndarray
    :param x2: y inputs.
    :return: z value at x,y
    :rtype z: np.ndarray
    """
    return multiply((0.3 - dot(1.8, x1) + dot(2.7, x1 ** 2)), exp(- 1 - dot(6, x2) - dot(9, x1 ** 2) - dot(9, x2 ** 2))) - multiply( (dot(0.6, x1) - dot(27, x1 ** 3) - dot(243, x2 ** 5)), exp(dot(- 9, x1 ** 2) - dot(9, x2 ** 2))) - dot(1 / 30, exp( - 1 - dot( 6, x1) - dot( 9, x1 ** 2) - dot( 9, x2 ** 2)))
