from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy import random, dot, multiply

from lib import util

"""
Plot and prep data.

"""


def gen(num_points, shape=None, noise=None):
    # TODO: add ability to give function to generate dataset for function approximation and with specific shape.
    inputs = dot(2, random.rand(num_points, 2)) - 1
    x1 = inputs[:, 0]
    x2 = inputs[:, 1]
    y = multiply((0.3 - dot(1.8, x1) + dot(2.7, x1 ** 2)),
                             exp(- 1 - dot(6, x2) - dot(9, x1 ** 2) - dot(9, x2 ** 2))) - multiply(
        (dot(0.6, x1) - dot(27, x1 ** 3) - dot(243, x2 ** 5)), exp(dot(- 9, x1 ** 2) - dot(9, x2 ** 2))) - dot(1 / 30, exp(
        - 1 - dot(6, x1) - dot(9, x1 ** 2) - dot(9, x2 ** 2)))
    y = y.reshape((y.shape[0], 1))
    if noise:
        # TODO: If we add noise, we have to adjust our RBF parameters to remove outliers. Will consider this later
        # when I get everything working.
        noise = dot(sqrt(0.01), random.normal(size=size(y)))
        y += noise
    # TODO: Convert to sklearn dataset class to standardize.
    return util.add_column(inputs, y)


