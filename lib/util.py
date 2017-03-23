# coding=utf-8
import numpy as np


def augment_ones(x):
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)


def add_column(np_array, column):
    return np.concatenate((np_array, column), axis=1)


def row_normalize(np_array):
    return np.array([x / np.linalg.norm(x) for x in np_array])


if __name__ == '__main__':
    pass
