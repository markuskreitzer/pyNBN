# coding=utf-8
import numpy as np


def augment_ones(x):
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)


def add_column(np_array, column):
    return np.concatenate((np_array, column), axis=1)


def row_normalize(np_array):
    return np.array([x / np.linalg.norm(x) for x in np_array])


def sse(ideal_out, predicted_out):
    """
    https://en.wikipedia.org/wiki/Root-mean-square_deviation
    :type ideal_out: np.ndarray
    :type predicted_out: np.ndarray
    :param ideal_out: Known values
    :param predicted_out: Values that we are testing.
    :return: Sum-Squared Error
    """
    return ((predicted_out - ideal_out) ** 2).sum() / ideal_out.size


def mse(ideal_out, predicted_out):
    """
    https://en.wikipedia.org/wiki/Root-mean-square_deviation

    :type ideal_out: np.ndarray
    """
    return ((predicted_out - ideal_out) ** 2).sum() / ideal_out.size


def rmse(ideal_out, predicted_out):
    """
    https://en.wikipedia.org/wiki/Root-mean-square_deviation
    :param ideal_out:
    :param predicted_out:
    :return: RMSE

    """
    return np.sqrt(((predicted_out - ideal_out) ** 2).sum() / ideal_out.size)


if __name__ == '__main__':
    pass
