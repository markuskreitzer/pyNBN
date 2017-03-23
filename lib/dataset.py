# coding=utf-8
import numpy as np


def _normalize_by_rows(y):
    return np.array([x / np.linalg.norm(x) for x in y])


class Dataset(object):
    def __init__(self, input_data, desired_output=None, output_is_last_column=False):
        if output_is_last_column:
            self.input_data = input_data[0:-2]
            self.desired_output = input_data[-1]
        else:
            self.input_data = input_data
            self.desired_output = desired_output

        self.ni = self.input_data.shape[1]
        self.number_of_inputs = self.ni

        self.np = self.input_data.shape[0]
        self.number_of_patterns = self.np

        self.number_of_outputs = self.desired_output.shape[0]
        self.no = self.number_of_outputs

    @property
    def input_data_normalized(self):
        return _normalize_by_rows(self.input_data)

    @property
    def desired_output_normalized(self):
        return _normalize_by_rows(self.desired_output)


if __name__ == '__main__':
    pass
