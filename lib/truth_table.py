# coding=utf-8
import numpy as np


class TruthTableCreator(object):
    def __init__(self, inputs=2, num_type='bipolar'):
        self.num_type = num_type
        self.inputs = inputs

    @property
    def table(self):
        bits = 2 ** self.inputs
        if self.num_type == 'unipolar':
            return np.array([[int(x) for x in "{ROW:0{INPUTS}b}".format(ROW=n, INPUTS=self.inputs)] for n in xrange(bits)])
        elif self.num_type == 'bipolar':
            return np.array([[int(-1 if int(x) == 0 else 1) for x in "{ROW:0{INPUTS}b}".format(ROW=n, INPUTS=self.inputs)] for n in xrange(bits)])
        else:
            raise

