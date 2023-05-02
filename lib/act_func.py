# coding=utf-8
from math import exp
import numpy as np


class ActivationFunction(object):
    def __init__(self, n, net, act, gain):
        self.gain = gain
        self.act = act
        self.net = net
        self.n = n

    def activate(self, n, net, act, gain):
        # TODO Add a way to call the other functions with the "act" argument as the specification of what we'll use.
        self.activate_unipolar_elliot()

    def activate_linear(self):
        return self.gain[self.n] * self.net

    def activate_unipolar(self):
        return 1 / (1 + exp(-self.gain[self.n] * self.net))

    def activate_bipolar(self):
        return np.tanh(self.gain[self.n] * self.net)

    def activate_biploar_elliot(self):
        return self.gain[self.n] * self.net / (1 + self.gain[self.n] * abs(self.net))

    def activate_unipolar_elliot(self):
        return 2 * self.gain[self.n] * self.net / (1 + self.gain[self.n] * abs(self.net)) - 1

    def actFunc(self, n, net, act, gain):
        # Wilamowski activatation function
        switch = act(n)
        if switch == 0:
            # linear neuron
            return gain[n] * net
        elif switch == 1:
            # unipolar neuron
            return 1 / (1 + exp(-gain[n] * net))
        elif switch == 2:
            # bipolar neuron
            return np.tanh(gain[n] * net)
        elif switch == 3:
            # bipolar elliot neuron
            return gain[n] * net / (1 + gain[n] * abs(net))
        elif switch == 4:
            # unipolar elliot neuron
            return 2 * gain[n] * net / (1 + gain[n] * abs(net)) - 1
