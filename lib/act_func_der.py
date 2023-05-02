import numpy as np


def activation_function_derivative(n, net, act, gain):
    if act[n] == 0:
        out = gain[n] * net
        der = gain[n]
    elif act[n] == 1:
        out = 1 / (1 + np.exp(-gain[n] * net))
        der = gain[n] * (1 - out) * out
        # log-likelyhood cost function:
        # der = gain[n]/(1-out)/out
    elif act[n] == 2:
        out = np.tanh(gain[n] * net)
        der = gain[n] * (1 - out * out)
    elif act[n] == 3:
        out = gain(n) * net / (1 + gain(n) * abs(net))
        der = 1 / ((gain[n] * abs(net) + 1) ** 2)
    elif act[n] == 4:
        out = 2 * gain[n] * net / (1 + gain[n] * abs(net)) - 1
        der = 2 * gain[n] / (gain[n] * abs(net) + 1) ** 2
    else:
        out = None
        der = None
    return out, der



