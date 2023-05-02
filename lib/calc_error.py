# coding=utf-8
from act_func import ActivationFunction


def calculate_error(inputs, desired_output, topography, weights, activation_function, gain, parameters,
                    initial_weights):
    # TODO: 1) Why do I have initial weights?
    # TODO: 2) Parameters. Make dict or calculate?

    """
    I hope this code works!

    :param inputs: Input Patterns
    :param desired_output: Desired Output Patterns
    :param topography: Layout of Neuron Layers
    :param weights: Weights to test.
    :param activation_function: Type of activation function to use
    :param gain: Gain
    :param parameters: [number of patterns, number of inputs, number of outputs, number of neurons]
    :param initial_weights: Initial weights?
    :return:
    """
    af = ActivationFunction()
    # TODO Need to convert to a dict...
    np = parameters[0]  # number of pattern
    ni = parameters[1]  # number of input
    no = parameters[2]  # number of output
    nn = parameters[4]  # number of neurons
    error = 0
    node = np.array()
    for p in range(1, np):  # number of patterns
        node[1:ni] = inputs[p, 1:ni]
        for n in range(1, nn):  # number of neurons
            j = ni + n
            net = weights[initial_weights[n]]
            for i in range((initial_weights[n] + 1), (initial_weights[n + 1] - 1)):
                net += node[topography[i]] * weights[i]

            out = af.activate(n, net, activation_function, gain)
            node[j] = out

        for k in range(1, no):
            error += (desired_output[p, k] - node[nn + ni - no + k]) ** 2  # calculate total error
