# coding=utf-8
import numpy as np
from lib.act_func_der import activation_function_derivative


class Hessian(object):
    def __init__(self, inputs, desired_output, topology, weights, activation_function, gain, parameters,
                 initial_weights):
        self.ratio = 2
        self.inputs = inputs
        self.desired_output = desired_output
        self.topology = topology
        self.weights = weights
        self.activation_function = activation_function
        self.gain = gain
        self.parameters = parameters
        self.initial_weights = initial_weights

    def calc(self, inputs=None, dout=None, topo=None, ww=None, act=None, gain=None, param=None, iw=None):
        number_of_patterns = param[0]
        number_of_inputs = param[1]
        number_of_outputs = param[2]
        number_of_weights = param[3]
        number_of_neurons = param[4]

        gradient = np.zeros(number_of_weights, 1)
        hessian = np.zeros(number_of_weights, number_of_weights)

        # TODO: It should be last output node I believe.
        node = np.zeros(topo[-1]+1)
        de = np.zeros(topo[-1]+1)

        for p in np.arange(1, number_of_patterns).reshape(-1):
            node[1:number_of_inputs] = inputs[p, 1:number_of_inputs]

            for n in np.arange(1, number_of_neurons).reshape(-1):
                j = number_of_inputs + n
                net = ww[iw[n]]

                for i in np.arange((iw[n] + 1), (iw[n + 1] - 1)).reshape(-1):
                    net += np.dot(node[topo[i]], ww[i])

                out, de[j] = activation_function_derivative(n, net, act, gain)
                node[j] = out

            for k in np.arange(1, number_of_outputs).reshape(-1):
                error = dout[p, k] - node[number_of_neurons + number_of_inputs - number_of_outputs + k]
                jacobian = np.zeros(1, number_of_weights)
                o = number_of_neurons + number_of_inputs - number_of_outputs + k
                s = iw[o - number_of_inputs]
                jacobian[s] = - de[o]
                delo = np.zeros(1, number_of_neurons + number_of_inputs - number_of_outputs + 1)

                for i in np.arange((s + 1), (iw[o + 1 - number_of_inputs] - 1)).reshape(-1):
                    jacobian[i] = np.dot(node[topo[i]], jacobian[s])
                    delo[topo[i]] -= np.dot(ww[i], jacobian[s])

                for n in np.arange(1, (number_of_neurons - number_of_outputs)).reshape(-1):
                    j = number_of_neurons + number_of_inputs - number_of_outputs + 1 - n
                    s = iw[j - number_of_inputs]
                    jacobian[s] = np.dot(- de[j], delo[j])

                    for i in np.arange((s + 1), (iw[j - number_of_inputs + 1] - 1)).reshape(-1):
                        jacobian[i] = np.dot(node[topo[i]], jacobian[s])
                        delo[topo[i]] -= np.dot(ww[i], jacobian[s])

                if dout[p] > 0.5:
                    jacobian *= self.ratio

                gradient += np.dot(jacobian.T, error)
                hessian += np.dot(jacobian.T, jacobian)

        return gradient, hessian

if __name__ == '__main__':
    pass