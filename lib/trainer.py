# coding=utf-8
import numpy as np
from theano.configdefaults import param

from act_func import ActivationFunction
from lib.hessian import Hessian


class Trainer(object):
    def __init__(self, dataset_obj, topography, weights, activation_function, gain, parameters,
                 initial_weights, settings):
        # Datasets
        self.dataset = dataset_obj
        self.inputs = self.dataset.input_data
        self.desired_output = self.dataset.desired_output

        # Neural Network
        self.topography = topography
        self.weights = weights
        self.activation_function = activation_function
        self.gain = gain
        self.parameters = parameters
        self.initial_weights = initial_weights
        self.settings = settings

        # Hessian
        self.hessian = Hessian.calc

        # Just some janky stuff I kept of Dr. W.
        self.ww = self.weights                    # weight
        self.nw = self.parameters[3]    # number of weights

        # Settings
        self.maxite = self.settings[0]  # max iterations
        self.mu = self.settings[1]      # mu
        self.muH = self.settings[2]     # high bound of mu
        self.muL = self.settings[3]     # low bound of mu
        self.scale = self.settings[4]   # scale
        self.maxerr = self.settings[5]  # max requred error

        self.SSE = np.zeros(maxite)

    def train(self):
        SSE[0] = self._calculate_error(self.inputs,self.desired_output,self.topography,self.weights,self.activation_function,self.gain,self.parameters,self.initial_weights)
        I = np.identity(self.nw)

        for cnt in range(2, self.maxite +1):
            jw = 0
            (gradient, hessian) = self.hessian(self.inputs, self.desired_output, self.topography, self.weights, self.activation_function, self.gain, self.parameters, self.initial_weights)
            ww_backup = ww
            while True:
                ww = ww_backup - np.linalg.solve((hessian+mu*I), gradient).transpose()
                SSE[cnt] = self._calculate_error(inp, dout, topo, ww, act, gain, param, iw)
                if SSE[cnt] <= SSE[cnt-1]:
                    if mu > muL:
                        mu = mu/scale
                    break

                if mu < muh:
                    mu = mu*scale

                jw += 1
                if jw > 30:
                    break

            if (SSE[cnt-1] - SSE[cnt])/SSE[cnt-1] < 0.000000000000001:
                break



        [weights, iterations, sum_squared_error] = ww, cnt, SSE
        return weights, iterations, sum_squared_error

    """
    TER = calculate_error(inp,dout,topo,ww,act,gain,param,iw);
    SSE(1) = TER;
    I = eye(nw);
    for iter = 2:maxite
        jw = 0;
        [gradient,hessian] = Hessian(inp,dout,topo,ww,act,gain,param,iw);
        ww_backup = ww;
        while 1
            ww = ww_backup - ((hessian+mu*I)\gradient)';
            TER = calculate_error(inp,dout,topo,ww,act,gain,param,iw);
            SSE(iter) = TER;
            if TER <= SSE(iter-1)
                if mu > muL
                    mu = mu/scale;
                end;
                break;
            end;
            if mu < muH
                mu = mu*scale;
            end;
            jw = jw + 1;
            if jw > 30
                break;
            end;
        end;
        %disp(sprintf('iter %d, SSE=%f',iter,SSE(iter)));
        if SSE(iter) < maxerr
            break;
        end;
        if (SSE(iter-1)-SSE(iter))/SSE(iter-1)<0.000000000000001
            break;
        end;
    end;
    return;



    """




    def _calculate_error(self, inputs, desired_output, topography, weights, activation_function, gain, parameters,
                        initial_weights):
        # TODO: 1) Why do I have initial weights?
        # TODO: 2) Parameters. Make dict or calculate?

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


