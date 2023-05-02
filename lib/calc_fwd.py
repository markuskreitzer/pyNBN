# coding=utf-8
from act_func import ActivationFunction


def calculate_forward(inputs, topography, weights, activation, gain, parameters, initial_weights):
    af = ActivationFunction()
    np = np.shape(inputs)  # TODO: Make sure this is right.
    ni = parameters[1]
    no = parameters[2]
    nn = parameters[4]
    y = np.zeros(np, no)
    for p in range(0, np - 1):
        node[0:ni - 1] = inputs[p, 0:ni - 1]
        for n in range(0, nn - 1):
            j = ni + n
            net = weights[initial_weights[n]]
            for i in range(initial_weights[n] + 1, initial_weights[n + 1] - 1):
                net += node[topography[i]] * weights[i]

            out = af.activate(n, net, activation, gain)
            node[j] = out

        y[p, :] = node[ni + nn - no + 1:ni + nn]
    return y


"""
function [y_train] = calc_fwd(inp,topo,w,act,gain,param,iw)
np = size(inp,1);           % number of pattern
ni = param(2);           % number of input
no = param(3);           % number of output
nn = param(5);           % number of neurons
y_train = zeros(np,no);
for p = 1:np     % number of patterns
    node(1:ni) = inp(p,1:ni);
    for n = 1:nn % number of neurons
        j = ni + n;
        net = w(iw(n));
        for i = (iw(n)+1):(iw(n+1)-1)
            net = net + node(topo(i))*w(i);
        end;
        out=actFunc(n,net,act,gain);
        node(j) = out;
    end;
    y_train(p,:)=node(ni+nn-no+1:ni+nn);
end;


"""
