import numpy as np

def calculate_error_vectorized(inp, dout, topo, w, act, gain, param, iw):
    np = param[0]  # number of pattern
    ni = param[1]  # number of input
    no = param[2]  # number of output
    nn = param[4]  # number of neurons

    def calc_fwd_vectorized(inp):
        node = np.zeros((inp.shape[0], ni + nn))
        node[:, :ni] = inp

        for n in range(nn):
            j = ni + n
            net = np.full((inp.shape[0],), w[iw[n]])
            for i in range(iw[n] + 1, iw[n + 1] - 1):
                net += node[:, topo[i]] * w[i]
            out = np.vectorize(actFunc)(n, net, act, gain)
            node[:, j] = out

        return node[:, -no:]

    y_pred = calc_fwd_vectorized(inp)
    err = np.sum((dout - y_pred) ** 2)
    return err

