def calculate_error(inp, dout, topo, w, act, gain, param, iw):
    np = param[0]  # number of pattern
    ni = param[1]  # number of input
    no = param[2]  # number of output
    nn = param[4]  # number of neurons
    err = 0

    for p in range(np):  # number of patterns
        node = [0] * (ni + nn)
        node[:ni] = inp[p, :ni]

        for n in range(nn):  # number of neurons
            j = ni + n
            net = w[iw[n]]
            for i in range(iw[n] + 1, iw[n + 1] - 1):
                net += node[topo[i]] * w[i]
            out = actFunc(n, net, act, gain)
            node[j] = out

        for k in range(no):
            err += (dout[p, k] - node[nn + ni - no + k]) ** 2  # calculate total error

    return err

