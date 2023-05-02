def calc_fwd(inp, topo, w, act, gain, param, iw):
    np, ni, no, _, nn = param
    y = np.zeros((np, no))
    
    for p in range(np):  # number of patterns
        node = np.zeros(ni + nn)
        node[:ni] = inp[p, :ni]
        
        for n in range(nn):  # number of neurons
            j = ni + n
            net = w[iw[n]]
            
            for i in range(iw[n] + 1, iw[n + 1] - 1):
                net += node[topo[i]] * w[i]
            
            out = actFunc(n, net, act, gain)
            node[j] = out
        
        y[p, :] = node[ni + nn - no: ni + nn]
    
    return y

