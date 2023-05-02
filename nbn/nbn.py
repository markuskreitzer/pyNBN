import numpy as np
import matplotlib.pyplot as plt
import time

def nbn(data, network_type, network, ntrial, maxite, maxerr, nFig):
    def gen_topo(type, network):
        # Implement your gen_topo function here
        pass
    
    def checkingInputs(Ti, Td, topo):
        # Implement your checkingInputs function here
        pass
    
    def generate_weights(nw):
        # Implement your generate_weights function here
        pass

    def Trainer(Ti, Td, topo, w_ini, act, gain, param, iw, setting):
        # Implement your Trainer function here
        pass
    
    def calculate_error(Ti_tst, Td_tst, topo, w, act, gain, paramt, iw):
        # Implement your calculate_error function here
        pass
    
    def gen_outputs(Ti_tst, Td_tst, topo, w, act, gain, paramt, iw):
        # Implement your gen_outputs function here
        pass

    topo = gen_topo(network_type, network)
    m, n = data.shape
    Ti = data[:, :n - 1]
    Td = data[:, n - 1]

    # normalize
    Ti = Ti / np.max(np.abs(Ti))
    Td = Td - np.min(Td)
    Td = Td / np.max(Td)

    ind = np.random.permutation(m)
    Tnp = np.round(m)
    Ti_tst = Ti[ind[int(Tnp):], :]
    Ti = Ti[ind[:int(Tnp)], :]
    Td_tst = Td[ind[int(Tnp):]]
    Td = Td[ind[:int(Tnp)]]

    Tnp = Ti.shape[0]
    Tnp_tst = m - Tnp
    nd = Ti.shape[1]

    # Set train parameters
    mu = 0.01
    muH = 1e15
    muL = 1e-15
    scale = 10
    setting = [maxite, mu, muH, muL, scale, maxerr]

    np, ni, no, nw, nn, iw = checkingInputs(Ti, Td, topo)
    act = 2 * np.ones(nn)
    act[-1] = 0
    gain = np.ones(nn)
    param = [np, ni, no, nw, nn]

    threshold = 0.09
    RMSE_rc = []
    time_rc = []
    is_trial = 0
    RMSEt_rc = []
    ttime_rc = []
    training_error = 1e5

    plt.figure(nFig)
    plt.clf()

    for tr in range(ntrial):
        w_ini = generate_weights(nw)
        start_time = time.time()
        w, iter, SSE = Trainer(Ti, Td, topo, w_ini, act, gain, param, iw, setting)
        t = time.time() - start_time
        RMSE = np.sqrt(SSE / Tnp)
        paramt = param
        paramt[0] = Tnp_tst
        start_time = time.time()
        SSEt = calculate_error(Ti_tst, Td_tst, topo, w, act, gain, paramt, iw)
        tt = time.time() - start_time
        RMSEt_rc.append(np.sqrt(SSEt / Tnp_tst))

        if RMSE[-1] < threshold:
            is_trial += 1

        RMSE_rc.append

