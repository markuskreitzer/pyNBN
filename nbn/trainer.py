import numpy as np

def trainer(inp: np.ndarray, dout: np.ndarray, topo: np.ndarray, w: np.ndarray,
            act: str, gain: float, param: np.ndarray, iw: np.ndarray, setting: np.ndarray):
    ww = w.copy()    # weight
    nw = param[3]    # number of weights
    maxite = setting[0]   # max iteration
    mu = setting[1]       # mu
    muH = setting[2]      # high bound of mu
    muL = setting[3]      # low bound of mu
    scale = setting[4]    # scale
    maxerr = setting[5]   # max requred error

    TER = calculate_error(inp, dout, topo, ww, act, gain, param, iw)
    SSE = np.zeros(maxite)
    SSE[0] = TER
    
    I = np.eye(nw)
    for iter in range(1, maxite):
        jw = 0
        gradient, hessian = hessian(inp, dout, topo, ww, act, gain, param, iw)
        ww_backup = ww.copy()
        
        while True:
            ww = ww_backup - np.linalg.solve((hessian + mu*I), gradient)
            TER = calculate_error(inp, dout, topo, ww, act, gain, param, iw)
            SSE[iter] = TER
            
            if TER <= SSE[iter-1]:
                if mu > muL:
                    mu = mu / scale
                break
            if mu < muH:
                mu = mu * scale
            jw += 1
            if jw > 30:
                break
        
        if SSE[iter] < maxerr:
            break
        if (SSE[iter-1]-SSE[iter]) / SSE[iter-1] < 0.000000000000001:
            break
    
    return ww, iter, SSE

