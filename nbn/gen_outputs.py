import numpy as np
from typing import List, Tuple

def gen_outputs(inp: np.ndarray, dout: np.ndarray, topo: List[int], w: List[float], 
                act: List[int], gain: List[float], param: List[int], iw: List[int]) -> np.ndarray:
    np_, ni, no, nn = param[0], param[1], param[2], param[4]
    err = 0
    outputs = np.zeros((np_, ni + 1))

    for p in range(np_):  # number of patterns
        node = np.zeros(ni + nn)
        node[0:ni] = inp[p, 0:ni]
        
        for n in range(nn):  # number of neurons
            j = ni + n
            net = w[iw[n]]
            
            for i in range(iw[n] + 1, iw[n + 1] - 1):
                net += node[topo[i]] * w[i]
            
            out = act_func(n, net, act, gain)
            node[j] = out
        
        outputs[p, 0:ni] = inp[p]
        outputs[p, ni] = out

    return outputs

