import numpy as np
from typing import List, Tuple

def Hessian(inp: np.ndarray, dout: np.ndarray, topo: List[int], ww: np.ndarray, act: List[int], gain: np.ndarray, param: Tuple[int, int, int, int, int], iw: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    np, ni, no, nw, nn = param

    gradient = np.zeros(nw)
    hessian = np.zeros((nw, nw))

    for p in range(np):
        node = np.zeros(ni + nn + 1)
        node[:ni] = inp[p, :ni]
        de = np.zeros(ni + nn + 1)

        for n in range(nn):
            j = ni + n
            net = ww[iw[n]]
            net += np.sum(node[topo[iw[n] + 1:iw[n+1]]] * ww[iw[n]+1:iw[n+1]])
            out, der = actFuncDer(n, net, act, gain)
            node[j] = out
            de[j] = der

        for k in range(no):
            error = dout[p,k] - node[ni+nn-no+k]
            J = np.zeros(nw)
            o = nn + ni - no + k
            s = iw[o-ni]
            J[s] = -de[o]
            delo = np.zeros(ni + nn + 1)
            delo[topo[iw[o-ni]:iw[o-ni+1]]] = -ww[iw[o-ni]:iw[o-ni+1]] * J[s]
            for i in range(s+1, iw[o-ni+1]-1):
                J[i] = node[topo[i]] * J[s]
                delo[topo[i]] -= ww[i] * J[s]

            for n in range(nn - no):
                j = nn + ni - no + 1 - n
                s = iw[j-ni]
                J[s] = -de[j] * delo[j]
                for i in range(s+1, iw[j-ni+1]-1):
                    J[i] = node[topo[i]] * J[s]
                    delo[topo[i]] -= ww[i] * J[s]

            ratio = 2 if dout[p] > 0.5 else 1
            gradient += J * error * ratio
            hessian += np.outer(J, J) * ratio

    return gradient, hessian

