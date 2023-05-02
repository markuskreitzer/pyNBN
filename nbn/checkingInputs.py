import numpy as np

def checkingInputs(inp, dout, topo):
    def findiw(topo):
        iw = [0]
        for i, t in enumerate(topo[:-1]):
            if t == 0:
                iw.append(i + 1)
        iw.append(len(topo))
        return iw

    iw = findiw(topo)
    np, ni = inp.shape
    y, no = dout.shape

    if np != y:
        raise ValueError("input and output patterns are not equal")

    nw = len(topo)
    y = len(topo)
    nn = len(iw) - 1

    if np.min(topo) < 1:
        raise ValueError("all elements of topo must be positive")

    if nw == 0:
        raise ValueError("weights must not be zero")

    return np, ni, no, nw, nn, iw

