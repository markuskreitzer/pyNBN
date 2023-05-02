import numpy as np
from enum import Enum
from dataclasses import dataclass

class Activation(Enum):
    LINEAR = 0
    UNIPOLAR = 1
    BIPOLAR = 2
    ELLIOT = 3
    UNIPOLAR_ELLIOT = 4

@dataclass
class Params:
    np: int
    ni: int
    no: int
    nw: int
    nn: int

def main():
    data_file = 'parity7_int.dat'
    data = np.loadtxt(data_file)
    m, n = data.shape
    ninp = n - 1
    ntrial = 10
    maxite = 2000
    maxerr = 1e-3
    type = 'FCC'
    num_neurons = 3
    network = [ninp] + [1] * num_neurons
    topo = generate_topo(Topology.FCC, network)
    n_fig = 11 + num_neurons - 1
    stats, weights, outputs = nbn(data, type, network, ntrial, maxite, maxerr, n_fig)

