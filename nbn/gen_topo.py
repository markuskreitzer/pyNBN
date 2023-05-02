from enum import Enum
from typing import List

class NetworkType(Enum):
    SLP = "SLP"
    MLP = "MLP"
    FCC = "FCC"
    BMLP = "BMLP"

"""
MLP, network=>  ninp 3 4 2 1
SLP, network=>  ninp 17 1
FCC, network=>  ninp 1 1 1 1 1 1
MLP, network=>  ninp 3 4 2 1
"""

def gen_topo(type_: NetworkType, network: List[int]) -> List[int]:
    topo = []
    nl = len(network)
    
    for i in range(1, nl):  # for number of layers
        s = sum(network[0:i])  # starting a new layer
        
        for j in range(network[i]):  # in each layer
            if type_ == NetworkType.SLP:
                topo.extend([s + j, s - network[i - 1] + 1, s])
            elif type_ == NetworkType.MLP:
                topo.extend([s + j, s - network[i - 1] + 1, s])
            elif type_ == NetworkType.FCC:
                topo.extend([s + j, 1, s])  # s+j node number and j is always 1
            elif type_ == NetworkType.BMLP:
                topo.extend([s + j, 1, s])  # s+j node number
    
    return topo

