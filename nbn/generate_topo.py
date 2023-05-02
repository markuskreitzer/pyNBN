from typing import List

def generate_topo(type: str, lbl: List[int]) -> List[int]:
    topo = []
    nl = len(lbl)

    for i in range(1, nl):
        s = sum(lbl[:i])
        
        for j in range(1, lbl[i] + 1):
            if type == 'MLP':
                topo.extend([s + j] + list(range(s - lbl[i - 1] + 1, s + 1)))
            elif type == 'BMLP':
                topo.extend([s + j] + list(range(1, s + 1)))

    return topo

