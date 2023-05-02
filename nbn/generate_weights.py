import random
from typing import List

def generate_weights(nw: int) -> List[float]:
    weights = []
    
    for _ in range(nw):  # number of weights
        ra = 2 * random.random() - 1  # generate random weights between -1 and 1
        
        while ra == 0:
            ra = 2 * random.random() - 1
        
        weights.append(ra)
    
    return weights

