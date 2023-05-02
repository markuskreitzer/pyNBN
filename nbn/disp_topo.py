from typing import List

def disp_topo(topo: List[int]) -> None:
    iw = findiw(topo)
    # print("Neural Network Topology")
    
    for i in range(len(iw) - 1):
        print(f" neuron #{topo[iw[i]]:2d} connected to:", end="")
        
        j1, j2 = iw[i] + 1, iw[i + 1] - 1
        
        for j in range(j1, j2 + 1):
            print(f" #{topo[j]:2d},", end="")
        
        print()

    return

