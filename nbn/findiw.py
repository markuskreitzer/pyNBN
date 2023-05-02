from typing import List

def findiw(topo: List[int]) -> List[int]:
    nmax = 0
    j = 0
    iw = []

    for i in range(len(topo)):
        if topo[i] > nmax:
            nmax = topo[i]
            iw.append(i)
            j += 1

    iw.append(i + 1)
    return iw

