import numpy as np
from typing import List
from . import gen_topo

ndim = 4
h = 3
topo = gen_topo('BMLP', [ndim] + [1]*h + [1])
print(topo)

topo = gen_topo('FCC', [ndim] + [1]*h + [1])
print(topo)

