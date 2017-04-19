# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lib.util import *
parity = 8 # Must be even

given_inputs = pd.Series(np.arange(-parity+1, parity+1, 2)).values.reshape(parity, 1)
#print given_inputs
desired_output = pd.Series([-1, 1] * (parity/2)).values.reshape(parity, 1)
#print desired_output
#print "inputs:", given_inputs.shape
#print "outputs:", desired_output.shape
weights = np.random.rand(*given_inputs.shape)
gradient = np.gradient(add_column(given_inputs, desired_output))
hessian = np.gradient(gradient)
print hessian[0]
print hessian[1]
print hessian[2]
#plt.plot(*gradient[0], marker='o')
#plt.plot(*gradient[1], marker='x')

plt.plot(given_inputs, desired_output)
plt.scatter(given_inputs, np.sign(desired_output), marker='o')
plt.show()


# 5 pm TV