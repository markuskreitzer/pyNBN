# coding=utf-8

"""
Since I'll be using the Matlab example code in class, I will only use Python to generate data files.

"""
import numpy as np
import pandas as pd
from lib.synthesis import synthesize_data
from lib.util import add_column

train_points = 80000
test_points = 1000
grid_dim = 30

#Wrap in Pandas dataframe for easy export.
train_data = pd.DataFrame(synthesize_data.gen(train_points), columns=['x', 'y', 'z'])
test_data = pd.DataFrame(synthesize_data.gen(test_points), columns=['x', 'y', 'z'])

# Export the data as CSV.
train_data.to_csv("ErrCor_ribbonDemo/train_80000.csv", index=False)
test_data.to_csv("ErrCor_ribbonDemo/test_1000.csv", index=False)


# Generate grid:
x = np.linspace(-1, 1, grid_dim).reshape((grid_dim, 1))
y = np.linspace(-1, 1, grid_dim).reshape((grid_dim, 1))
X, Y = np.meshgrid(x, y)
XY = add_column(X.reshape(-1, 1), Y.reshape(-1, 1))

# Generate ideal surface and plot it.
Z_ideal = synthesize_data.surface_function(X, Y)

grid_data = pd.DataFrame(add_column(XY, Z_ideal.reshape(-1, 1)), columns=list('xyz'))
grid_data.to_csv("ErrCor_ribbonDemo/grid_data.csv", index=False)