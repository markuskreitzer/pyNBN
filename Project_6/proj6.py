# coding=utf-8
import numpy as np
import time

from lib.synthesis import synthesize_data
from elm.elm import ELMRegressor
from lib.plotting.plot import Plot

from lib.util import sse, add_column

# ##########################################################################
# Step 2: Write a script to train peaks
# function using 2000 data points for training, and 1000 data points for verification. Report the RMSE for both
# cases.

train_points = 2000
test_points = 1000
train_data = synthesize_data.gen(train_points)
test_data = synthesize_data.gen(test_points)

hidden_layers = 50
reg = ELMRegressor(n_hidden=hidden_layers)
tic = time.time()
train = reg.fit(train_data[:, :-1], train_data[:, -1])
toc = time.time() - tic
print "Train Time for {layers} hidden layers: {time} seconds".format(layers=hidden_layers, time=toc)
test_y_predicted = reg.predict(test_data[:, :-1])
train_y_predicted = reg.predict(train_data[:, :-1])
print "SSE Train Data", sse(train_y_predicted, train_data[:, -1])
print "SSE Test Data", sse(test_y_predicted, test_data[:, -1])

###########################################################################
# Step 3: Generate a 30x30 grid to plot resulting surface.

grid_dim = 30
x = np.linspace(-1, 1, grid_dim).reshape((grid_dim, 1))
y = np.linspace(-1, 1, grid_dim).reshape((grid_dim, 1))
X, Y = np.meshgrid(x, y)
XY = add_column(X.reshape(-1, 1), Y.reshape(-1, 1))

# Generate ideal surface and plot it.
Z_ideal = synthesize_data.surface_function(X, Y)
p = Plot(x, y, Z_ideal)
p.surf(save_fig_path='img/ideal.png', title='Ideal Surface Plot')

###########################################################################
# Step 4: Plot the resulting peaks function in 30x30 mesh using the surf.m function.
#         Calculate the Root Mean Squared Error value.
# Create surface based on ELM based on evenly spaced grid.
Z = reg.predict(XY)
Z = Z.reshape((grid_dim, grid_dim))
SSE = sse(Z_ideal, Z)
print "SSE Grid Data", SSE
error_surface = np.abs(Z_ideal - Z)
p = Plot(x, y, error_surface)
p.error_map(save_fig_path='img/elm_error_surf_{0}_layer.png'.format(hidden_layers), title='Error Surface SSE: %s, Hidden Layers: %s' % (str(SSE),str(hidden_layers)))
p = Plot(x, y, Z)
p.mesh(save_fig_path='img/elm_{0}_layer.png'.format(hidden_layers), title='SSE: %s, Hidden Layers: %s' % (str(SSE),str(hidden_layers)))
