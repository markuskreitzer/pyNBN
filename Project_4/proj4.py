# coding=utf-8
"""
  Support Vector Regression.
  Markus Kreitzer
  Last Updated: Friday March 10, 2017
"""

from __future__ import print_function
from pprint import pprint
import numpy as np

# Custom Libraries
from lib.plotting.plot import Plot
import synthesize_data as synth_data

###########################################################################
# Step 1: Download and install LIBSVM. Note: This is my own wrapper for the official LIBSVM library. The library is a
#  bit more abstracted than the example Matlab code that was given for this class project.
from grid_search import GridSearch

###########################################################################
# Step 2: Generate 30*30 points to verify and plot resulting data. I generated the following data by randomly
# selecting numbers on a specific interval. Normally, one set of data is generated and then partitioned and cross
# validated. I suspect that generating two independent random sets of points exacerbates the problem of over-fitting
# the data.
side = 30
num_points = side ** 2
train_data = synth_data.gen(num_points)
test_data = synth_data.gen(num_points)


###########################################################################
# Step 3: Modify the SVR.m script so for each set of training parameters gamma and C (20 different plots) the
# resulted surface can be plotted. On figures print values of training and verifications errors.
param_grid = {
    'gamma': [0.001, 0.01, 0.1, 1, 10],
    'C': [1, 10, 100, 1000]
}

# But first, lets plot what our surface ideally looks like from our generated test (not training) data. surf_rand is
# a method I specifically wrote that can take unevenly spaced (or random) data points in a 3-dimensional space, and
# using linear triangulation, interpolate a a grid for plotting.
ideal = Plot(test_data[:, 0], test_data[:, 1], test_data[:, 2])
ideal.surf_rand(title='Ideal Surface', save_fig_path='img/ideal_surface.png')

# This function will train and plot all the surfaces.
grid = GridSearch(train_data, test_data, plot_filename_prefix='img/step3')
winner = grid.svr_grid_search(param_grid, make_plots=True)

###########################################################################
# Step 4: Run the SVR.m and find the best values for gamma and C parameters using the simple grid search.
# We use the grid_search function again but this time we are interested in the best results.
grid = GridSearch(train_data, test_data, winner_plot=True, plot_filename_prefix='img/step4')
winner = grid.svr_grid_search(param_grid)
print("Best Performing Set:")
pprint(winner)
print()
print()


###########################################################################
# Step 5: Based on results from (4) try to select try to reduce increments in the grid search so even better
# verification errors can be obtained.
param_grid = {'gamma': np.arange(0.001, 100),
              'C': np.arange(80, 120)}
grid = GridSearch(train_data, test_data, winner_plot=True, plot_filename_prefix='img/step5')
winner = grid.svr_grid_search(param_grid)
print("Best Performing Set:")
pprint(winner)
print()
print()


###########################################################################
# Step 6: Plot the final surface and indicate resulted parameters such as number of RBF units, gamma, C, and errors.
# Also please inspect heights of RBF units (do not print them)
#
# Performed in step 5 with winner_plot=True flag.
#
if __name__ == '__main__':
    pass

