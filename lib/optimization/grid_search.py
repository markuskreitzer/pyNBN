# coding=utf-8
import time
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import ParameterGrid

# Custom Library
from lib.plotting.plot import Plot


class GridSearch(object):
    def __init__(self, train_data, test_data=None, make_plots=False, winner_plot=False, plot_filename_prefix='plot'):
        self.plot_filename_prefix = plot_filename_prefix
        self.make_plots = make_plots
        self.winner_plot = winner_plot

        # TODO: Create data class inherited from sklearn.
        # Fancy way of slicing all but last column.
        x_train = train_data[:, :-1]

        # Grab the last column.
        self.y_train = train_data[:, -1]

        if test_data is None:
            x_train, x_test, self.y_train, self.y_test = train_test_split(x_train, self.y_train, test_size=0.7,
                                                                          random_state=0)
        else:
            x_test = test_data[:, :-1]
            self.y_test = test_data[:, -1]

        # Per the LIBSVM online user guide, it is recommended to run input data through the given scaling function for
        # faster processing.
        self.x_train = preprocessing.scale(x_train)
        self.x_test = preprocessing.scale(x_test)

        # This will store the best prediction the SVR can come up with that
        # corresponds with the mse above.
        self.winner = {}

        # This is an arbitrary high number just to start updating this variable with the lowest mse.
        self.best_mse = float("inf")

        self.prediction = np.zeros(self.y_test.shape)
        self.best_prediction = np.zeros(self.y_test.shape)

    def svr_grid_search(self, param_grid, make_plots=False, winner_plot=False):
        # TODO: Generalize to pass in any arbitrary learning object, e.g. my Neuron-By-Neuron object to search.
        """
        Iterates over a grid of parameters. Scoring is based on Mean-Squared Error (mse).
        Returns the parameters with the lowest mse.

        Parameter Explanation:
        =====================

        gamma: influence of a single training point on the rest.
            low -> "far influence"
            high -> "near influence"
        C:
            low -> smooth but inaccurate
            high -> accurate surface (slow computationally)

        """
        # Update variable that determines whether we generate a plot for the lowest mse values.
        if winner_plot:
            self.winner_plot = winner_plot

        # Iterate over each parameter and generate a surface.
        cnt = 1

        for params in ParameterGrid(param_grid):
            my_gamma = params['gamma']
            my_c = params['C']

            # Set shrinking heuristics to false as per the example Matlab code.
            clf = SVR(shrinking=False, kernel='rbf', C=my_c, gamma=my_gamma, cache_size=500)
            # Train against training set.
            tic = time.time()
            clf.fit(self.x_train, self.y_train)
            toc = time.time() - tic

            # Generate predicted set using SVR network using the test set.
            train_prediction = clf.predict(self.x_train)

            # Create output based on SVR prediction.
            self.prediction = clf.predict(self.x_test)

            # Calculate the Mean Squared Error based on the ideal test output and that predicted by the SVR.
            train_mse = mean_squared_error(self.y_train, train_prediction)
            test_mse = mean_squared_error(self.y_test, self.prediction)

            # Record Best mse and update with a better
            if test_mse < self.best_mse:
                self.best_mse = test_mse
                self.best_prediction = self.prediction
                self.winner = {
                    'SVR': clf,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_time': toc
                }
            if make_plots:
                # Generate graph and save as file for upload to LaTeX document.
                # Have to do this because I wanted to add another parameter with gamma = 'auto'
                # When gamma = 'auto', it sets it to the reciprocal of the number of RBFs, if I'm not mistaken.

                if my_gamma != 'auto':
                    gamma_str = '%0.3f' % my_gamma
                else:
                    gamma_str = my_gamma

                my_title = 'Surface with gamma=%s C=%0.3f and mse=%0.5f (train mse=%0.5f)' % (
                    gamma_str, my_c, test_mse, train_mse)
                my_filename = '%s_%d' % (self.plot_filename_prefix, cnt)
                Plot(self.x_test[:, 0], self.x_test[:, 1], self.prediction).surf_rand(title=my_title,
                                                                                      save_fig_path=my_filename)
            cnt += 1

        if self.winner_plot:
            my_gamma = self.winner['SVR'].gamma
            my_c = self.winner['SVR'].C
            if my_gamma != 'auto':
                gamma_str = '%0.3f' % my_gamma
            else:
                gamma_str = my_gamma

            my_title = 'Surface with gamma=%s C=%0.3f and Test mse=%0.5f (Train mse=%0.5f)' % (
                gamma_str, my_c, self.winner['test_mse'], self.winner['train_mse'])
            my_filename = '%s_winner.png' % self.plot_filename_prefix
            Plot(self.x_test[:, 0], self.x_test[:, 1], self.best_prediction).surf_rand(title=my_title,
                                                                                       save_fig_path=my_filename)
        return self.winner
