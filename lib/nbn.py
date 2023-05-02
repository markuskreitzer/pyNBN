# coding=utf-8
import numpy as np
import time

from lib.dataset import Dataset
from lib.topography import Topography


class NBN(object):
    def __init__(self, data_obj, topography_type, network, max_iterations, max_errors, test_proportion, generate_figures=False):
        assert isinstance(data_obj, Dataset), "%r is not a valid Dataset" % data_obj
        self.data_obj = data_obj
        self.training_parameters = self.parameters
        self.max_errors = max_errors
        self.max_iterations = max_iterations

        self.topography = Topography(topography_type, network).generate()

        self.results = {}

        # Input and Output Data
        self.training_inputs = self.data_obj.input_data
        self.training_desired_outputs = self.data_obj.desired_output

        # Normalize Data
        self.training_inputs = self.data_obj.input_data_normalized
        self.training_desired_outputs = self.data_obj.desired_output_normalized

        # Dr. Wilamoski's Method for normalization
        #self.training_inputs /= max(max(abs(self.training_inputs)))
        #self.training_desired_outputs -= min(self.training_desired_outputs)
        #self.training_desired_outputs /= max(self.training_desired_outputs)


        # Set up how we are going to train and verify our training.
        #
        number_of_patterns = self.data_obj.number_of_patterns
        ind = np.random.permutation(number_of_patterns)
        self.train_number_of_patterns = round(self.data_obj.number_of_patterns * test_proportion)

        self.training_inputs = self.training_inputs[ind[1:self.train_number_of_patterns], :]
        self.training_desired_outputs = self.training_desired_outputs[ind[1:self.train_number_of_patterns]]

        self.test_inputs = self.training_inputs[ind[self.train_number_of_patterns + 1:-1]]
        self.test_desired_outputs = self.training_desired_outputs[ind[self.train_number_of_patterns + 1:-1]]

        (self.train_number_of_patterns, self.nd) = self.training_inputs.shape
        self.test_number_of_patterns = number_of_patterns - self.train_number_of_patterns

        # Some constants (I'd like to remove these or put them in a class.
        self.mu = 0.01
        self.muH = 1e15
        self.muL = 1e-15
        self.scale = 10


        self.settings = [self.max_iterations, self.mu, self.muH, self.muL, self.scale, self.max_errors]

        self.np, self.ni, self.no, self.nw, self.nn, self.initial_weights = self._check_inputs(self.training_inputs, self.training_desired_outputs, self.topography)
        self.activation = 2 * np.ones(1, self.nn)
        self.activation[self.nn] = 0
        self.gain = 1 * np.ones(1, self.nn)
        self.parameters = [self.np, self.ni, self.no, self.nw, self.nn]
        self.threshold = 0.09  # Threshold for evaluating the success rate of running training
        self.RMSE_rc = []
        self.time_rc = []
        self.IS = 0
        self.RMSEt_rc = []
        self.ttime_rc = []
        self.training_error = 1000000


    def run_trails(self, number_of_trails):
        for tr in number_of_trails:
            initial_weights = self._generate_weights(self.nw)

            # Start Training
            tic = time.time()
            [self.weights, self.training_iterations, self.training_SSE] = self.train(initial_weights)
            self.train_time = time.time() - tic

            # Calculate Root Mean Square Error for training
            self.RMSE = np.sqrt(self.training_SSE / self.train_number_of_patterns)

            # I dont' like what's happening here:
            self.training_parameters[0] = self.test_number_of_patterns

            # Start Error Calculation On test patterns based on training.
            tic = time.time()
            SSE_testing = self._calculate_error(self.test_inputs, self.test_desired_outputs, self.topography, self.weights,
                                                self.activation, self.gain, self.training_parameters, initial_weights)
            self.test_time = time.time() - tic

            # Calculate target_rmse for training test
            self.RMSEt_rc[tr] = np.sqrt(SSE_testing / self.test_number_of_patterns)
            if self.RMSE[-1] < self.threshold:
                self.IS += 1
            self.RMSE_rc[tr] = self.RMSE[-1]
            self.time_rc[tr] = self.train_time
            self.train_time_rc[tr] = self.test_time

            self.testing_rmse = self.RMSE_rc[tr]
            if self.testing_rmse < self.training_error:
                self.results = {
                    'trails': tr,
                    'final_weights': self.weights,
                    'outputs': self._gen_outputs(self.test_inputs, self.test_desired_outputs, self.topography, self.weights, self.activation, self.gain, self.training_parameters, self.initial_weights),
                    'training': {
                        'error': self.RMSE[-1],
                        'time': self.train_time
                    },
                    'testing': {
                        'error': self.RMSEt_rc[tr],
                        'time': self.test_time
                    }
                }

    def _generate_weights(self, number_of_weights, seed=None):
        if seed:
            np.random.seed(seed)
        return 2 * np.random.rand(number_of_weights) - 1


    def _generate_topography(self, topography_type, network):




    def _rand_perm(self, m):
        pass

    def _check_inputs(self, Ti, Td, topography):
        pass

    def _calculate_error(self, Ti_test, Td_test, topography, weights, activation, gain, training_parameters,
                         initial_weights):
        pass

    def _gen_outputs(self, Ti_test, Td_test, topography, weights, activation, gain, training_parameters,
                     initial_weights):
        pass

    def train(self, initial_weights):
        pass

