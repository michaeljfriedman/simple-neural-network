#-------------------------------------------------------------------------------
# neural_network.py
# Author: Michael Friedman
#
# Library implementing a simple neural network based on Nielson's book:
#   http://neuralnetworksanddeeplearning.com/index.html
# This implements feedforward and backpropagation-based learning in a network
# containing variable number of inputs, outputs, and hidden layers, with a
# client-defined learning rate. This version also allows the client to choose
# from two error heuristics: Euclidean distance and cross entropy.
#
# Note on notation:
#   Since the code deals with various mathematical types (scalars, vectors,
#   matrices), the following notation is used in naming variables to make it
#   clear which type they are:
#     - Scalars are named with *lowercase letters*. Ex: x
#     - Vectors are named with *lowercase letters, followed by 'v'*. Ex: xv
#     - Matrices are named with *uppercase letters*. Ex: X
#     - Derivatives are preceded by 'dx_', where x is the variable of
#       differentiation. Ex: dx_f is d/dx(f), dxv_f is grad_x(f).
#     - Delta values are preceded by 'delta_'. Ex: delta_W, delta_bv.
#-------------------------------------------------------------------------------

from copy import deepcopy
from time import sleep
import cPickle
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork(object):

    # Constants to represent the error functions/heuristics supported
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    CROSS_ENTROPY = "cross_entropy"

    def __init__(self, layers, weights=None, biases=None):
        '''
        Initializes the NeuralNetwork given `layers`, a list of integers
        indicating the number of neurons in each layer of the network. (The
        first and last value in this list are thus the number of inputs and
        outputs of the network, respectively.) Values in this list must all be
        positive numbers, or an exception will be raised. (...)

        (see docs/API.md for full description)
        '''
        # Validate parameters
        for n in layers:
            if n <= 0:
                raise ValueError('Number of neurons in each layer must be positive')

        # Initialize number of layers
        self.L = len(layers)

        # Initialize weights arbitrarily. Each consecutive pair of layers
        # (prev/next layers) has a matrix of weights.
        if not weights:
            weights = [np.random.rand(layers[i+1], layers[i]) for i in range(0, len(layers)-1)]

        # Initialize biases arbitrarily. Same principle as with weights, except
        # each pair of layers has a vector of biases.
        if not biases:
            biases = [np.random.rand(layers[i+1],) for i in range(0, len(layers)-1)]

        # Layers are represented by tuples:
        #   (number of neurons, weight matrix, bias vector)
        # Populate these layer tuples, store in a list
        self.layers = [(layers[0], None, None)] + [(layers[i], weights[i-1], biases[i-1]) for i in range(1, len(layers))]

        # Stats to be recorded at each round of training
        self.errors = []
        self.accuracies = []
        self.last_round = 0


    # Required for testing

    def __eq__(self, other):
        '''
        Defines equality
        '''
        if not isinstance(other, self.__class__): return False
        if self.L != other.L: return False
        if self.errors != other.errors: return False
        if self.accuracies != other.accuracies: return False
        if self.last_round != other.last_round: return False
        for (n1, W1, bv1), (n2, W2, bv2) in zip(self.layers, other.layers):
            if n1 != n2: return False
            if not np.array_equal(W1, W2): return False
            if not np.array_equal(bv1, bv2): return False
        return True

    def __ne__(self, other):
        '''
        Defines inequality
        '''
        return not (self == other)


    #---------------------------------------------------------------------------

    # Helper methods

    @staticmethod
    def basis_vector(i, n):
        '''
        Returns the standard basis vector e_i with n dimensions as a np array.
        '''
        v = np.zeros(n)
        v[i] = 1
        return v

    @staticmethod
    def sigmoid(x):
        '''
        Returns a np array that applies the sigmoid function element-wise to the
        np array x.
        '''
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def error(av, yv, error_function):
        '''
        Returns the error between the activation av and correct output yv
        for the heuristic given by error_function.
        '''
        if error_function == NeuralNetwork.EUCLIDEAN_DISTANCE:
            return 0.5 * np.linalg.norm(av - yv)**2
        elif error_function == NeuralNetwork.CROSS_ENTROPY:
            return -np.sum(yv * np.log(av) + (1 - yv) * np.log(1 - av))

    def feedforward(self, xv):
        '''
        Returns the list of activation vectors from each layer after feeding
        forward the input xv through the network. (Special notes: the first
        item in this list is the input xv, and the last item is the output
        vector.)
        '''
        av = xv
        avs = [av]
        for _, W, bv in self.layers[1:]:
            # Compute activation vector at this layer
            zv = W.dot(av) + bv
            av = NeuralNetwork.sigmoid(zv)
            avs.append(av)
        return avs

    def backpropagate(self, y, avs, eta, error_function):
        '''
        Compute error in the last layer from the correct label y, and
        backpropagate that error using: the list avs of activations from each
        layer in the network (the result of feedforward(xv)), the learning
        rate eta, and the error heuristic given by `error_function`.

        Refer to docs for the explanation of this algorithm, and for its
        derivation.
        '''
        # Convert y to vector form
        n, _, _ = self.layers[-1]
        yv = NeuralNetwork.basis_vector(y, n)

        # Definition of sigmoid_prime
        # NOTE: Since sigmoid_prime can be evaluated in terms of sigmoid, I
        # define it in terms of sigmoid, rather than the original input.
        # This saves us from having to store the original inputs zv to sigmoid.
        sigmoid_prime = lambda sigmoid: sigmoid * (1 - sigmoid)

        # Initialize vars based on the error function
        av = avs[-1]
        if error_function == NeuralNetwork.EUCLIDEAN_DISTANCE:
            dav_C = av - yv
            dzv_C = dav_C * sigmoid_prime(av)
        elif error_function == NeuralNetwork.CROSS_ENTROPY:
            dav_C = None    # does not need to be initialized
            dzv_C = av - yv
        for i in reversed(range(1, self.L)):
            _, W, bv = self.layers[i]
            av = avs[i]
            av_prev = avs[i-1]

            # Compute deltas in biases and weights
            delta_bv = -eta * dzv_C
            delta_W = -eta * np.outer(dzv_C, av_prev)

            # Compute new intermediary derivatives
            dav_C = np.transpose(W).dot(dzv_C)
            dzv_C = dav_C * sigmoid_prime(av_prev)

            # Adjust biases and weights by their deltas
            bv += delta_bv
            W += delta_W

    #---------------------------------------------------------------------------

    # Public methods

    def train(self, examples, error_function=CROSS_ENTROPY, eta=0.25, nd=1e-5, max_rounds=None, manual_stop=False):
        '''
        Trains the network using the list `examples` of training examples. This
        is a list of 2-tuples (training input, training label), where "training
        input" is a vector (`np.array`) with the same length as the number of
        inputs in the network, and "training label" is the correct
        classification for that input, indexed from 0. Returns the number of
        rounds the network trained for. (...)

        (see docs/API.md for full description)
        '''
        # Validate parameters
        for xv, _ in examples:
            n, _, _ = self.layers[0]
            if xv.shape != (n,):
                raise ValueError('Training inputs must be the same length as the number of inputs in the network')
        if (nd != None) and (nd <= 0):
            raise ValueError('nd must be positive')
        if (max_rounds != None) and (max_rounds <= 0):
            raise ValueError('max_rounds must be positive')
        if (nd == None) and (max_rounds == None):
            raise ValueError('One of nd or max_rounds must be set')


        # Helper functions
        def train_one_round():
            for xv, y in examples:
                avs = self.feedforward(xv)                       # feedforward xv
                self.backpropagate(y, avs, eta, error_function)  # backpropagate error

        def train_one_round_toward_settling():
            # Compare new weights/biases to old weights/biases. Return True
            # if we reached the settling point, False if not
            old_layers = deepcopy(self.layers)
            train_one_round()
            settled = True
            for i in range(1, self.L):
                _, W_old, bv_old = old_layers[i]
                _, W, bv = self.layers[i]
                if np.any(np.abs(W - W_old) > nd) or np.any(np.abs(bv - bv_old) > nd):
                    settled = False
                    break
            return settled

        def record_and_print_stats(r):
            # Record total error
            n, _, _ = self.layers[-1]
            avs = [self.feedforward(xv)[-1] for xv, _ in examples]
            yvs = [NeuralNetwork.basis_vector(y, n) for _, y in examples]
            error = sum([NeuralNetwork.error(av, yv, error_function) for av, yv in zip(avs, yvs)])
            self.errors.append(error)

            # Record accuracy
            accuracy = self.test(examples)
            self.accuracies.append(accuracy)

            self.last_round = r

            # Print stats for this round
            print '%6d | %15.10f | %15.10f' % (r, error, accuracy)

        def ask_to_stop(r):
            # Pause and ask user if they want to stop every 5th round, if manual
            # stop is enabled. Return True if user stops training, False if not.
            round_interval = 5
            pause_time = 5
            if manual_stop and (r % round_interval == 0):
                print '[PAUSE] Press Ctrl-C to stop training, or wait 5 seconds to continue'
                try:
                    sleep(pause_time)
                except KeyboardInterrupt:
                    return True
            return False


        # Train over all rounds
        print '%6s | %15s | %15s' % ('Round', 'Error', 'Accuracy')
        print '-' * 42
        if (nd != None) and (max_rounds != None):
            # Train until network settles at nd, or until we pass max_rounds
            for r in range(1, max_rounds+1):
                settled = train_one_round_toward_settling()
                record_and_print_stats(r)
                if ask_to_stop(r): return r
                if settled: return r
            return r
        elif nd != None:
            # Train until network settles at nd
            r = 1
            while True:
                settled = train_one_round_toward_settling()
                record_and_print_stats(r)
                if ask_to_stop(r): return r
                if settled: return r
                r += 1
        else:
            # Train for max_rounds rounds
            for r in range(1, max_rounds+1):
                train_one_round()
                record_and_print_stats(r)
                if ask_to_stop(r): return r
            return r


    def plot_error(self, start=1, end=None):
        '''
        During training, the sum of error over all examples is recorded at each
        round. This function plots these measurements on a graph. (...)

        (see docs/API.md for full description)
        '''
        if self.last_round < 1:
            raise Exception('Network has not been trained yet!')
        if end == None:
            end = self.last_round

        actual_start = start - 1  # correct index to start at 0
        plt.plot(self.errors[actual_start:end])
        plt.show()


    def plot_accuracy(self, start=1, end=None):
        '''
        Same as `plot_error()`, but it plots the accuracy rate of the network
        on all examples.
        '''
        if self.last_round < 1:
            raise Exception('Network has not been trained yet!')
        if end == None:
            end = self.last_round

        actual_start = start - 1  # correct index to start at 0
        plt.plot(self.accuracies[actual_start:end])
        plt.show()


    def test(self, examples):
        '''
        Tests the network on the list `examples`, which has the same format as
        the list provided to train(). Returns the accuracy rate, a floating
        point number out of 1.
        '''
        # Validate length of inputs/outputs in examples
        for xv, _ in examples:
            n, _, _ = self.layers[0]
            if xv.shape != (n,):
                raise ValueError('Training inputs must be the same length as the number of inputs in the network')

        # Test over all examples
        num_correct = 0
        for xv, y in examples:
            y_test = self.classify(xv)
            if y_test == y:
                num_correct += 1
        return float(num_correct) / float(len(examples))


    def classify(self, xv):
        '''
        Returns the classification chosen by the network for input xv (the index
        of the output with highest probability, indexed from 0).
        '''
        yv = self.feedforward(xv)[-1]
        max_index = 0
        max_value = yv[0]
        for i in range(0, yv.size):
            if yv[i] > max_value:
                max_value = yv[i]
                max_index = i
        return max_index


    @staticmethod
    def save(nn, filename):
        with open(filename, 'wb') as f:
            cPickle.dump(nn, f)


    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return cPickle.load(f)
