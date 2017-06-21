#-------------------------------------------------------------------------------
# neural_network.py
# Author: Michael Friedman
#
# Library implementing a simple neural network based on Nielson's book:
#   http://neuralnetworksanddeeplearning.com/index.html
# This implements feedforward and backpropagation-based learning in a network
# containing variable number of inputs, outputs, and hidden layers, with a
# client-defined learning rate.
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

import numpy as np

class NeuralNetwork(object):

    class Layer(object):
        '''
        Container for attributes of each layer
        '''

        def __init__(n, W, bv):
            '''
            Sets attributes of this layer: `n` the number of neurons, `W` the
            matrix of weights coming *in* to this layer, `bv` the vector of
            biases coming *in* to this layer.
            '''
            self.n = n
            self.W = W
            self.bv = bv

    #---------------------------------------------------------------------------

    def __init__(layers, eta=0.25):
        '''
        Initializes the NeuralNetwork given `layers`, a list of integers
        indicating the number of neurons in each layer of the network. (The
        first and last value in this list are thus the number of inputs and
        outputs of the network, respectively.) Values in this list must all be
        positive numbers, or an exception will be raised.

        Also optionally provide `eta`, the learning rate.
        '''
        # Validate parameters
        for n in layers:
            if n <= 0:
                raise ValueError('Number of neurons in each layer must be positive')

        # Initialize basic parameters
        self.L = len(layers)
        self.eta = eta

        # Initialize weights arbitrarily. Each consecutive pair of layers
        # (prev/next layers) has a matrix of weights.
        weights = [np.empty(shape=(layers[i], layers[i+1])) for i in range(0, len(layers)-1)]

        # Initialize biases arbitrarily. Same principle as with weights, except
        # each pair of layers has a vector of biases.
        biases = [np.empty(shape=(layers[i+1],)) for i in range(0, len(layers)-1)]

        # Place info into list of Layer containers
        self.layers = [Layer(layers[0], None, None)] + [Layer(layers[i], weights[i-1], biases[i-1]) for i in range(1, len(layers))]


    #---------------------------------------------------------------------------

    # Helper methods

    def sigmoid(x):
        '''
        Returns a np array that applies the sigmoid function element-wise to the
        np array x.
        '''
        return 1 / (1 + np.exp(x))

    def sigmoid_prime(x):
        '''
        Returns a np array that applies the derivative of sigmoid element-wise
        to the np array x.
        '''
        return sigmoid(x) * (1 - sigmoid(x))

    #---------------------------------------------------------------------------

    # Public methods

    def train(examples):
        '''
        Trains the network using the list `examples` of training examples. This
        is a list of 2-tuples (training input, training label), where "training
        input" is a vector (np array) with the same length as the number of
        inputs in the network, and "training label" is the correct
        classification for that input, indexed from 0.

        Note that the inputs are *not* validated as numbers, so it's up to the
        caller to make sure they are numbers, or risk unpredictable errors.
        '''
        # Validate length of inputs/outputs in examples
        for xv, _ in examples:
            if xv.shape != (self.layers[0].n,):
                raise ValueError('Training inputs must be the same length as the number of inputs in the network')

        # Train over all examples
        for xv, y in examples:
            # Feedforward xv

            # Backpropagate and adjust weights and biases for next iteration
            pass
