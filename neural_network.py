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

        def __init__(self, n, W, bv):
            '''
            Sets attributes of this layer: `n` the number of neurons, `W` the
            matrix of weights coming *in* to this layer, `bv` the vector of
            biases coming *in* to this layer.
            '''
            self.n = n
            self.W = W
            self.bv = bv

    #---------------------------------------------------------------------------

    def __init__(self, layers, eta=0.25, weights=None, biases=None):
        '''
        Initializes the NeuralNetwork given `layers`, a list of integers
        indicating the number of neurons in each layer of the network. (The
        first and last value in this list are thus the number of inputs and
        outputs of the network, respectively.) Values in this list must all be
        positive numbers, or an exception will be raised.

        Also optionally provide `eta`, the learning rate.

        For testing purposes, also optionally provide `weights`, a list of
        initial weight matrices between each pair of layers. Each matrix
        corresponds to the weights applied to the *second* layer in the pair.
        Similarly, provide `biases`, the corresponding list of initial bias
        vectors.
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
        if not weights:
            weights = [np.empty(shape=(layers[i], layers[i+1])) for i in range(0, len(layers)-1)]

        # Initialize biases arbitrarily. Same principle as with weights, except
        # each pair of layers has a vector of biases.
        if not biases:
            biases = [np.empty(shape=(layers[i+1],)) for i in range(0, len(layers)-1)]

        # Place info into list of Layer containers
        self.layers = [NeuralNetwork.Layer(layers[0], None, None)] + [NeuralNetwork.Layer(layers[i], weights[i-1], biases[i-1]) for i in range(1, len(layers))]


    #---------------------------------------------------------------------------

    # Helper methods

    @staticmethod
    def sigmoid(x):
        '''
        Returns a np array that applies the sigmoid function element-wise to the
        np array x.
        '''
        return 1 / (1 + np.exp(-x))

    def feedforward(self, xv):
        '''
        Returns the list of activation vectors from each layer after feeding
        forward the input xv through the network. (Special notes: the first
        item in this list is the input xv, and the last item is the output
        vector.)
        '''
        av = xv
        avs = [av]
        for l in self.layers[1:]:
            # Compute activation vector at this layer
            zv = l.W.dot(av) + l.bv
            av = NeuralNetwork.sigmoid(zv)
            avs.append(av)
        return avs

    def backpropagate(self, y, avs):
        '''
        Compute error in the last layer from the correct label y, and
        backpropagate that error using the list avs of activations from each
        layer in the network (the result of feedforward(xv)).

        Refer to docs for the explanation of this algorithm, and for its
        derivation.
        '''
        # Convert y to vector form
        yv = np.zeros(self.layers[-1].n)
        yv[y] = 1

        # Definition of sigmoid_prime
        # NOTE: Since sigmoid_prime can be evaluated in terms of sigmoid, I
        # define it in terms of sigmoid, rather than the original input.
        # This saves us from having to store the original inputs zv to sigmoid.
        sigmoid_prime = lambda sigmoid: sigmoid * (1 - sigmoid)

        # Backpropagation algorithm
        av = avs[-1]
        dav_C = av - yv
        dzv_C = dav_C * sigmoid_prime(av)
        for i in reversed(range(1, self.L)):
            l = self.layers[i]
            av = avs[i]
            av_prev = avs[i-1]

            # Compute deltas in biases and weights
            delta_bv = -self.eta * dzv_C
            delta_W = -self.eta * np.outer(dzv_C, av_prev)

            # Compute new intermediary derivatives
            dav_C = np.transpose(l.W).dot(dzv_C)
            dzv_C = dav_C * sigmoid_prime(av_prev)

            # Adjust biases and weights by their deltas
            l.bv += delta_bv
            l.W += delta_W

    #---------------------------------------------------------------------------

    # Public methods

    def train(self, examples):
        '''
        Trains the network using the list `examples` of training examples. This
        is a list of 2-tuples (training input, training label), where "training
        input" is a vector (np.array) with the same length as the number of
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
            avs = self.feedforward(xv)

            # Backpropagate error
            self.backpropagate(y, avs)


    def test(self, examples):
        '''
        Tests the network on the list `examples`, which has the same format as
        the list provided to train(). Returns the accuracy rate, a floating
        point number out of 1.
        '''
        # Validate length of inputs/outputs in examples
        for xv, _ in examples:
            if len(xv.shape) != 1 and xv.shape != (self.layers[0].n,):
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
