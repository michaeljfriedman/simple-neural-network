#-------------------------------------------------------------------------------
# tests.py
# Author: Michael Friedman
#
# Tests for neural_network.py.
#-------------------------------------------------------------------------------

from neural_network import NeuralNetwork
import numpy as np
import os
import unittest

# Some helper functions to make preset neural networks

def make_1_1():
    '''
    Makes a [1, 1] network with preset weights and biases. Returns the network,
    the weight matrix, and the bias vector in a list.
    '''
    W = np.array([0.3]).reshape(1, 1)
    bv = np.array([0.5])
    nn = NeuralNetwork(layers=[1, 1], weights=[W], biases=[bv])
    return [nn, W, bv]

def make_3_1():
    '''
    Makes a [3, 1] network with preset weights and biases. Returns the network,
    the weight matrix, and the bias vector in a list.
    '''
    W = np.array([0.7, 0.11, 0.13]).reshape(1, 3)
    bv = np.array([0.17])
    nn = NeuralNetwork(layers=[3, 1], weights=[W], biases=[bv])
    return [nn, W, bv]

def make_1_3():
    '''
    Makes a [1, 3] network with preset weights and biases. Returns the network,
    the weight matrix, and the bias vector in a list.
    '''
    W = np.array([0.3, 0.5, 0.7]).reshape(3, 1)
    bv = np.array([0.11, 0.13, 0.17])
    nn = NeuralNetwork(layers=[1, 3], weights=[W], biases=[bv])
    return [nn, W, bv]

def make_3_4():
    '''
    Makes a [3, 4] network with preset weights and biases. Returns the network,
    the weight matrix, and the bias vector in a list.
    '''
    W = np.array([0.7, 0.11, 0.13, 0.17, 0.19, 0.23, 0.29, 0.31, 0.37, 0.41, 0.43, 0.47]).reshape(4, 3)
    bv = np.array([0.53, 0.59, 0.61, 0.67])
    nn = NeuralNetwork(layers=[3, 4], weights=[W], biases=[bv])
    return [nn, W, bv]

def make_1_1_1():
    '''
    Makes a [1, 1, 1] network with preset weights and biases. Returns the
    network, the list of weight matrices, and the list of bias vectors in a
    list.
    '''
    W1 = np.array([0.3]).reshape(1, 1)
    bv1 = np.array([0.5])
    W2 = np.array([0.03]).reshape(1, 1)
    bv2 = np.array([0.05])
    nn = NeuralNetwork(layers=[1, 1, 1], weights=[W1, W2], biases=[bv1, bv2])
    return [nn, [W1, W2], [bv1, bv2]]

def make_1_3_1():
    '''
    Makes a [1, 3, 1] network with preset weights and biases. Returns the
    network, the list of weight matrices, and the list of bias vectors in a
    list.
    '''
    W1 = np.array([0.3, 0.5, 0.7]).reshape(3, 1)
    bv1 = np.array([0.11, 0.13, 0.17])
    W2 = np.array([0.03, 0.05, 0.07]).reshape(1, 3)
    bv2 = np.array([0.011])
    nn = NeuralNetwork(layers=[1, 3, 1], weights=[W1, W2], biases=[bv1, bv2])
    return [nn, [W1, W2], [bv1, bv2]]

def make_2_3_2():
    '''
    Makes a [2, 3, 2] network with preset weights and biases. Returns the
    network, the list of weight matrices, and the list of bias vectors in a
    list.
    '''
    W1 = np.array([0.2, 0.3, 0.5, 0.7, 0.11, 0.13]).reshape(3,2)
    bv1 = np.array([0.17, 0.19, 0.23])
    W2 = np.array([0.02, 0.03, 0.05, 0.07, 0.011, 0.013]).reshape(2, 3)
    bv2 = np.array([0.017, 0.019])
    nn = NeuralNetwork(layers=[2, 3, 2], weights=[W1, W2], biases=[bv1, bv2])
    return [nn, [W1, W2], [bv1, bv2]]

def make_2_3_4():
    '''
    Makes a [2, 3, 4] network with preset weights and biases. Returns the
    network, the list of weight matrices, and the list of bias vectors in a
    list.
    '''
    W1 = np.array([0.2, 0.3, 0.5, 0.7, 0.11, 0.13]).reshape(3,2)
    bv1 = np.array([0.17, 0.19, 0.23])
    W2 = np.array([0.02, 0.03, 0.05, 0.07, 0.011, 0.013, 0.017, 0.019, 0.023, 0.029, 0.031, 0.037]).reshape(4, 3)
    bv2 = np.array([0.041, 0.043, 0.047, 0.053])
    nn = NeuralNetwork(layers=[2, 3, 4], weights=[W1, W2], biases=[bv1, bv2])
    return [nn, [W1, W2], [bv1, bv2]]


class TestFeedforwardAndEvaluate(unittest.TestCase):
    '''
    Tests for the functions feedforward() and classify().
    '''

    #---------------------------------
    # No hidden layers
    #---------------------------------

    def test_1_1(self):
        '''
        Test results of feedfoward() and classify() on 1 input and 1 output.
        '''
        nn, W, bv = make_1_1()
        xv = np.array([0.2])
        av = NeuralNetwork.sigmoid(np.array([0.56]))  # W.dot(xv) + bv
        y = 0

        self.assertTrue(np.array_equal(nn.feedforward(xv)[-1], av), 'feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y, 'classify() incorrect')


    def test_3_1(self):
        '''
        Test results of feedfoward() and classify() on 3 inputs and 1 output.
        '''
        nn, W, bv = make_3_1()
        xv = np.array([0.2, 0.3, 0.5])
        av = NeuralNetwork.sigmoid(np.array([0.408]))  # W.dot(xv) + bv
        y = 0

        self.assertTrue(np.array_equal(nn.feedforward(xv)[-1], av), 'feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y, 'classify() incorrect')


    def test_1_3(self):
        '''
        Test results of feedfoward() and classify() on 1 input and 3 outputs.
        '''
        nn, W, bv = make_1_3()
        xv = np.array([0.2])
        av = NeuralNetwork.sigmoid(np.array([0.17, 0.23, 0.31]))  # W.dot(xv) + bv
        y = 2

        self.assertTrue(np.array_equal(nn.feedforward(xv)[-1], av), 'feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y, 'classify() incorrect')


    def test_3_4(self):
        '''
        Test results of feedfoward() and classify() on 3 inputs and 4 outputs.
        '''
        nn, W, bv = make_3_4()
        xv = np.array([0.2, 0.3, 0.5])
        av = NeuralNetwork.sigmoid(np.array([0.768, 0.796, 0.946, 1.116]))  # W.dot(xv) + bv
        y = 3

        self.assertTrue(np.array_equal(nn.feedforward(xv)[-1], av), 'feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y, 'classify() incorrect')

    #---------------------------------
    # 1 hidden layer
    #---------------------------------


    def test_1_1_1(self):
        '''
        Test results of feedforward() and classify() on 1 input, layer of 1,
        and 1 output.
        '''
        nn, weights, biases = make_1_1_1()
        xv = np.array([0.2])
        av = NeuralNetwork.sigmoid(np.array([0.06909358]))
        y = 0

        np.testing.assert_allclose(nn.feedforward(xv)[-1], av, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y, 'classify() incorrect')


    def test_1_3_1(self):
        '''
        Test results of feedforward() and classify() on 1 input, layer of 3,
        and 1 output.
        '''
        nn, weights, biases = make_1_3_1()
        xv = np.array([0.2])
        av = NeuralNetwork.sigmoid(np.array([0.0955163]))
        y = 0

        np.testing.assert_allclose(nn.feedforward(xv)[-1], av, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y, 'classify() incorrect')


    def test_2_3_4(self):
        '''
        Test results of feedforward() and classify() on 2 inputs, layer of 3,
        and 4 outputs.
        '''
        nn, weights, biases = make_2_3_4()
        xv = np.array([0.1, 0.2])
        av = NeuralNetwork.sigmoid(np.array([0.09837754, 0.09624759, 0.08086678, 0.10866837]))
        y = 3

        np.testing.assert_allclose(nn.feedforward(xv)[-1], av, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y, 'classify() incorrect')


    #---------------------------------
    # Test without predefined weights
    # and biases
    #---------------------------------

    def test_network_defined_weights_and_biases(self):
        '''
        Tests that when the network creates the weight matrices and bias
        vectors that feedforward() still works (i.e. no crashes) due to wrong
        dimensions.
        '''
        nn = NeuralNetwork(layers=[2, 3, 4])
        xv = np.array([0.1, 0.2])
        try:
            nn.feedforward(xv)
        except Exception as e:
            raise Exception('feedforward() failed when the network created weight matrices and bias vectors: ' + str(e))


class TestTrainEuclideanDistance(unittest.TestCase):
    '''
    Tests for train() (i.e. the backpropagation algorithm) using the Euclidean
    distance error function
    '''

    def test_returns_num_rounds(self):
        '''
        Tests that train() returns the correct number of rounds of training.
        '''
        nn, W, bv = make_3_4()
        xv = np.array([0.2, 0.3, 0.5])
        y = 0
        examples = [(xv, y)]

        rounds = nn.train(examples, nd=None, max_rounds=10)
        self.assertEqual(rounds, 10, 'nn.train() returns incorrect number of rounds')

    #-------------------------------------
    # Training on a network with only one
    # output (should yield no change)
    #-------------------------------------

    def test_1_1_train_1(self):
        '''
        Tests that the [1, 1] network yields the correct activation and output
        after 1 training example
        '''
        nn, W, bv = make_1_1()
        xv = np.array([0.2])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.64149773])  # result after training on xv
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y_new, 'classify() incorrect')


    def test_3_1_train_1(self):
        '''
        Tests that the [3, 1] network yields the correct activation and output
        after 1 training example -- namely, no change, since the output will
        always be correct.
        '''
        nn, W, bv = make_3_1()
        xv = np.array([0.2, 0.3, 0.5])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.60850987])  # result after training on xv
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y_new, 'classify() incorrect')


    #--------------------------------------
    # No hidden layers, 1 training example
    #--------------------------------------

    def test_1_3_train_1(self):
        '''
        Tests that the [1, 3] network yields the correct activation and output
        after 1 training example.
        '''
        nn, W, bv = make_1_3()
        xv = np.array([0.2])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.54971773, 0.54841129, 0.56792473])  # result after training on xv
        y_new = 2

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y_new, 'classify() incorrect')


    def test_3_4_train_1(self):
        '''
        Tests that the [3, 4] network yields the correct activation and output
        after 1 training example.
        '''
        nn, W, bv = make_3_4()
        xv = np.array([0.2, 0.3, 0.5])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.68818950, 0.67810281, 0.71011341, 0.74415907])  # result after training on xv
        y_new = 3

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y_new, 'classify() incorrect')


    #----------------------------------------
    # No hidden layers, 2 training examples
    #----------------------------------------

    def test_1_3_train_2(self):
        '''
        Tests that the [1, 3] network yields the correct activation and output
        after 2 training examples.
        '''
        nn, W, bv = make_1_3()
        xv1 = np.array([0.2])
        y1 = 0
        xv2 = np.array([0.02])
        y2 = 1
        examples = [(xv1, y1), (xv2, y2)]

        av_new = np.array([0.54142166,  0.55574055,  0.55968207])  # result of xv1 after training
        y_new = 2

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv1)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv1), y_new, 'classify() incorrect')


    def test_3_4_train_2(self):
        '''
        Tests that the [3, 4] network yields the correct activation and output
        after 2 training examples.
        '''
        nn, W, bv = make_3_4()
        xv1 = np.array([0.2, 0.3, 0.5])
        y1 = 0
        xv2 = np.array([0.02, 0.03, 0.05])
        y2 = 1
        examples = [(xv1, y1), (xv2, y2)]

        av_new = np.array([0.67992246,  0.68279612,  0.70215662,  0.73677188])  # result of xv1 after training
        y_new = 3

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv1)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv1), y_new, 'classify() incorrect')


    #-------------------------------------
    # 1 hidden layer, 1 training example
    #-------------------------------------

    def test_2_3_2_train_1(self):
        '''
        Tests that the [2, 3, 2] network yields the correct activation and
        output after 1 training example.
        '''
        nn, weights, biases = make_2_3_2()
        xv = np.array([0.1, 0.2])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.53348918, 0.50197726])  # result of xv after training
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y_new, 'classify() incorrect')


    def test_2_3_4_train_1(self):
        '''
        Tests that the [2, 3, 2] network yields the correct activation and
        output after 1 training example.
        '''
        nn, weights, biases = make_2_3_4()
        xv = np.array([0.1, 0.2])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.53925307, 0.50780625, 0.50407204, 0.51082614])  # result of xv after training
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y_new, 'classify() incorrect')



    #--------------------------------------
    # 1 hidden layer, 2 training examples
    #--------------------------------------

    def test_2_3_2_train_2(self):
        '''
        Tests that the [2, 3, 2] network yields the correct activation and
        output after 2 training examples.
        '''
        nn, weights, biases = make_2_3_2()
        xv1 = np.array([0.1, 0.2])
        y1 = 0
        xv2 = np.array([0.01, 0.02])
        y2 = 1
        examples = [(xv1, y1), (xv2, y2)]

        av_new = np.array([0.51736881, 0.51714835])  # result of xv1 after training
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv1)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv1), y_new, 'classify() incorrect')


    def test_2_3_4_train_2(self):
        '''
        Tests that the [2, 3, 4] network yields the correct activation and
        output after 2 training examples.
        '''
        nn, weights, biases = make_2_3_4()
        xv1 = np.array([0.1, 0.2])
        y1 = 0
        xv2 = np.array([0.01, 0.02])
        y2 = 1
        examples = [(xv1, y1), (xv2, y2)]

        av_new = np.array([0.52300829, 0.52278460, 0.48872059, 0.49527995])  # result of xv1 after training
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv1)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv1), y_new, 'classify() incorrect')


    #--------------------------------------
    # Training for multiple rounds
    #--------------------------------------

    def test_3_4_train_1_for_2_rounds(self):
        '''
        Tests that the [3, 4] network yields correct activation and output
        after training 1 example for 2 time steps.
        '''
        nn, W, bv = make_3_4()
        xv = np.array([0.2, 0.3, 0.5])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.69312131, 0.66685643, 0.69962301, 0.73474270])
        y_new = 3

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=None, max_rounds=2)
        np.testing.assert_allclose(nn.feedforward(xv)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y_new, 'classify() incorrect')


    def test_3_4_train_2_for_2_rounds(self):
        '''
        Tests that the [3, 4] network yields correct activation and output
        after training 2 examples for 2 time steps.
        '''
        nn, W, bv = make_3_4()
        xv1 = np.array([0.2, 0.3, 0.5])
        y1 = 0
        xv2 = np.array([0.02, 0.03, 0.05])
        y2 = 1
        examples = [(xv1, y1), (xv2, y2)]

        av_new = np.array([0.67683375, 0.67646289, 0.68326675, 0.71942654])
        y_new = 3

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=None, max_rounds=2)
        np.testing.assert_allclose(nn.feedforward(xv1)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv1), y_new, 'classify() incorrect')


    def test_3_4_train_1_until_network_settles(self):
        '''
        Tests that the [3, 4] network yields correct activation and output
        after training 1 example until the network "settles" within a factor
        of 1/1000.
        '''
        nn, W, bv = make_3_4()
        xv = np.array([0.2, 0.3, 0.5])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.93825760, 0.06477398, 0.06501966, 0.06530958])
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=1e-3)
        np.testing.assert_allclose(nn.feedforward(xv)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y_new, 'classify() incorrect')


    def test_2_3_4_train_1_until_network_settles(self):
        '''
        Tests that the [2, 3, 4] network yields correct activation and output
        after training 1 example until the network "settles" within a factor
        of 1/1000.
        '''
        nn, weights, biases = make_2_3_4()
        xv = np.array([0.1, 0.2])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.93545749, 0.06508182, 0.06500411, 0.06509811])
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=1e-3)
        np.testing.assert_allclose(nn.feedforward(xv)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y_new, 'classify() incorrect')


    def test_2_3_4_train_2_until_network_settles(self):
        '''
        Tests that the [2, 3, 4] network yields correct activation and output
        after training 2 examples until the network "settles" within a factor
        of 1/1000.
        '''
        nn, weights, biases = make_2_3_4()
        xv1 = np.array([0.1, 0.2])
        y1 = 0
        xv2 = np.array([0.01, 0.02])
        y2 = 1
        examples = [(xv1, y1), (xv2, y2)]

        av_new = np.array([0.91118657, 0.08873599, 0.00922352, 0.00923353])
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.EUCLIDEAN_DISTANCE, eta=0.25, nd=1e-3)
        np.testing.assert_allclose(nn.feedforward(xv1)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv1), y_new, 'classify() incorrect')


class TestTrainCrossEntropy(unittest.TestCase):
    '''
    Tests for train() (i.e. the backpropagation algorithm) using the cross
    entropy error function.

    Since the algorithm for training is the same as with Euclidean distance,
    just with different initialization, I assume that the tests from the
    previous class sufficiently test the algorithm. So the purpose of this
    class is simply to test a few cases to check that the cross-entropy-based
    version of the algorithm computes the correct output.
    '''

    def test_3_4_train_1(self):
        '''
        Tests that the [3, 4] network yields the correct activation and output
        after 1 training example.
        '''
        nn, W, bv = make_3_4()
        xv = np.array([0.2, 0.3, 0.5])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.70626958, 0.63604851, 0.66763171, 0.70185149])  # result after training on xv
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.CROSS_ENTROPY, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y_new, 'classify() incorrect')


    def test_3_4_train_2(self):
        '''
        Tests that the [3, 4] network yields the correct activation and output
        after 2 training examples.
        '''
        nn, W, bv = make_3_4()
        xv1 = np.array([0.2, 0.3, 0.5])
        y1 = 0
        xv2 = np.array([0.02, 0.03, 0.05])
        y2 = 1
        examples = [(xv1, y1), (xv2, y2)]

        av_new = np.array([0.66988359, 0.65935475, 0.63149142, 0.66673127])  # result of xv1 after training
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.CROSS_ENTROPY, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv1)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv1), y_new, 'classify() incorrect')


    def test_2_3_4_train_1(self):
        '''
        Tests that the [2, 3, 2] network yields the correct activation and
        output after 1 training example.
        '''
        nn, weights, biases = make_2_3_4()
        xv = np.array([0.1, 0.2])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.58290633, 0.45900184, 0.45566781, 0.46171057])  # result of xv after training
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.CROSS_ENTROPY, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y_new, 'classify() incorrect')


    def test_2_3_4_train_2(self):
        '''
        Tests that the [2, 3, 4] network yields the correct activation and
        output after 2 training examples.
        '''
        nn, weights, biases = make_2_3_4()
        xv1 = np.array([0.1, 0.2])
        y1 = 0
        xv2 = np.array([0.01, 0.02])
        y2 = 1
        examples = [(xv1, y1), (xv2, y2)]

        av_new = np.array([0.51287332, 0.52466737, 0.40130401, 0.40647212])  # result of xv1 after training
        y_new = 1

        nn.train(examples, error_function=NeuralNetwork.CROSS_ENTROPY, eta=0.25, nd=None, max_rounds=1)
        np.testing.assert_allclose(nn.feedforward(xv1)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv1), y_new, 'classify() incorrect')


    def test_3_4_train_2_for_2_rounds(self):
        '''
        Tests that the [3, 4] network yields correct activation and output
        after training 2 examples for 2 time steps.
        '''
        nn, W, bv = make_3_4()
        xv1 = np.array([0.2, 0.3, 0.5])
        y1 = 0
        xv2 = np.array([0.02, 0.03, 0.05])
        y2 = 1
        examples = [(xv1, y1), (xv2, y2)]

        av_new = np.array([0.65848101, 0.63172401, 0.54537277, 0.57967869])
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.CROSS_ENTROPY, eta=0.25, nd=None, max_rounds=2)
        np.testing.assert_allclose(nn.feedforward(xv1)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv1), y_new, 'classify() incorrect')


    def test_3_4_train_1_until_network_settles(self):
        '''
        Tests that the [3, 4] network yields correct activation and output
        after training 1 example until the network "settles" within a factor
        of 1/1000.
        '''
        nn, W, bv = make_3_4()
        xv = np.array([0.2, 0.3, 0.5])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.99606464, 0.00398431, 0.00398756, 0.0039911])
        y_new = 0

        nn.train(examples, error_function=NeuralNetwork.CROSS_ENTROPY, eta=0.25, nd=1e-3)
        np.testing.assert_allclose(nn.feedforward(xv)[-1], av_new, atol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.classify(xv), y_new, 'classify() incorrect')


class TestProgressStats(unittest.TestCase):
    '''
    Tests to make sure progress stats are computed correctly during training.
    '''

    def test_euclidean_distance_error(self):
        '''
        Tests that Euclidean distance error function is computed correctly.
        '''
        v = np.array([1, 1, 1])
        w = np.array([0, 0, 0])
        error = 1.5
        np.testing.assert_almost_equal(NeuralNetwork.error(v, w, NeuralNetwork.EUCLIDEAN_DISTANCE),
            error, decimal=7, err_msg='Cross entropy error computed incorrectly.')


    def test_cross_entropy_error(self):
        '''
        Tests that cross entropy error function is computed correctly.
        '''
        v = np.array([0.2, 0.3, 0.5])
        w = np.array([0.1, 0.15, 0.25])
        error = -np.sum(w * np.log(v) + (1 - w) * np.log(1 - v))
        np.testing.assert_almost_equal(NeuralNetwork.error(v, w, NeuralNetwork.CROSS_ENTROPY),
            error, decimal=7, err_msg='Cross entropy error computed incorrectly.')

    def test_error_stats_while_training(self):
        '''
        Tests that error stats are computed correctly during training.
        '''
        nn, _, _ = make_2_3_4()
        xv = np.array([0.1, 0.2])
        y = 0
        examples = [(xv, y)]

        errors = [2.3816226, 2.0046691, 1.7105535]
        nn.train(examples, error_function=NeuralNetwork.CROSS_ENTROPY, nd=None, max_rounds=3)
        np.testing.assert_allclose(nn.errors, errors, atol=1e-7, err_msg='nn.errors computed incorrectly')

    def test_accuracy_stats_while_training(self):
        '''
        Tests that accuracy stats are computed correctly during training.
        '''
        nn, _, _ = make_2_3_2()
        xv1 = np.array([0.1, 0.2])
        y1 = 0
        xv2 = np.array([0.01, 0.02])
        y2 = 1
        xv3 = np.array([0.3, 0.5])
        y3 = 0
        xv4 = np.array([0.03, 0.05])
        y4 = 1
        examples = [(xv1, y1), (xv2, y2), (xv3, y3), (xv4, y4)]

        accuracies = ([0.5] * 37) + ([0.75] * 3)
        nn.train(examples, error_function=NeuralNetwork.CROSS_ENTROPY, nd=None, max_rounds=40)
        self.assertEqual(nn.accuracies, accuracies,'nn.accuracies computed incorrectly')

    def test_num_rounds_stat_while_training(self):
        '''
        Tests that the last_round stat is computed correctly during training.
        '''
        nn, _, _ = make_2_3_4()
        xv = np.array([0.1, 0.2])
        y = 0
        examples = [(xv, y)]

        nn.train(examples, nd=None, max_rounds=3)
        self.assertEqual(nn.last_round, 3, 'nn.last_round is incorrect')


class TestSaveLoad(unittest.TestCase):
    '''
    Tests for the save() and load() methods
    '''

    def test_save_load_success(self):
        '''
        Tests that a NeuralNetwork saved to a file can be loaded back again
        to the same object.
        '''
        nn, _, _ = make_3_4()
        filename = '__tmp.pickle'
        NeuralNetwork.save(nn, filename)
        nn2 = NeuralNetwork.load(filename)
        self.assertEqual(nn2, nn, 'NeuralNetwork loaded from file does not match original')
        os.remove(filename)

    def test_save_load_failure(self):
        '''
        Tests that a NeuralNetwork that's modified after being saved to a
        file is different from the saved one.
        '''
        nn, _, _ = make_3_4()
        filename = '__tmp.pickle'
        NeuralNetwork.save(nn, filename)

        # Modify nn
        xv = np.array([0.2, 0.3, 0.5])
        y = 3
        examples = [(xv, y)]
        nn.train(examples, nd=None, max_rounds=1)

        nn2 = NeuralNetwork.load(filename)
        self.assertNotEqual(nn2, nn, 'NeuralNetwork saved to disk is the same as the original *after* modifying the original')
        os.remove(filename)


if __name__ == '__main__':
    unittest.main()
