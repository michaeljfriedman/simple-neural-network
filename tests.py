#-------------------------------------------------------------------------------
# tests.py
# Author: Michael Friedman
#
# Tests for neural_network.py.
#-------------------------------------------------------------------------------

from neural_network import NeuralNetwork
import numpy as np
import unittest

class TestFeedforwardAndEvaluate(unittest.TestCase):
    '''
    Tests for the functions feedforward() and evaluate().
    '''

    #---------------------------------
    # No hidden layers
    #---------------------------------

    def test_1_1(self):
        '''
        Test results of feedfoward() and evaluate() on 1 input and 1 output.
        '''
        xv = np.array([0.2])
        W = np.array([0.3]).reshape(1, 1)
        bv = np.array([0.5])
        av = NeuralNetwork.sigmoid(np.array([0.56]))  # W.dot(xv) + bv
        y = 0

        nn = NeuralNetwork(layers=[1, 1], weights=[W], biases=[bv])
        self.assertTrue(np.array_equal(nn.feedforward(xv), av), 'feedforward() returned incorrect value')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() returned incorrect value')


    def test_3_1(self):
        '''
        Test results of feedfoward() and evaluate() on 3 inputs and 1 output.
        '''
        xv = np.array([0.2, 0.3, 0.5])
        W = np.array([0.7, 0.11, 0.13]).reshape(1, 3)
        bv = np.array([0.17])
        av = NeuralNetwork.sigmoid(np.array([0.408]))  # W.dot(xv) + bv
        y = 0

        nn = NeuralNetwork(layers=[1, 1], weights=[W], biases=[bv])
        self.assertTrue(np.array_equal(nn.feedforward(xv), av), 'feedforward() returned incorrect value')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() returned incorrect value')


    def test_1_3(self):
        '''
        Test results of feedfoward() and evaluate() on 1 input and 3 outputs.
        '''
        xv = np.array([0.2])
        W = np.array([0.3, 0.5, 0.7]).reshape(3, 1)
        bv = np.array([0.11, 0.13, 0.17])
        av = NeuralNetwork.sigmoid(np.array([0.17, 0.23, 0.31]))  # W.dot(xv) + bv
        y = 2

        nn = NeuralNetwork(layers=[1, 1], weights=[W], biases=[bv])
        self.assertTrue(np.array_equal(nn.feedforward(xv), av), 'feedforward() returned incorrect value')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() returned incorrect value')


    def test_3_4(self):
        '''
        Test results of feedfoward() and evaluate() on 3 inputs and 4 outputs.
        '''
        xv = np.array([0.2, 0.3, 0.5])
        W = np.array([0.7, 0.11, 0.13, 0.17, 0.19, 0.23, 0.29, 0.31, 0.37, 0.41, 0.43, 0.47]).reshape(4, 3)
        bv = np.array([0.53, 0.59, 0.61, 0.67])
        av = NeuralNetwork.sigmoid(np.array([0.768, 0.796, 0.946, 1.116]))  # W.dot(xv) + bv
        y = 3

        nn = NeuralNetwork(layers=[1, 1], weights=[W], biases=[bv])
        self.assertTrue(np.array_equal(nn.feedforward(xv), av), 'feedforward() returned incorrect value')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() returned incorrect value')

    #---------------------------------
    # 1 hidden layer
    #---------------------------------


    def test_1_1_1(self):
        '''
        Test results of feedforward() and evaluate() on 1 input, layer of 1,
        and 1 output.
        '''
        xv = np.array([0.2])
        W1 = np.array([0.3]).reshape(1, 1)
        bv1 = np.array([0.5])
        W2 = np.array([0.03]).reshape(1, 1)
        bv2 = np.array([0.05])
        av = NeuralNetwork.sigmoid(np.array([0.06909358]))
        y = 0

        nn = NeuralNetwork(layers=[1, 1, 1], weights=[W1, W2], biases=[bv1, bv2])
        np.testing.assert_allclose(nn.feedforward(xv), av, rtol=1e-8, err_msg='feedforward() returned incorrect value')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() returned incorrect value')


    def test_1_3_1(self):
        '''
        Test results of feedforward() and evaluate() on 1 input, layer of 3,
        and 1 output.
        '''
        xv = np.array([0.2])
        W1 = np.array([0.3, 0.5, 0.7]).reshape(3, 1)
        bv1 = np.array([0.11, 0.13, 0.17])
        W2 = np.array([0.03, 0.05, 0.07]).reshape(1, 3)
        bv2 = np.array([0.011])
        av = NeuralNetwork.sigmoid(np.array([0.0955163]))
        y = 0

        nn = NeuralNetwork(layers=[1, 3, 1], weights=[W1, W2], biases=[bv1, bv2])
        np.testing.assert_allclose(nn.feedforward(xv), av, rtol=1e-8, err_msg='feedforward() returned incorrect value')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() returned incorrect value')


    def test_2_3_4(self):
        '''
        Test results of feedforward() and evaluate() on 2 inputs, layer of 3,
        and 4 outputs.
        '''
        xv = np.array([0.1, 0.2])
        W1 = np.array([0.2, 0.3, 0.5, 0.7, 0.11, 0.13]).reshape(3,2)
        bv1 = np.array([0.17, 0.19, 0.23])
        W2 = np.array([0.02, 0.03, 0.05, 0.07, 0.011, 0.013, 0.017, 0.019, 0.023, 0.029, 0.031, 0.037]).reshape(4, 3)
        bv2 = np.array([0.041, 0.043, 0.047, 0.053])
        av = NeuralNetwork.sigmoid(np.array([0.09837754, 0.09624759, 0.08086678, 0.10866837]))
        y = 3

        nn = NeuralNetwork(layers=[2, 3, 4], weights=[W1, W2], biases=[bv1, bv2])
        np.testing.assert_allclose(nn.feedforward(xv), av, rtol=1e-8, err_msg='feedforward() returned incorrect value')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() returned incorrect value')


    #---------------------------------
    # 3 hidden layers
    #---------------------------------



if __name__ == '__main__':
    unittest.main()
