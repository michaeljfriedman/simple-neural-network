#-------------------------------------------------------------------------------
# tests.py
# Author: Michael Friedman
#
# Tests for neural_network.py.
#-------------------------------------------------------------------------------

from neural_network import NeuralNetwork
import numpy as np
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
    Tests for the functions feedforward() and evaluate().
    '''

    #---------------------------------
    # No hidden layers
    #---------------------------------

    def test_1_1(self):
        '''
        Test results of feedfoward() and evaluate() on 1 input and 1 output.
        '''
        nn, W, bv = make_1_1()
        xv = np.array([0.2])
        av = NeuralNetwork.sigmoid(np.array([0.56]))  # W.dot(xv) + bv
        y = 0

        self.assertTrue(np.array_equal(nn.feedforward(xv), av), 'feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() incorrect')


    def test_3_1(self):
        '''
        Test results of feedfoward() and evaluate() on 3 inputs and 1 output.
        '''
        nn, W, bv = make_3_1()
        xv = np.array([0.2, 0.3, 0.5])
        av = NeuralNetwork.sigmoid(np.array([0.408]))  # W.dot(xv) + bv
        y = 0

        self.assertTrue(np.array_equal(nn.feedforward(xv), av), 'feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() incorrect')


    def test_1_3(self):
        '''
        Test results of feedfoward() and evaluate() on 1 input and 3 outputs.
        '''
        nn, W, bv = make_1_3()
        xv = np.array([0.2])
        av = NeuralNetwork.sigmoid(np.array([0.17, 0.23, 0.31]))  # W.dot(xv) + bv
        y = 2

        self.assertTrue(np.array_equal(nn.feedforward(xv), av), 'feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() incorrect')


    def test_3_4(self):
        '''
        Test results of feedfoward() and evaluate() on 3 inputs and 4 outputs.
        '''
        nn, W, bv = make_3_4()
        xv = np.array([0.2, 0.3, 0.5])
        av = NeuralNetwork.sigmoid(np.array([0.768, 0.796, 0.946, 1.116]))  # W.dot(xv) + bv
        y = 3

        self.assertTrue(np.array_equal(nn.feedforward(xv), av), 'feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() incorrect')

    #---------------------------------
    # 1 hidden layer
    #---------------------------------


    def test_1_1_1(self):
        '''
        Test results of feedforward() and evaluate() on 1 input, layer of 1,
        and 1 output.
        '''
        nn, weights, biases = make_1_1_1()
        xv = np.array([0.2])
        av = NeuralNetwork.sigmoid(np.array([0.06909358]))
        y = 0

        np.testing.assert_allclose(nn.feedforward(xv), av, rtol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() incorrect')


    def test_1_3_1(self):
        '''
        Test results of feedforward() and evaluate() on 1 input, layer of 3,
        and 1 output.
        '''
        nn, weights, biases = make_1_3_1()
        xv = np.array([0.2])
        av = NeuralNetwork.sigmoid(np.array([0.0955163]))
        y = 0

        np.testing.assert_allclose(nn.feedforward(xv), av, rtol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() incorrect')


    def test_2_3_4(self):
        '''
        Test results of feedforward() and evaluate() on 2 inputs, layer of 3,
        and 4 outputs.
        '''
        nn, weights, biases = make_2_3_4()
        xv = np.array([0.1, 0.2])
        av = NeuralNetwork.sigmoid(np.array([0.09837754, 0.09624759, 0.08086678, 0.10866837]))
        y = 3

        np.testing.assert_allclose(nn.feedforward(xv), av, rtol=1e-8, err_msg='feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv), y, 'evaluate() incorrect')



class TestTrain(unittest.TestCase):
    '''
    Tests for train() (i.e. the backpropagation algorithm)
    '''

    #-------------------------------------
    # Training on a network with only one
    # output (should yield no change)
    #-------------------------------------

    def test_1_1_train_1(self):
        '''
        Tests that the [1, 1] network yields the correct activation and output
        after 1 training example -- namely, no change, since the output will
        always be correct.
        '''
        nn, W, bv = make_1_1()
        xv = np.array([0.2])
        y = 0
        example = (xv, y)

        av_new = NeuralNetwork.sigmoid(np.array([0.56]))  # result after training on xv
        y_new = 0

        nn.train([example])
        self.assertTrue(np.array_equal(nn.feedforward(xv)), av_new, 'feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv), y_new, 'evaluate() incorrect')


    def test_3_1_train_1(self):
        '''
        Tests that the [3, 1] network yields the correct activation and output
        after 1 training example -- namely, no change, since the output will
        always be correct.
        '''
        nn, W, bv = make_3_1()
        xv = np.array([0.2, 0.3, 0.5])
        y = 0
        example = (xv, y)

        av_new = NeuralNetwork.sigmoid(np.array([0.408]))  # result after training on xv
        y_new = 0

        nn.train([example])
        self.assertTrue(np.array_equal(nn.feedforward(xv)), av_new, 'feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv), y_new, 'evaluate() incorrect')


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
        example = (xv, y)

        av_new = np.array([0.54971773, 0.54841129, 0.56792473])  # result after training on xv
        y_new = 2

        nn.train([example])
        self.assertTrue(np.array_equal(nn.feedforward(xv)), av_new, 'feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv), y_new, 'evaluate() incorrect')


    def test_3_4_train_1(self):
        '''
        Tests that the [3, 4] network yields the correct activation and output
        after 1 training example.
        '''
        nn, W, bv = make_3_4()
        xv = np.array([0.2, 0.3, 0.5])
        y = 0
        example = (xv, y)

        av_new = np.array([0.68818950, 0.67810281, 0.71011341, 0.74415907])  # result after training on xv
        y_new = 3

        nn.train([example])
        self.assertTrue(np.array_equal(nn.feedforward(xv)), av_new, 'feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv), y_new, 'evaluate() incorrect')


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

        nn.train(examples)
        self.assertTrue(np.array_equal(nn.feedforward(xv1)), av_new, 'feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv1), y_new, 'evaluate() incorrect')


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

        nn.train(examples)
        self.assertTrue(np.array_equal(nn.feedforward(xv1)), av_new, 'feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv1), y_new, 'evaluate() incorrect')


    #-------------------------------------
    # 1 hidden layer, 1 training example
    #-------------------------------------

    def test_2_3_2_train_1(self):
        '''
        Tests that the [2, 3, 2] network yields the correct activation and
        output after 1 training example.
        '''
        nn, W, bv = make_2_3_2()
        xv = np.array([0.1, 0.2])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.53349089, 0.50197267])  # result of xv after training
        y_new = 0

        nn.train(examples)
        self.assertTrue(np.array_equal(nn.feedforward(xv)), av_new, 'feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv), y_new, 'evaluate() incorrect')


    def test_2_3_4_train_1(self):
        '''
        Tests that the [2, 3, 2] network yields the correct activation and
        output after 1 training example.
        '''
        nn, W, bv = make_2_3_4()
        xv = np.array([0.1, 0.2])
        y = 0
        examples = [(xv, y)]

        av_new = np.array([0.53924222, 0.50779901, 0.50407205, 0.51082315])  # result of xv after training
        y_new = 0

        nn.train(examples)
        self.assertTrue(np.array_equal(nn.feedforward(xv)), av_new, 'feedforward() incorrect')
        self.assertEqual(nn.evaluate(xv), y_new, 'evaluate() incorrect')



    #--------------------------------------
    # 1 hidden layer, 2 training examples
    #--------------------------------------

    # TODO: Write tests for 1 hidden layer, 2 training examples



if __name__ == '__main__':
    unittest.main()
