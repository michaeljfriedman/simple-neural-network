# API Reference
([Back to contents](README.md))

## Class: `NeuralNetwork`
### `__init__(layers, weights=None, biases=None)`
Initializes the NeuralNetwork given `layers`, a list of integers indicating the number of neurons in each layer of the network. (The first and last value in this list are thus the number of inputs and outputs of the network, respectively.) Values in this list must all be positive numbers, or an exception will be raised.

For testing purposes, also optionally provide `weights`, a list of initial weight matrices between each pair of layers. Each matrix corresponds to the weights applied to the *second* layer in the pair. Similarly, provide `biases`, the corresponding list of initial bias vectors.

### `train(examples, eta=0.25, nd=1e-5, max_rounds=None)`
Trains the network using the list `examples` of training examples. This is a list of 2-tuples (training input, training label), where "training input" is a vector (`np.array`) with the same length as the number of inputs in the network, and "training label" is the correct classification for that input, indexed from 0.

Optionally provide `eta`, the learning rate.

Also optionally provide `nd` and/or `max_rounds`, the "negligible delta" value and maximum number of rounds, respectively. If only `nd` is specified, then the network trains until it "settles" within this negligible delta: that is, it trains for as many rounds as necessary until the weights and biases change by no more than `nd` in a round. If `max_rounds` is also specified, then the network will train either until it settles or until it passes `max_rounds` rounds. If only `max_rounds` is specified, then the network will just train for that fixed number of rounds.

### `test(examples)`
Tests the network on the list `examples`, which has the same format as the list provided to `train()`. Returns the accuracy rate, a floating point number out of 1.

### `classify(xv)`
Returns the classification chosen by the network for input `xv` (the index of the output with highest probability, indexed from 0).
