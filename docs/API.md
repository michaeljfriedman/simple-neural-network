# API Reference
([Back to contents](README.md))

## Class: `NeuralNetwork`

### `__init__(layers, weights=None, biases=None)`
Initializes the NeuralNetwork given `layers`, a list of integers indicating the number of neurons in each layer of the network. (The first and last value in this list are thus the number of inputs and outputs of the network, respectively.) Values in this list must all be positive numbers, or an exception will be raised.

For testing purposes, also optionally provide `weights`, a list of initial weight matrices between each pair of layers. Each matrix corresponds to the weights applied to the *second* layer in the pair. Similarly, provide `biases`, the corresponding list of initial bias vectors.

### `train(examples, error_function=CROSS_ENTROPY, eta=0.25, nd=1e-5, max_rounds=None, manual_stop=False)`
Trains the network using the list `examples` of training examples. This is a list of 2-tuples (training input, training label), where "training input" is a vector (`np.array`) with the same length as the number of inputs in the network, and "training label" is the correct classification for that input, indexed from 0. Returns the number of rounds the network trained for.

Optionally specify the error (i.e. "loss") heuristic with the `error_function` parameter. Valid values for this are the class variables: EUCLIDEAN_DISTANCE (for Euclidean distance heuristic), or CROSS_ENTROPY (for cross entropy heuristic). Each round, the total error over all examples and the accuracy rate on all examples will be printed to show progress throughout the training.

Also optionally provide `eta`, the learning rate.

Additionally, optionally provide `nd` and/or `max_rounds`, the "negligible delta" value and maximum number of rounds, respectively. If only `nd` is specified, then the network trains until it "settles" within this negligible delta: that is, it trains for as many rounds as necessary until the weights and biases change by no more than `nd` in a round. If `max_rounds` is also specified, then the network will train either until it settles or until it passes `max_rounds` rounds. If only `max_rounds` is specified, then the network will just train for that fixed number of rounds. If neither is set, you must set `manual_stop` to `True` (see next paragraph) or an exception will be raised.

Lastly, optionally set `manual_stop` to `True` to have training pause briefly every few rounds after stats are printed. While paused, you will have the opportunity to manually stop training. You can also use this in conjunction with `nd` and/or `max_rounds` to give training an automatic stopping point as well. If you do not set `nd` or `max_rounds`, you must set `manual_stop` to `True` or an exception will be raised (otherwise there would be no stopping point!).

### `plot_error(start=1, end=None)`
During training, the sum of error over all examples is recorded at each round. This function plots these measurements on a graph.

Optionally specify the starting and ending rounds with `start` and `end`, respectively (rounds are indexed from 1). Note the default value for `end` being `None` indicates that it plots up to the last round.

### `plot_accuracy(start=1, end=None)`
Same as `plot_error()`, but it plots the accuracy rate of the network on all examples.

### `test(examples)`
Tests the network on the list `examples`, which has the same format as the list provided to `train()`. Returns the accuracy rate, a floating point number out of 1.

### `classify(xv)`
Returns the classification chosen by the network for input `xv` (the index of the output with highest probability, indexed from 0).

### `save(filename)`
Saves the network to the file `filename` (provide a string name).

### `load(filename)`
Loads a network from the file `filename` (provide a string name). Returns the `NeuralNetwork` object loaded.
