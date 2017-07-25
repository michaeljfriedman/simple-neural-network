# Overview of Project
([Back to contents](README.md))

This is a small library implementing a simple neural network. It is originally based on the [book written by Michael Nielson](http://neuralnetworksanddeeplearning.com/index.html), though I am extending it beyond the book with more features.

The goal of this project is not to provide a feature-rich, super efficient implementation of a neural network. Rather, it is to provide reference material for neural networks: the algorithms, their derivations, and a corresponding working implementation in code. In accordance with this goal, the code aims for clarity in presenting the algorithms rather than maximum efficiency. Namely, I leave out many optimizations for the sake of making the underlying algorithms clearer.

## Features
Currently, neural_network.py implements a network with:

- Variable number of inputs
- Variable number of output classes
- Variable number of hidden layers
- Fully connected layers
- Neuron activation function: sigmoid
- Choice of two error (i.e. "loss") functions to base learning on: Euclidean distance and cross entropy
- User-defined learning rate
- Training until the network settles at an optimum

You can train the network with training data, test it with testing data to get an accuracy rate, and then use it to classify new data.
