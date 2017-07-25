# Overview of Project
([Back to contents](README.md))

This is a small library implementing a simple neural network. It is originally based on the [book written by Michael Nielson](http://neuralnetworksanddeeplearning.com/index.html), though I am extending it beyond the book with more features.

The goal of this project is not to provide a feature-rich, super efficient implementation of a neural network. Rather, it is to provide reference material for neural networks: the algorithms, their derivations, and a corresponding working implementation in code. In accordance with this goal, the code aims for clarity in presenting the algorithms rather than maximum efficiency. Namely, I leave out many optimizations for the sake of making the underlying algorithms clearer.

## Branches and versions

There are multiple versions of the neural network, with each version adding more features and complexity. Each version is on its own branch:

- **v1-basic**: A basic implementation of the neural network and its learning algorithm. Most notably, training consists of only one round of backpropagation (rather than many rounds, as would be necessary to fully train the network), for the sake of demonstrating the core algorithm.
- **v2-train-until-settled**: Enhances training to incorporate multiple rounds of backpropagation, until the network "settles" (i.e. effectively stops learning). This was added in a separate version to avoid cluttering the core training algorithm with the logic that determines when to stop training. With this addition, this version is a fully functional neural network.
- **v3-parametrized-error**: Improves the results of training by introducing a different error heuristic: cross entropy. To compare the results with the original heuristic, Euclidean distance, this version allows the client to choose which heuristic they want to use when training.

## Features in this version
This version implements a neural network with:

- Variable number of inputs
- Variable number of output classes
- Variable number of hidden layers
- Fully connected layers
- Neuron activation function: sigmoid
- Choice of two error (i.e. "loss") functions to base learning on: Euclidean distance and cross entropy
- User-defined learning rate
- Training until the network settles at an optimum

You can train the network with training data, test it with testing data to get an accuracy rate, and then use it to classify new data.
