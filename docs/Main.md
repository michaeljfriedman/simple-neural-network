# Overview of Project
([Back to contents](README.md))

This is a small library implementing a simple neural network. It is originally based on the [book written by Michael Nielson](http://neuralnetworksanddeeplearning.com/index.html), though I plan to extend it beyond the book with more features that interest me.

## Features
Currently, neural_network.py implements a network with:

- Variable number of inputs
- Variable number of output classes
- Variable number of hidden layers
- Fully connected layers
- Neuron activation function: sigmoid
- Learning algorithm's error ("loss") function: Euclidean distance
- User-defined learning rate

You can train the network with training data, test it with testing data to get an accuracy rate, and then use it to classify new data.
