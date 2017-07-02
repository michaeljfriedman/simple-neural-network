# Examples of Usage
([Back to contents](README.md))

This page contains some examples to demonstrate how to use the library in your code. See the [API Reference page](API.md) for a full reference.


## Creating a network
This is relatively simple. This example creates a network with 240 inputs, 2 output classes, and no hidden layers, using the default learning rate:
```python
from neural_network import NeuralNetwork
nn = NeuralNetwork(layers=[240, 2])
```

This one does the same, but adds two hidden layers, each with 10 neurons. It also specifies a learning rate of 0.05.
```python
from neural_network import NerualNetwork
nn = NeuralNetwork(layers=[240, 10, 10, 2], eta=0.05)
```
The following sections will use the variable `nn` to refer to this network.


## Training a network
Once you've created a network, you can train it on your training data. In addition to demonstrating the training itself, we also demonstrate how you might have to rearrange the data into the format that the `NeuralNetwork` reads. So in the following example:

- `training_xs` is a list of vectors (`np.array`s), each of length 240, in accordance with the number of input neurons in the network.
- `training_ys` is the list of output classifications that correspond to the input vectors. Each is either 0 or 1, in accordance with the 2 output classes in the network.

The `train()` method takes training examples as a list of `(x, y)` tuples, where `x` is an input vector and `y` is the corresponding classification. So you must rearrange `training_xs` and `training_ys` into this format. Then you can train the network on this data:
```python
training_data = [(training_xs[i], training_ys[i]) for i in range(0, len(training_xs))]
nn.train(training_data)
```


## Testing a network
Testing has a very similar structure to training. In this example, `testing_data` is a list of `(x, y)` tuples, in the same format as `training_data` from the last section. To test on this data:
```python
accuracy = nn.test(testing_data)
print 'Accuracy rate =', accuracy
```
This prints the rate `accuracy` out of 1, indicating the percentage of correct classifications the network got on `testing_data`.


## Classifying data
You can classify an input as follows. In this example, `x` is a vector (`np.array`) of length 240, in accordance with the number of input neurons in the network.
```python
y = nn.classify(x)
print 'Classification =', y
```
This prints the classification `y` determined by the network, an integer from 0 to *number of output neurons* - 1. In this case, it's a 0 or 1, since there are 2 outputs in this network.


## Full example: create, train, test, and classify
For completeness, here we put everything together from the previous sections, so you can see the full flow. The following snippet:

- Creates the network from the second example under **Creating a network**
- Trains it on the data `training_xs` and `training_ys` from **Training a network**
- Tests it on `testing_data` from **Testing a network**
- Classifies the input `x` from **Classifying data**

```python
# Create the network
from neural_network import NeuralNetwork
nn = NeuralNetwork(layers=[240, 10, 10, 2], eta=0.05)

# Train the network
training_data = [(training_xs[i], training_ys[i]) for i in range(0, len(training_xs))]
nn.train(training_data)

# Test the network
# Note: testing_data would be constructed in a similar manner to training_data
accuracy = nn.test(testing_data)
print 'Accuracy rate =', accuracy

# Classify an input
y = nn.classify(x)
print 'Classification =', y
```

Example stdout:
```
Accuracy rate = 0.997
Classification = 0
```
