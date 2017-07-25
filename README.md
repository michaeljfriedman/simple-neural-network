# Simple neural network
This is an implementation of a neural network. It is originally based on the the algorithms from [Michael Nielson's book](http://neuralnetworksanddeeplearning.com/index.html), but it also adds some of my own features. Find a full description of the features in the [project overview](docs/Main.md).

There are multiple versions of the neural network, with more features and increasing complexity in each version. You can find each version on its own branch. Also see the project overview for a description of each version.

## Setup and dependencies
### Quick start
To use this module, you must download the source code from this repository, so you will need to install the dependencies for it. But don't worry, this process has been automated with the provided `configure` scripts.

You can set up your environment to either use the module or work on the project by first cloning the repository:

```
git clone https://github.com/michaeljfriedman/simple-neural-network.git && cd simple-neural-network
```

and then running the provided `configure` script for your OS. (However, note that to use this script, you must have the following baseline software installed: Python 2.7, pip package manager, Python virtualenv) Run:

```
./configure --help
```

in a Bash shell (Linux/Mac) or:

```
.\configure --help
```

in PowerShell (Windows) for instructions on its usage. Read on for details on the project's dependencies.

### Dependencies
This project is written in Python 2.7, and it currently requires only Numpy (Python vector/matrix algebra package) for vector/matrix computation in neural_network.py. If you already have this installed or would rather just install it yourself, you don't need the `configure` script. It's only here as a convenience.

## Documentation
For more details, see the [project documentation](docs/README.md).
