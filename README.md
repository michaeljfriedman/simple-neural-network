# Neural network
This is an implementation of the neural network algorithm described in [Michael Nielson's book](http://neuralnetworksanddeeplearning.com/index.html). After reading the first two chapters (through the backpropagation algorithm) and working out the theory, I decided to write out an implementation of the algorithm myself.

## Setup and dependencies
### Quick start
You can set up your environment from scratch to work on this project, with all dependencies and other configuration done, by first cloning the repository:
```
git clone [repo url] && cd [repo]
```
and then running the provided `configure` script. (However, note that to use this script, you must have the following baseline software installed: Python 2.7, pip package manager, Python virtualenv) Run:
```
./configure --help
```
on a Bash shell (i.e. Linux/Mac) or:
```
configure.cmd --help
```
on Windows for instructions on its usage. Read on for details on the project's dependencies.

### Dependencies
This project is written in Python 2.7, and uses the following dependencies, frameworks, packages, etc.:

- Numpy: Python vector/matrix algebra package, used for vector/matrix computation
