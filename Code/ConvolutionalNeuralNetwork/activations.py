"""

This code defines three classes for different activation functions used in neural networks:

Tanh Class:
Defines the hyperbolic tangent (tanh) activation function and its derivative.
Initializes the Activation class with the tanh function and its derivative.

Sigmoid Class:
Defines the sigmoid activation function and its derivative.
Initializes the Activation class with the sigmoid function and its derivative.

Softmax Class:
Defines the forward pass for the softmax function, which computes the exponentials of the input and normalizes them.
Defines the backward pass for the softmax function, which computes the gradient of the loss with respect to the input. The commented-out original formula is an alternative, less efficient way to compute this gradient.

"""

import numpy as np
from layer import Layer
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        # Define the tanh function
        def tanh(x):
            return np.tanh(x)

        # Define the derivative of the tanh function
        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        # Initialize the Activation class with the tanh function and its derivative
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        # Define the sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Define the derivative of the sigmoid function
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        # Initialize the Activation class with sigmoid and its derivative
        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def forward(self, input):
        # Compute the exponentials of the input
        tmp = np.exp(input)
        # Normalize the exponentials to get the softmax output
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        # Compute the gradient of the loss with respect to the input
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)
