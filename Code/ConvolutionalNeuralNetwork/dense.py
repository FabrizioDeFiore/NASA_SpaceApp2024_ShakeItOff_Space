"""

This code defines a dense (fully connected) layer for the neural network, including methods for the forward and backward passes

"""

import numpy as np
from layer import Layer

# Define a Dense (fully connected) layer class that inherits from Layer
class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Initialize weights with random values, shape (output_size, input_size)
        self.weights = np.random.randn(output_size, input_size)
        # Initialize biases with random values, shape (output_size, 1)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        # Store the input for use in the backward pass
        self.input = input
        # Compute the output of the layer by performing a matrix multiplication
        # of the weights and input, and adding the bias
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        # Compute the gradient of the weights
        weights_gradient = np.dot(output_gradient, self.input.T)
        # Compute the gradient of the input
        input_gradient = np.dot(self.weights.T, output_gradient)
        # Update the weights using the gradient and learning rate
        self.weights -= learning_rate * weights_gradient
        # Update the biases using the gradient and learning rate
        self.bias -= learning_rate * output_gradient
        # Return the gradient of the input
        return input_gradient