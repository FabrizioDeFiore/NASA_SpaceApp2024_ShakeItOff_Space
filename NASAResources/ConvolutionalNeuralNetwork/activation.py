"""

This code defines an activation layer for a neural network, including methods for the forward and backward passes. 
The forward pass applies the activation function to the input. 
The backward pass computes the gradient of the loss with respect to the input by multiplying the output gradient with the derivative of the activation function.

"""

import numpy as np
from layer import Layer

# Define an Activation layer class that inherits from Layer
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        # Store the activation function and its derivative
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        # Store the input for use in the backward pass
        self.input = input
        # Apply the activation function to the input and return the result
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        # Compute the gradient of the loss with respect to the input
        # by multiplying the output gradient with the derivative of the activation function
        return np.multiply(output_gradient, self.activation_prime(self.input))
