"""

This code defines the convolutional layer for the neural network, including methods for the forward and backward passes. 
The forward pass computes the output of the layer by convolving the input with the kernels and adding biases. 
The backward pass computes the gradients of the kernels and input, and updates the kernels and biases using these gradients and the learning rate.

"""

import numpy as np
from scipy import signal
from layer import Layer

# Define a Convolutional layer class that inherits from Layer
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        # Unpack the input shape into depth, height, and width
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        # Calculate the output shape based on the input shape and kernel size
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        # Define the shape of the kernels (filters)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        # Initialize the kernels with random values
        self.kernels = np.random.randn(*self.kernels_shape)
        # Initialize the biases with random values
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        # Store the input for use in the backward pass
        self.input = input
        # Initialize the output with the biases
        self.output = np.copy(self.biases)
        # Perform the convolution operation for each depth slice
        for i in range(self.depth):
            for j in range(self.input_depth):
                # Correlate the input with the kernel and add to the output
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Initialize gradients for kernels and input
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        # Compute the gradients for each depth slice
        for i in range(self.depth):
            for j in range(self.input_depth):
                # Correlate the input with the output gradient to get the kernel gradient
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                # Convolve the output gradient with the kernel to get the input gradient
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        # Update the kernels and biases using the gradients and learning rate
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
