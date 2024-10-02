import numpy as np
from layer import Layer

# Define a Reshape layer class that inherits from Layer
class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        # Store the input and output shapes
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        # Reshape the input to the desired output shape
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        # Reshape the output gradient back to the input shape
        return np.reshape(output_gradient, self.input_shape)