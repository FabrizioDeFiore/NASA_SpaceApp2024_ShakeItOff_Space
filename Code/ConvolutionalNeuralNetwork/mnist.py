import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict

def preprocess_data(x, y, limit):
    # Reshape input data to (number of samples, 28*28, 1)
    x = x.reshape(x.shape[0], 28 * 28, 1)
    # Normalize input data to the range [0, 1]
    x = x.astype("float32") / 255
    # Encode output labels to one-hot vectors of size 10
    # e.g., number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    # Reshape output labels to (number of samples, 10, 1)
    y = y.reshape(y.shape[0], 10, 1)
    # Return the first 'limit' samples of input and output data
    return x[:limit], y[:limit]

# Load MNIST dataset from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Preprocess training data, limit to 1000 samples
x_train, y_train = preprocess_data(x_train, y_train, 1000)
# Preprocess test data, limit to 20 samples
x_test, y_test = preprocess_data(x_test, y_test, 20)

# Define the neural network architecture
network = [
    # Dense (fully connected) layer with input size 28*28 and output size 40
    Dense(28 * 28, 40),
    # Tanh activation function
    Tanh(),
    # Dense (fully connected) layer with input size 40 and output size 10
    Dense(40, 10),
    # Tanh activation function
    Tanh()
]

# Train the neural network
train(
    network,            # The neural network architecture
    mse,                # Loss function (Mean Squared Error)
    mse_prime,          # Derivative of the loss function
    x_train,            # Training data
    y_train,            # Training labels
    epochs=100,         # Number of epochs
    learning_rate=0.1   # Learning rate
)

# Test the neural network
for x, y in zip(x_test, y_test):
    # Predict the output for each test sample
    output = predict(network, x)
    # Print the predicted and true labels
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))