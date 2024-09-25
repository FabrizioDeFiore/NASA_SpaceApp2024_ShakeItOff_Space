import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict


def preprocess_data(x, y, limit):
    # Find indices of all instances of class 0 and class 1, limit to 'limit' instances per class
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    # Combine indices of both classes
    all_indices = np.hstack((zero_index, one_index))
    # Shuffle the combined indices
    all_indices = np.random.permutation(all_indices)
    # Select and shuffle the data and labels based on the combined indices
    x, y = x[all_indices], y[all_indices]
    # Reshape the data to (number of samples, 1, 28, 28)
    x = x.reshape(len(x), 1, 28, 28)
    # Normalize the data to the range [0, 1]
    x = x.astype("float32") / 255
    # Convert labels to one-hot encoding
    y = np_utils.to_categorical(y)
    # Reshape labels to (number of samples, 2, 1)
    y = y.reshape(len(y), 2, 1)
    return x, y

# Load MNIST dataset from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Preprocess training data
x_train, y_train = preprocess_data(x_train, y_train, 100)
# Preprocess test data
x_test, y_test = preprocess_data(x_test, y_test, 100)

# Define the neural network architecture
network = [
    # Convolutional layer with input shape (1, 28, 28), kernel size 3, and 5 filters
    Convolutional((1, 28, 28), 3, 5),
    # Sigmoid activation function
    Sigmoid(),
    # Reshape layer to flatten the output of the convolutional layer
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    # Dense (fully connected) layer with 100 neurons
    Dense(5 * 26 * 26, 100),
    # Sigmoid activation function
    Sigmoid(),
    # Dense (fully connected) layer with 2 neurons (output layer)
    Dense(100, 2),
    # Sigmoid activation function
    Sigmoid()
]

# Train the neural network
train(
    network,                      # The neural network architecture
    binary_cross_entropy,         # Loss function
    binary_cross_entropy_prime,   # Derivative of the loss function
    x_train,                      # Training data
    y_train,                      # Training labels
    epochs=20,                    # Number of epochs
    learning_rate=0.1             # Learning rate
)

# Test the neural network
for x, y in zip(x_test, y_test):
    # Predict the output for each test sample
    output = predict(network, x)
    # Print the predicted and true labels
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")