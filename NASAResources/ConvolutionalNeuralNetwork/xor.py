import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict


# Define the input data for the XOR problem
# Reshape the input data to (4, 2, 1)
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
# Define the output data for the XOR problem
# Reshape the output data to (4, 1, 1)
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# Define the neural network architecture
network = [
    # Dense (fully connected) layer with input size 2 and output size 3
    Dense(2, 3),
    # Tanh activation function
    Tanh(),
    # Dense (fully connected) layer with input size 3 and output size 1
    Dense(3, 1),
    # Tanh activation function
    Tanh()
]

# Train the neural network
train(
    network,            # The neural network architecture
    mse,                # Loss function (Mean Squared Error)
    mse_prime,          # Derivative of the loss function
    X,                  # Training data
    Y,                  # Training labels
    epochs=10000,       # Number of epochs
    learning_rate=0.1   # Learning rate
)

# Generate a decision boundary plot
points = []
# Generate points in the range [0, 1] for both x and y axes
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        # Predict the output for each point
        z = predict(network, [[x], [y]])
        # Append the point and its prediction to the points list
        points.append([x, y, z[0, 0]])

# Convert the points list to a numpy array
points = np.array(points)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
# Scatter plot of the points with color based on the prediction
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
# Show the plot
plt.show()