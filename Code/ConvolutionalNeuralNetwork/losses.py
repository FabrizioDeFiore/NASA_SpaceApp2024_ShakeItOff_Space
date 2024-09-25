import numpy as np

# Mean Squared Error (MSE) loss function
def mse(y_true, y_pred):
    # Compute the mean of the squared differences between true and predicted values
    return np.mean(np.power(y_true - y_pred, 2))

# Derivative of the Mean Squared Error (MSE) loss function
def mse_prime(y_true, y_pred):
    # Compute the gradient of the MSE loss with respect to the predicted values
    return 2 * (y_pred - y_true) / np.size(y_true)

# Binary Cross-Entropy loss function
def binary_cross_entropy(y_true, y_pred):
    # Compute the mean of the binary cross-entropy loss
    # -y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred)
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

# Derivative of the Binary Cross-Entropy loss function
def binary_cross_entropy_prime(y_true, y_pred):
    # Compute the gradient of the binary cross-entropy loss with respect to the predicted values
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)