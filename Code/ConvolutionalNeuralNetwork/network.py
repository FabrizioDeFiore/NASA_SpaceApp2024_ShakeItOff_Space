def predict(network, input):
    # Initialize the output with the input data
    output = input
    # Pass the input through each layer of the network
    for layer in network:
        output = layer.forward(output)
    # Return the final output after passing through all layers
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
    # Loop over the number of epochs
    for e in range(epochs):
        # Initialize the error for this epoch
        error = 0
        # Loop over each training sample and its corresponding label
        for x, y in zip(x_train, y_train):
            # Forward pass: compute the network's output
            output = predict(network, x)

            # Compute the loss (error) for this sample
            error += loss(y, output)

            # Backward pass: compute the gradient of the loss with respect to the output
            grad = loss_prime(y, output)
            # Update the network's parameters by propagating the gradient backward through the network
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        # Compute the average error over all training samples
        error /= len(x_train)
        # If verbose is True, print the current epoch and the average error
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")