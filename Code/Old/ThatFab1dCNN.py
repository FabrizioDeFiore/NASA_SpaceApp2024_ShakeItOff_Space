import numpy as np
from obspy import read
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Load mseed data
st = read('data.mseed')
data = np.array([tr.data for tr in st])  # Assuming each trace is a separate sample

# Normalize/standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Reshape data for 1D CNN
data = data.reshape(data.shape[0], data.shape[1], 1)

# Generate dummy labels for demonstration (replace with actual labels)
labels = np.random.randint(0, 2, size=(data.shape[0],))  # Binary classification

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the 1D CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(data.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')