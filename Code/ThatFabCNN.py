import numpy as np
from obspy import read
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load mseed data
st = read('data.mseed')
data = np.array([tr.data for tr in st])  # Assuming each trace is a separate sample

# Normalize/standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Reshape data if necessary (e.g., for 1D CNN)
data = data.reshape(data.shape[0], data.shape[1], 1)

# Generate dummy labels for demonstration (replace with actual labels)
labels = np.random.randint(0, 2, size=(data.shape[0],))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Assuming binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))