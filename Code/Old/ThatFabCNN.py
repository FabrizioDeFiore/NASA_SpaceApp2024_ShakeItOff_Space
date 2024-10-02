import os
import numpy as np
from obspy import read
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Directory containing mseed files
apollo12_cat_directory = './NASAResources/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/'

# List all files in the directory and filter out .mseed files
mseed_files = [f for f in os.listdir(apollo12_cat_directory) if f.endswith('.mseed')]

# Initialize an empty list to store data
data_list = []

# Read each mseed file and append the data to the list
for filename in mseed_files:
    st = read(os.path.join(apollo12_cat_directory, filename))
    data_list.extend([tr.data for tr in st])  # Assuming each trace is a separate sample

# Convert the list to a numpy array
data = np.array(data_list)

# Normalize/standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Reshape data if necessary (e.g., for 1D CNN)
data = data.reshape(data.shape[0], data.shape[1], 1)

# Generate dummy labels for demonstration (replace with actual labels)
labels = np.random.randint(0, 2, size=(data.shape[0],))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define a simple CNN model (linear stack of layers)
model = tf.keras.models.Sequential([
    # 1D Convolutional layer with 32 filters, kernel size of 3, ReLU activation, and input shape
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    # Max pooling layer with pool size of 2
    tf.keras.layers.MaxPooling1D(pool_size=2),
    # Flatten layer to convert 2D matrix data to 1D vector
    tf.keras.layers.Flatten(),
    # Dense layer with 64 neurons and ReLU activation
    tf.keras.layers.Dense(64, activation='relu'),
    # Output dense layer with 1 neuron and sigmoid activation for binary classification
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')