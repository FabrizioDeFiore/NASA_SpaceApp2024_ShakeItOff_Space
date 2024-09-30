import obspy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten

# Load data
data = pd.read_csv("tr2.csv")

# Extract features 
X = data[["time_abs(%Y-%m-%dT%H:%M:%S.%f)", "time_rel(sec)", "velocity(m/s)"]]  # Try Catalog file for time_abs 
y = data["time_abs(%Y-%m-%dT%H:%M:%S.%f)"]  

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Stream object from your data 
st = obspy.Stream()
tr = obspy.Trace(data=data['acceleration_x'].values)  # Assuming acceleration_x is your time series
tr.stats.sampling_rate = 100  # Adjust sampling rate if necessary
st.append(tr)

# Calculate STA/LTA using obspy
st.taper(type='cosine', duration=0.05, max_percentage=0.1)
st.detrend(type='demean')
st.filter(type='bandpass', freqmin=0.5, freqmax=10)
st.rt(st_window=10, lt_window=60)

# Extract STA/LTA values
st_lta_values = st[0].data

# Add STA/LTA values to your DataFrame
data['st_lta'] = st_lta_values

# Convert data to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Reshape data for 1D CNN 
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create CNN model
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))  # Use linear activation for regression

# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test mean absolute error (MAE): {mae}")

# Make predictions
predictions = model.predict(X_test)