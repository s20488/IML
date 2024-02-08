# Training of the neural network using Keras/TensorFlow <= 2.14.1
import pickle

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy

# No of epochs for training
EPOCHS = 1000

# Input vector size
NIN = 2

# Output vector size
NOUT = 1

# Load dataset
dataset = numpy.loadtxt("train_data.csv", delimiter=",")

# Split into input (X) and output (Y) variables
# Data format: x1,x2,...,xNIN,y1,y2,...,yNOUT
#              x2,y2,...,zNIN,y1,y2,...,yNOUT

# Create model
# https://keras.io/layers/core/
# https://keras.io/activations/
X = dataset[:, 0:NIN]
Y = dataset[:, NIN:]

# Normalization parameters
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y)

# Save normalization parameters to a file
with open('normalization_parameters.pkl', 'wb') as f:
    pickle.dump([scaler_X, scaler_Y], f)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

model = Sequential()
# Simple perceptron (when NOUT = 1) or single layer (when NOUT > 1)
model.add(Dense(NOUT, input_dim=NIN, activation="tanh"))

# Example multi-layer network (NIN - 3 - 12 - NOUT neurons):
#model.add(Dense(3, input_dim=NIN, activation="tanh"))
#model.add(Dense(12, input_dim=NIN, activation="tanh"))
#model.add(Dense(NOUT, activativon="tanh"))

# Compile model
# Loss functions: https://keras.io/losses/
# Optimizers: https://keras.io/optimizers/
model.compile(loss="mean_squared_error", optimizer="sgd")

# Train the model
model.fit(X, Y, epochs=EPOCHS, batch_size=1)

# Save model to a file
model.save("summation.keras")