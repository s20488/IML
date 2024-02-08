# Predictions, using previously trained neural network
import pickle

import joblib
from tensorflow.keras.models import load_model
import numpy

# Load test dataset
dataset = numpy.loadtxt("test_data.csv", delimiter=",")

# Input vector size
NIN = 2

# Split into input (X) and output (Y) variables
X = dataset[:, 0:NIN]
Y = dataset[:, NIN:]

# Loading normalisation parameters
with open('normalization_parameters.pkl', 'rb') as f:
    scaler_X, scaler_Y = pickle.load(f)

# Normalization parameters
X = scaler_X.transform(X)
Y = scaler_Y.transform(Y)

model = load_model("summation.keras")

predictions = model.predict(X)

# Reversing normalisation
predictions = scaler_Y.inverse_transform(predictions)

print("\nTest results:")
# Caution: the following loop makes sense only for summation of 2 numbers
for i in range(len(Y)):
    print(f"{X[i][0]:.1f} + {X[i][1]:.1f} = {predictions[i][0]:.4f} (expected: {Y[i][0]})")