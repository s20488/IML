import os
import tensorflow.keras as keras
from keras_tuner import RandomSearch

# Disable TF warning messages and set backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"

MODEL_FILENAME = "model.keras"
NO_OF_CLASSES = 10
VAL_SPLIT = 0.2
EPOCHS = 1
HIDDEN_UNITS = 1
BATCH_SIZE = 1


class FullyConnectedForMnist:
    '''Simple NN for MNIST database. INPUT => FC/RELU => FC/SOFTMAX'''
    def build_model(hp):
        # Initialize the model
        model = keras.models.Sequential()
        # Flatten the input data of (x, y, 1) dimension
        model.add(keras.layers.Input(shape=(28,28,1)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=hp.Int('dense_units', min_value=32, max_value=512, step=32),
                                     activation=hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'])))
        # Softmax classifier (10 classes)
        model.add(keras.layers.Dense(NO_OF_CLASSES, activation="softmax"))
        model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop']),
                      loss=hp.Choice('loss', values=['categorical_crossentropy', 'binary_crossentropy']),
                      metrics=['accuracy'])
        return model


if __name__ == "__main__":
    # Load dataset as train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Convert from uint8 to float32 and normalize to [0,1]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Transform labels to 'one-hot' encoding, e.g.
    # 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # 6 -> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    y_train = keras.utils.to_categorical(y_train, NO_OF_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NO_OF_CLASSES)

    tuner = RandomSearch(
        FullyConnectedForMnist.build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='output',
        project_name='MNIST')

    tuner.search_space_summary()

    # Train the model
    tuner.search(x_train, y_train, epochs=EPOCHS, validation_split=VAL_SPLIT,
                 batch_size=BATCH_SIZE)

    best_model = tuner.get_best_models(num_models=1)[0]

    best_model.summary()

    # Save model to a file
    best_model.save(MODEL_FILENAME)

    # Evaluate the model on the test data
    best_model.evaluate(x_test, y_test)
