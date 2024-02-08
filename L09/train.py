import os
# Disable TF warning messages and set backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_core as keras
import matplotlib.pyplot as plt

MODEL_FILENAME = "model.keras"
NO_OF_CLASSES = 10
VAL_SPLIT = 0.2
EPOCHS = 3
HIDDEN_UNITS = 32
BATCH_SIZE = 256

class FullyConnectedForMnist:
    '''Simple NN for MNIST database. INPUT => FC/RELU => FC/SOFTMAX'''
    def build(hidden_units):
        # Initialize the model
        model = keras.models.Sequential()
        # Flatten the input data of (x, y, 1) dimension
        model.add(keras.layers.Input(shape=(28,28,1)))
        model.add(keras.layers.Flatten())
        # FC/RELU layer
        model.add(keras.layers.Dense(hidden_units, activation='relu'))
        # Softmax classifier (10 classes)
        model.add(keras.layers.Dense(NO_OF_CLASSES, activation="softmax"))
        return model


if __name__ == "__main__":
    
    # Load dataset as train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Convert from uint8 to float32 and normalize to [0,1]
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    # Transform labels to 'one-hot' encoding, e.g.
    # 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # 6 -> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    y_train = keras.utils.to_categorical(y_train, NO_OF_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NO_OF_CLASSES)

    # Construct the model
    model = FullyConnectedForMnist.build(HIDDEN_UNITS)

    # Compile the model and print summary
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    model.summary()

    # Train the model
    train_model = model.fit(x=x_train, y=y_train, epochs=EPOCHS, validation_split=VAL_SPLIT,
              batch_size=BATCH_SIZE)

    # Save model to a file
    model.save(MODEL_FILENAME)

    # Evaluate the model on the test data
    model.evaluate(x_test, y_test)

    # Plotting loss and accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_model.history['loss'], label='Train Loss')
    plt.plot(train_model.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_model.history['accuracy'], label='Train Accuracy')
    plt.plot(train_model.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_accuracy.png')
    plt.show()

# Zadanie 9.1 (1p)
# Uzupełnić plik o rysowanie wykresu funkcji kosztu (loss) oraz dokładności (accuracy) 
# w funkcji epoki, na zbiorach treningowym i walidującym (na jednym wykresie).

# Zadanie 9.2 (1p)
# Metodą prób i błędów (maks. kilkanaście prób) dobrać odpowiednie wartości 
# hiperparametrów: liczbę neuronów w warstwie ukrytej, liczbę epok treningu rozmiar wsadu.
# Warunek: trening nie powinien trwać dłużej niż kilkanaście sekund.
# Kluczowe pytanie: co jest wyznacznikiem jakości treningu?
# Odpowiedż: Accuracy, Loss Function, Learning Curves i czas treningu

# Wynik: Plik z uzupełnionym kodem oraz plik graficzny z przykładowym wykresem.