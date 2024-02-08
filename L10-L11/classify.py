import os
import json

# Disable TF warning messages and set backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_core as keras
import numpy as np
import cv2  # python -m pip install opencv-python
from keras.datasets import mnist
from collections import Counter

# Directory with test set
TEST_DATASET_DIR = 'mnist-test'

# Trained model filename
MODEL = 'model.keras'

if __name__ == "__main__":

    # Load trained model
    model = keras.models.load_model(MODEL)

    (x_test, y_test), (_, _) = mnist.load_data()

    confusion_matrix = np.zeros((10, 10))

    mislabelled_pairs = []

    for image_name in range(x_test.shape[0]):

        # Load the image
        image = x_test[image_name]
        actual = y_test[image_name]

        # Pre-process the image for classification
        image_data = image.astype('float32') / 255
        image_data = keras.preprocessing.image.img_to_array(image_data)
        # Expand dimension (28,28,1) -> (1,28,28,1)
        image_data = np.expand_dims(image_data, axis=0)

        # Classify the input image
        prediction = model.predict(image_data, verbose=0)

        # Find the winner class and the probability
        winner_class = np.argmax(prediction)

        if winner_class != actual:
            mislabelled_pairs.append(f"{actual},{winner_class}")

        confusion_matrix[actual][winner_class] += 1

        winner_probability = np.max(prediction) * 100

        # Build the text label
        label = f"prediction = {winner_class} ({winner_probability:.2f}%), actual = {actual}"

        # Class 2 and 3 probability
        classes_to_check = [2, 3]
        for class_num in classes_to_check:
            class_probability = prediction[0][class_num] * 100
            if class_probability > 1:
                label += f", {class_num} ({class_probability:.2f}%)"

        # Draw the label on the image
        output_image = cv2.resize(image, (500, 500))
        cv2.putText(output_image, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, 255, 2)

        # Show the output image
        cv2.imshow("Output", output_image)

        # Break on 'q' pressed, continue on the other key
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    print("Confusion Matrix:")
    print(confusion_matrix)

    mislabelled_counter = Counter(mislabelled_pairs)
    sorted_mislabelled = dict(sorted(mislabelled_counter.items(), key=lambda item: item[1], reverse=True))

    with open('mnist_mislabellings.json', 'w') as f:
        json.dump(sorted_mislabelled, f)
