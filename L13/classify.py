import os
# Disable TF warning messages and set backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_core as keras
import numpy as np
import cv2     # python -m pip install opencv-python

# Directory with test set
TEST_DATASET_DIR = 'mnist-test'

# Trained model filename
MODEL = 'model.keras'

if __name__ == "__main__":
    
    # Load trained model
    model = keras.models.load_model(MODEL)

    for image_name in os.listdir(TEST_DATASET_DIR):
        
        # Load the image
        image = cv2.imread(TEST_DATASET_DIR + os.path.sep + image_name, 
                           cv2.IMREAD_GRAYSCALE)
                
        # Pre-process the image for classification
        image_data = image.astype('float32') / 255
        image_data = keras.preprocessing.image.img_to_array(image_data)
        # Expand dimension (28,28,1) -> (1,28,28,1)
        image_data = np.expand_dims(image_data, axis=0)
        
        # Classify the input image
        prediction = model.predict(image_data, verbose=0)
        
        # Find the winner class and the probability
        winner_class = np.argmax(prediction)
        winner_probability = np.max(prediction)*100
        
        # Build the text label
        label = f"prediction = {winner_class} ({winner_probability:.2f}%)"
        
        # Draw the label on the image
        output_image = cv2.resize(image, (500,500))
        cv2.putText(output_image, label, (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)

        # Show the output image        
        cv2.imshow("Output", output_image)
        
        # Break on 'q' pressed, continue on the other key
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
