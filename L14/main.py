from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import os
import cv2
import numpy as np

train_dir = 'cats_dogs-1000/images'
test_dir = 'cats_dogs-1000/images-test'


# Loading of images and labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            images.append(img.flatten())
            labels.append(label)
    return images, labels


# Uploading images of cats and dogs
cats, cat_labels = load_images_from_folder(os.path.join(train_dir, 'class-0-cats'), 0)
dogs, dog_labels = load_images_from_folder(os.path.join(train_dir, 'class-1-dogs'), 1)

# Combination of images and labels
X_train = np.array(cats + dogs)
y_train = np.array(cat_labels + dog_labels)

# Data normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Creation and training of the SVC model
clf = svm.SVC()
clf.fit(X_train, y_train)

# Loading and processing test data
cats_test, cat_labels_test = load_images_from_folder(os.path.join(test_dir), 0)
dogs_test, dog_labels_test = load_images_from_folder(os.path.join(test_dir), 1)

X_test = np.array(cats_test + dogs_test)
y_test = np.array(cat_labels_test + dog_labels_test)

# Normalisation of test data
X_test = scaler.transform(X_test)

# Prediction and evaluation of the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
