import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

# Constants
IMG_SIZE = 224
DATASET_PATH = 'data/'
NUM_CLASSES = 2

def load_dataset(dataset_path):
    """
    Loads images and labels from the dataset directory.
    Expects each class to be in its own subdirectory.
    """
    images, labels = [], []
    
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir) or class_name == '.DS_Store':
            continue
        
        for filename in os.listdir(class_dir):
            if filename == '.DS_Store':
                continue
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(class_name)
    
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    """
    Normalizes images and converts labels into one-hot encoded format.
    Then shuffles and splits the data into training and test sets.
    """
    # Normalize images to [0, 1]
    images = images.astype('float32') / 255.0
    
    # Encode labels
    y_encoded = LabelEncoder().fit_transform(labels)
    ct = ColumnTransformer([('onehot', OneHotEncoder(), [0])], remainder='passthrough')
    y_onehot = ct.fit_transform(y_encoded.reshape(-1, 1))
    
    # Shuffle and split the dataset
    return train_test_split(images, y_onehot, test_size=0.10, random_state=42)

def build_model(input_shape, num_classes):
    """
    Builds and compiles an EfficientNetB0 model.
    """
    inputs = layers.Input(shape=input_shape)
    outputs = EfficientNetB0(include_top=True, weights=None, classes=num_classes)(inputs)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Load and preprocess the dataset
    images, labels = load_dataset(DATASET_PATH)
    x_train, x_test, y_train, y_test = preprocess_data(images, labels)
    
    # Build the model
    model = build_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)
    
    # Removed model.summary() so that only the training progress is printed
    # model.summary()
    
    # Train the model; the training process will print epoch progress
    history = model.fit(x_train, y_train, epochs=20, batch_size=16,
                        validation_data=(x_test, y_test))
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test accuracy:", test_acc)
    
    # Save the model
    model.save('model_two.h5')

if __name__ == '__main__':
    main()
