import os
import cv2
import numpy as np
import shutil
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model_two.h5')

# Define image size (same as used in training)
IMG_SIZE = 224

# Directory containing your test images
test_dir = 'test'

# Directories to save sorted images
frog_dir = 'test/frog'
not_frog_dir = 'test/not'

# Create directories if they don't exist
os.makedirs(frog_dir, exist_ok=True)
os.makedirs(not_frog_dir, exist_ok=True)

# Function to preprocess and predict images
def classify_image(img_path):
    # Load and preprocess image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        return None
    
    # Resize and normalize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0

    # Expand dimensions to add the batch dimension (1, IMG_SIZE, IMG_SIZE, 3)
    img = np.expand_dims(img, axis=0)

    # Get model prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class

# Iterate through the images in the test directory
for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)

    # Skip directories and non-image files
    if os.path.isdir(img_path) or img_name.startswith('.'):
        continue
    
    # Get the predicted class (0 for not frog, 1 for frog)
    predicted_class = classify_image(img_path)

    if predicted_class is not None:
        if predicted_class == 0:
            # Move the image to the 'not_frog' directory
            shutil.move(img_path, os.path.join(not_frog_dir, img_name))
        else:
            # Move the image to the 'frog' directory
            shutil.move(img_path, os.path.join(frog_dir, img_name))

print("Image sorting complete!")
