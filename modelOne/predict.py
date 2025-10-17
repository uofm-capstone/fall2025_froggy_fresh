import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("frog_detector.h5")

# Directories for classification
IMAGE_FOLDER = "test"
TEST_FROG_FOLDER = "test_frog"
TEST_NOT_FROG_FOLDER = "test_not_frog"

# Create folders if they don't exist
os.makedirs(TEST_FROG_FOLDER, exist_ok=True)
os.makedirs(TEST_NOT_FROG_FOLDER, exist_ok=True)

# Get image files
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process and classify images
for img_name in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_name)

    # Load image in RGB (3 channels)
    img = image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
    img_array = image.img_to_array(img) / 255.0

    # Convert to grayscale by averaging the channels 
    img_array = np.mean(img_array, axis=-1, keepdims=True)  
    img_array = np.repeat(img_array, 3, axis=-1)  
    img_array = np.expand_dims(img_array, axis=0) 

    # Make prediction
    prediction = model.predict(img_array)[0][0]

    # Classify as Frog or Not Frog
    label = "Not Frog" if prediction > 0.5 else "Frog"
    destination_folder = TEST_FROG_FOLDER if label == "Frog" else TEST_NOT_FROG_FOLDER

    # Move image
    shutil.copy(img_path, os.path.join(destination_folder, img_name))
    print(f"Moved {img_name} to {label} folder")

print("\nImage classification complete. Images have been moved to the respective folders.")
