import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("frog_detector.h5")

# Directory containing images
IMAGE_FOLDER = "data/test"  # Change this to the folder you want to classify
TEST_FROG_FOLDER = "test_frog"
TEST_NOT_FROG_FOLDER = "test_not_frog"

# Create folders if they don't exist
os.makedirs(TEST_FROG_FOLDER, exist_ok=True)
os.makedirs(TEST_NOT_FROG_FOLDER, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Run predictions and move images to the appropriate folders
for img_name in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Adjust classification based on label fix
    label = "Not Frog" if prediction > 0.5 else "Frog"

    # Define destination path based on prediction
    if label == "Frog":
        destination_folder = os.path.join(TEST_FROG_FOLDER, img_name)
    else:
        destination_folder = os.path.join(TEST_NOT_FROG_FOLDER, img_name)

    # Move the image to the appropriate folder
    shutil.copy(img_path, destination_folder)
    print(f"Moved {img_name} to {label} folder")

# Optionally, you can output a summary
print("\nImage classification complete. Images have been moved to the respective folders.")
