import os
import numpy as np
import shutil
import zipfile
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Define the backend base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model from the backend directory
model_path = os.path.join(BASE_DIR, "frog_detector.h5")
model = load_model(model_path)

# Directories for classification (output folders) within the backend directory
FROGS_FOLDER = os.path.join(BASE_DIR, "frogs")
FILTERED_FOLDER = os.path.join(BASE_DIR, "filtered")

# Create destination folders if they don't exist
os.makedirs(FROGS_FOLDER, exist_ok=True)
os.makedirs(FILTERED_FOLDER, exist_ok=True)

def process_images():
    # Save and process the zip file from within the backend directory
    zip_file_path = os.path.join(BASE_DIR, "uploaded_folder.zip")
    if not os.path.exists(zip_file_path):
        return print("No zip file found.")

    # Extract the zip file into the backend directory
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(BASE_DIR)
        all_names = zip_ref.namelist()
    print("Extracted zip file contents.")

    # Determine the common prefix (top-level folder) if it exists
    common_prefix = os.path.commonprefix(all_names)
    if common_prefix and common_prefix.endswith("/"):
        base_folder = os.path.join(BASE_DIR, common_prefix.rstrip("/"))
    else:
        # If no common prefix is found, we assume the BASE_DIR
        base_folder = BASE_DIR
    print(f"Processing images in folder: {base_folder}")

    # Recursively get image files in base_folder
    image_files = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    # Process and classify images
    for img_path in image_files:
        try:
            # Load image in RGB (3 channels)
            img = image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
            img_array = image.img_to_array(img) / 255.0

            # Convert to grayscale by averaging the channels and reassemble into 3 channels
            img_array = np.mean(img_array, axis=-1, keepdims=True)
            img_array = np.repeat(img_array, 3, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)[0][0]

            # Classify as Frog or Not Frog
            label = "Not Frog" if prediction > 0.5 else "Frog"
            destination_folder = FROGS_FOLDER if label == "Frog" else FILTERED_FOLDER

            # Copy image to the destination folder
            shutil.copy(img_path, os.path.join(destination_folder, os.path.basename(img_path)))
            print(f"Moved {img_path} to {label} folder")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("\nImage classification complete. Images have been moved to the respective folders.")

    # Cleanup: remove the zip file and extracted folder (if it is not BASE_DIR)
    try:
        os.remove(zip_file_path)
        print("Deleted the uploaded zip file.")
    except Exception as e:
        print(f"Error deleting zip file: {e}")

    if base_folder != BASE_DIR and os.path.exists(base_folder):
        try:
            shutil.rmtree(base_folder)
            print(f"Deleted the extracted folder: {base_folder}")
        except Exception as e:
            print(f"Error deleting extracted folder {base_folder}: {e}")

@app.route('/upload', methods=['POST'])
def upload_and_process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded zip file into the backend directory
    zip_file_path = os.path.join(BASE_DIR, "uploaded_folder.zip")
    file.save(zip_file_path)
    print(f"Received file: {file.filename}")

    # Process the uploaded zip file
    process_images()

    return jsonify({'message': 'File uploaded, processed, and cleanup complete'}), 200

if __name__ == '__main__':
    app.run(debug=False)