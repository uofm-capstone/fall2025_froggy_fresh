import os
import numpy as np
import shutil
import zipfile
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- add this import
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)  # <-- enable CORS for all routes

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
        print("No zip file found.")
        return None

    # Extract the zip file into the backend directory
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(BASE_DIR)
        all_names = zip_ref.namelist()
    print("Extracted zip file contents.")

    # Determine the common prefix (top-level folder) if it exists
    common_prefix = os.path.commonprefix(all_names)
    if (common_prefix and common_prefix.endswith("/")):
        base_folder = os.path.join(BASE_DIR, common_prefix.rstrip("/"))
    else:
        base_folder = BASE_DIR
    print(f"Processing images in folder: {base_folder}")

    # Recursively get image files in base_folder
    image_files = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    # Initialize stats counters
    frog_count = 0
    not_frog_count = 0
    confidence_total = 0.0
    processed_files = []
    last_file = ""

    # Process and classify images
    for img_path in image_files:
        try:
            img = image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
            img_array = image.img_to_array(img) / 255.0

            # Convert to grayscale and replicate to three channels
            img_array = np.mean(img_array, axis=-1, keepdims=True)
            img_array = np.repeat(img_array, 3, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]

            # Determine label and accumulate stats
            if prediction > 0.5:
                label = "Not Frog"
                not_frog_count += 1
                file_conf = prediction
            else:
                label = "Frog"
                frog_count += 1
                file_conf = 1 - prediction

            confidence_total += file_conf
            processed_files.append(os.path.basename(img_path))
            last_file = img_path

            destination_folder = FROGS_FOLDER if label == "Frog" else FILTERED_FOLDER
            shutil.copy(img_path, os.path.join(destination_folder, os.path.basename(img_path)))
            print(f"Moved {img_path} to {label} folder", flush=True)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    total_processed = frog_count + not_frog_count
    average_confidence = round((confidence_total / total_processed) * 100) if total_processed > 0 else 0

    print("\nImage classification complete. Images have been moved to respective folders.", flush=True)

    # Cleanup: Remove zip and extracted folder if applicable
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

    stats = {
        "frogs": frog_count,
        "notFrogs": not_frog_count,
        "confidence": average_confidence,
        "files": processed_files,
        "totalFiles": f"{total_processed}",
        "currentFile": last_file
    }
    return stats

@app.route('/upload', methods=['POST'])
def upload_and_process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    zip_file_path = os.path.join(BASE_DIR, "uploaded_folder.zip")
    file.save(zip_file_path)
    print(f"Received file: {file.filename}")

    stats = process_images()
    if stats:
        return jsonify(stats), 200
    else:
        return jsonify({'error': 'No zip file processed'}), 400

if __name__ == '__main__':
    app.run(debug=False)