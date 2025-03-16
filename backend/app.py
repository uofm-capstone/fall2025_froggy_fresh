import os
import numpy as np
import shutil
import zipfile
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- add this import
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import datetime

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

            if prediction > 0.5:
                label = "NOT FROG"
                not_frog_count += 1
                file_conf = prediction
            else:
                label = "FROG"
                frog_count += 1
                file_conf = 1 - prediction

            confidence_total += file_conf
            # Store file details instead of just name:
            processed_files.append({
                "name": os.path.basename(img_path),
                "classification": label,
                "confidence": round(file_conf * 100)  # store as percentage integer
            })
            last_file = img_path

            destination_folder = FROGS_FOLDER if label == "FROG" else FILTERED_FOLDER
            shutil.copy(img_path, os.path.join(destination_folder, os.path.basename(img_path)))
            print(f"Moved {img_path} to {label} folder", flush=True)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    total_processed = frog_count + not_frog_count
    average_confidence = round((confidence_total / total_processed) * 100) if total_processed > 0 else 0

    runDate = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = {
        "runDate": runDate,
        "frogs": frog_count,
        "notFrogs": not_frog_count,
        "confidence": average_confidence,
        "files": processed_files,
        "totalFiles": f"{total_processed}",
        "currentFile": last_file
    }

    # Save this run’s stats to a file for later retrieval.
    runs_file = os.path.join(BASE_DIR, "runs.json")
    try:
        with open(runs_file, "r") as f:
            runs = json.load(f)
    except Exception:
        runs = []
    runs.append(stats)
    with open(runs_file, "w") as f:
        json.dump(runs, f)

    # Also save last run stats separately if needed:
    with open(os.path.join(BASE_DIR, "last_stats.json"), "w") as f:
        json.dump(stats, f)

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

@app.route('/results', methods=['GET'])
def get_results():
    # Optionally, you can support a query parameter ?date=YYYY-MM-DD to filter runs.
    date_filter = request.args.get("date")
    runs_file = os.path.join(BASE_DIR, "runs.json")
    try:
        with open(runs_file, "r") as f:
            runs = json.load(f)
        if date_filter:
            filtered = [run for run in runs if run['runDate'].startswith(date_filter)]
            return jsonify(filtered), 200
        return jsonify(runs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)