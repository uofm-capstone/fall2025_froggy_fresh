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
import logging

import process_images

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

@app.route('/upload', methods=['POST'])
def upload_and_process():
    data = request.json
    folder_path = data.get('folderPath')
    if folder_path:
        # print(f"Received folderPath: {folder_path}")
        results = process_images.process_images(folder_path)
        print(results)
        return jsonify({"message": "FolderPath received", "folderPath": folder_path}), 200
    else:
        logging.warning("folder path not provided??")
        return jsonify({"error": "FolderPath not provided"}), 400

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