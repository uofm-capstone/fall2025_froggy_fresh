import os
import numpy as np
import shutil
import zipfile
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import datetime
import logging
from pathlib import Path

import process_images

app = Flask(__name__)
CORS(app)  # <-- enable CORS for all routes

# Define the backend base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, "models")  # Changed from "model" to "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Track current model ID (default to frog_detector)
current_model_id = 'frog_detector'

# Directories for classification (output folders) within the backend directory
FROGS_FOLDER = os.path.join(BASE_DIR, "frogs")
FILTERED_FOLDER = os.path.join(BASE_DIR, "filtered")

# Create destination folders if they don't exist
os.makedirs(FROGS_FOLDER, exist_ok=True)
os.makedirs(FILTERED_FOLDER, exist_ok=True)

# Move frog_detector.h5 to models directory if it's not already there
default_model_path = os.path.join(BASE_DIR, "frog_detector.h5")
default_model_dest = os.path.join(MODELS_DIR, "frog_detector.h5")
if os.path.exists(default_model_path) and not os.path.exists(default_model_dest):
    shutil.copy2(default_model_path, default_model_dest)

# Initialize default model
try:
    model = load_model(default_model_dest if os.path.exists(default_model_dest) else default_model_path)
except Exception as e:
    logging.error(f"Failed to load default model: {e}")
    model = None

@app.route('/upload', methods=['POST'])
def upload_and_process():
    data = request.json
    folder_path = data.get('folderPath')
    if folder_path:
        # Pass the current model ID to process_images
        results = process_images.process_images(folder_path, model_id=current_model_id)
        return jsonify({
            "message": "FolderPath received", 
            "folderPath": folder_path,
            "modelUsed": current_model_id
        }), 200
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

# Model Management Routes

@app.route('/models', methods=['GET'])
def list_available_models():
    """List all available models"""
    print("GET /models endpoint called")
    result = process_images.list_models(MODELS_DIR)
    print(f"Models API response: {result}")
    return jsonify(result)

@app.route('/models/switch', methods=['POST'])
def switch_model():
    data = request.json
    model_id = data.get('modelId')
    if not model_id:
        return jsonify({"success": False, "error": "Model ID not provided"}), 400
    
    try:
        # Pass the models directory to the function
        result = process_images.set_model(model_id, models_dir=MODELS_DIR)
        global current_model_id, model
        if result.get('success'):
            current_model_id = model_id
            # Update the global model
            model_path = os.path.join(MODELS_DIR, process_images.MODEL_CONFIGS[model_id]['file'])
            model = load_model(model_path)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/models/upload', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({"success": False, "error": "No model file provided"}), 400
    
    model_file = request.files['model']
    model_name = request.form.get('modelName')
    
    if model_file.filename == '':
        return jsonify({"success": False, "error": "Empty filename"}), 400
    
    try:
        # Save uploaded file directly to models directory
        model_path = os.path.join(MODELS_DIR, model_file.filename)
        model_file.save(model_path)
        
        # Register the model
        result = process_images.register_model(model_path, model_name, models_dir=MODELS_DIR)
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)