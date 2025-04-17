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

@app.route('/getruns', methods=['GET'])
def get_runs():
    runs_folder = os.path.join(os.path.expanduser("~"), "Documents", "Leapfrog", "runs")
    

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