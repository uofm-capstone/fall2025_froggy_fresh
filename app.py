import os
import logging
import warnings
from flask import Flask, request, render_template_string
from dotenv import load_dotenv

# Load environment variables from .env file.
load_dotenv()

# Suppress extra logs and warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Import the Roboflow InferenceHTTPClient.
from inference_sdk import InferenceHTTPClient

# Initialize the Roboflow client with API key from environment.
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", 
    api_key=ROBOFLOW_API_KEY  # API key loaded from .env file.
)

app = Flask(__name__)