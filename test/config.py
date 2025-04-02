import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# API credentials and settings
API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "frog-ukiu5/1")
WORKSPACE_ID = os.getenv("ROBOFLOW_WORKSPACE_ID")
PROJECT_ID = os.getenv("ROBOFLOW_PROJECT_ID", "frog-ukiu5")

# Directory structure - simplified
BASE_DIR = "test"
DATA_DIR = os.path.join(BASE_DIR, "data")  # Main data directory

# Image directories
DEFAULT_TEST_DIR = os.path.join(DATA_DIR, "test_images", "frogs")
FROG2_DIR = os.path.join(DATA_DIR, "frog-2")  # Dataset directory for frog-2

# Detection settings
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", 0.25))
DETECTION_IOU = 0.45
DEFAULT_YOLO_MODEL = "yolov8n.pt"

# Flask application settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
FLASK_PORT = int(os.getenv("PORT", 5000))
FLASK_HOST = os.getenv("HOST", "0.0.0.0")

# Roboflow API configuration
ROBOFLOW_API_URL = "https://detect.roboflow.com"
ROBOFLOW_CONFIDENCE = int(os.getenv("ROBOFLOW_CONFIDENCE", 40))
ROBOFLOW_OVERLAP = int(os.getenv("ROBOFLOW_OVERLAP", 30))

# Class mapping
CLASS_NAMES = {0: "frog"}

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create only the necessary directories
directories = [DATA_DIR, DEFAULT_TEST_DIR, FROG2_DIR]
for directory in directories:
    if not os.path.exists(directory):
        logging.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)

# Log directory structure in debug mode
if DEBUG:
    logging.debug("Directory configuration:")
    for name, path in {
        "Data directory": DATA_DIR,
        "Test images": DEFAULT_TEST_DIR,
        "Frog-2 dataset": FROG2_DIR,
    }.items():
        exists = "✓" if os.path.exists(path) else "✗"
        logging.debug(f"  {name}: {path} {exists}")