import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
import shutil

# Dictionary of model configurations
MODEL_CONFIGS = {
    'frog_detector': {
        'file': 'frog_detector.h5',
        'type': 'keras',
        'source': 'default'
    }
    # Additional models will be added dynamically
}

# Currently active model
current_model_id = 'frog_detector'

def get_model_type(file_path):
    """Determine model type from file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.h5':
        return 'keras'
    elif ext == '.pt':
        return 'pytorch'
    elif ext == '.onnx':
        return 'onnx'
    else:
        return 'unknown'

def load_specific_model(model_id, models_dir=None):
    """Load a model by ID"""
    global current_model_id
    
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_id}' not found in available models")
    
    config = MODEL_CONFIGS[model_id]
    
    # Determine model path
    if models_dir:
        model_path = os.path.join(models_dir, config['file'])
    else:
        model_path = os.path.join(".", "backend", config['file'])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model based on type
    if config['type'] == 'keras':
        model = load_model(model_path)
    else:
        raise ValueError(f"Unsupported model type: {config['type']}")
    
    current_model_id = model_id
    return model

def list_models(models_dir=None):
    """List all available models"""
    available_models = []
    
    for model_id, config in MODEL_CONFIGS.items():
        # Determine if model file exists
        if models_dir:
            model_path = os.path.join(models_dir, config['file'])
        else:
            model_path = os.path.join(".", "backend", config['file'])
        
        available = os.path.exists(model_path)
        
        if available:
            available_models.append({
                'id': model_id,
                'name': model_id.replace('_', ' ').title(),
                'type': config['type'],
                'source': config['source'],
                'file': config['file']
            })
    
    return {
        "success": True,
        "models": available_models,
        "activeModel": current_model_id
    }

def set_model(model_id, models_dir=None):
    """Set the current model"""
    try:
        # Try loading the model to verify it works
        model = load_specific_model(model_id, models_dir)
        
        # If successful, update current model
        global current_model_id
        current_model_id = model_id
        
        return {
            "success": True,
            "model": model_id,
            "message": f"Successfully switched to model: {model_id}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def register_model(model_path, model_name=None, models_dir=None):
    """Register a new model"""
    try:
        # Validate file exists
        if not os.path.exists(model_path):
            return {"success": False, "error": f"File not found: {model_path}"}
        
        # Use filename as model name if not provided
        if model_name is None:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        model_id = model_name.lower().replace(" ", "_")
        
        # Check if model ID already exists
        if model_id in MODEL_CONFIGS:
            i = 1
            while f"{model_id}_{i}" in MODEL_CONFIGS:
                i += 1
            model_id = f"{model_id}_{i}"
        
        # Get model type from extension
        model_type = get_model_type(model_path)
        
        # If models_dir is specified and the file isn't already there, copy it
        if models_dir:
            target_path = os.path.join(models_dir, os.path.basename(model_path))
            if model_path != target_path:
                shutil.copy2(model_path, target_path)
        
        # Add to model configs
        MODEL_CONFIGS[model_id] = {
            'file': os.path.basename(model_path),
            'type': model_type,
            'source': 'uploaded'
        }
        
        return {
            "success": True, 
            "model": {
                "id": model_id,
                "name": model_name,
                "file": os.path.basename(model_path),
                "type": model_type,
                "source": "uploaded"
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def process_images(folder_path, model_id=None, models_dir=None):
    """Process images with specified model"""
    # Use provided model_id or default to current
    model_id_to_use = model_id if model_id else current_model_id
    
    try:
        # Load the specified model
        model = load_specific_model(model_id_to_use, models_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to default model location
        model = load_model(os.path.join(".", "backend", "frog_detector.h5"))
        model_id_to_use = 'frog_detector'

    processed_files = []

    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]

    # Initialize stats counters
    frog_count = 0
    not_frog_count = 0
    confidence_total = 0.0
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
            print(f"{os.path.basename(img_path)} says: {label}; conf: {round(file_conf * 100)}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    total_processed = frog_count + not_frog_count
    average_confidence = round((confidence_total / total_processed) * 100) if total_processed > 0 else 0

    runDate = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = {
        "runDate": runDate,
        "frogs": frog_count,
        "notFrogs": not_frog_count,
        "confidence": average_confidence,
        "files": processed_files,
        "totalFiles": f"{total_processed}",
        "currentFile": last_file,
        "modelUsed": model_id_to_use
    }

    # Save this run's stats to a file for later retrieval.
    runs_file = os.path.join(".", "backend", "runs.json")
    try:
        with open(runs_file, "r") as f:
            runs = json.load(f)
    except Exception:
        runs = []
    runs.append(stats)
    with open(runs_file, "w") as f:
        json.dump(runs, f)

    return stats