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

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_configs.json")

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
    """List available models in the models directory."""
    try:
        import os
        global MODEL_CONFIGS  # Add this line
        
        # Use correct default path if none provided
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        
        if not os.path.exists(models_dir):
            return {"success": False, "models": [], "error": f"Models directory does not exist: {models_dir}"}
            
        content = os.listdir(models_dir)
        
        models = []
        for filename in content:
            if filename.endswith(('.h5', '.pt', '.onnx')):
                model_id = os.path.splitext(filename)[0]
                
                # Auto-register any model found in directory but not in MODEL_CONFIGS
                if model_id not in MODEL_CONFIGS:
                    model_type = get_model_type(filename)
                    MODEL_CONFIGS[model_id] = {
                        'file': filename,
                        'type': model_type,
                        'source': 'uploaded'
                    }
                
                models.append({
                    "id": model_id,
                    "name": model_id.replace('_', ' ').title(),
                    "type": os.path.splitext(filename)[1][1:].upper(),
                    "source": MODEL_CONFIGS[model_id].get('source', 'uploaded'),
                    "file": filename
                })
        
        # If we auto-registered any models, save the configurations
        save_model_configs()
        
        return {"success": True, "models": models}
    except Exception as e:
        return {"success": False, "models": [], "error": str(e)}

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
        
        # Save changes to config file
        save_model_configs()
        
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

def delete_model(model_id, models_dir=None):
    """Delete a model by ID"""
    try:
        if model_id not in MODEL_CONFIGS:
            return {"success": False, "error": f"Model not found: {model_id}"}
        
        # Prevent deletion of default models
        if MODEL_CONFIGS[model_id]["source"] == "default":
            return {"success": False, "error": "Cannot delete default models"}
        
        # Get file path
        model_file = MODEL_CONFIGS[model_id]["file"]
        file_path = os.path.join(models_dir, model_file) if models_dir else model_file
        
        # Delete the file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove from model config
        del MODEL_CONFIGS[model_id]
        
        # Save changes to config file
        save_model_configs()
        
        return {
            "success": True,
            "message": f"Model {model_id} deleted successfully"
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

def save_model_configs():
    """Save model configurations to a file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(MODEL_CONFIGS, f, indent=2)
    except Exception as e:
        print(f"Error saving model configs: {e}")

def load_model_configs():
    """Load model configurations from file"""
    global MODEL_CONFIGS
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                saved_configs = json.load(f)
                # Update with saved configs while preserving defaults
                for model_id, config in saved_configs.items():
                    if model_id not in MODEL_CONFIGS:
                        MODEL_CONFIGS[model_id] = config
    except Exception as e:
        print(f"Error loading model configs: {e}")

# Load saved configurations at startup
load_model_configs()

if __name__ == "__main__":
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(description='Process images and manage models')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--set-model', help='Set active model')
    parser.add_argument('--register-model', help='Path to model file to register')
    parser.add_argument('--model-name', help='Name for registered model')
    parser.add_argument('--delete-model', help='Delete a model by ID')
    
    args = parser.parse_args()
    
    # Get models directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # Handle list-models command
    if args.list_models:
        # IMPORTANT: Don't print anything except the JSON output
        result = list_models(models_dir)
        # Only output the JSON result, no debug prints
        print(json.dumps(result))
        sys.exit(0)
        
    # Handle set-model command
    elif args.set_model:
        result = set_model(args.set_model, models_dir)
        print(json.dumps(result))
        sys.exit(0)
        
    # Handle register-model command
    elif args.register_model and args.model_name:
        # Use the existing register_model function
        result = register_model(args.register_model, args.model_name, models_dir)
        print(json.dumps(result))
        sys.exit(0)
        
    # Handle delete-model command
    elif args.delete_model:
        result = delete_model(args.delete_model, models_dir)
        print(json.dumps(result))
        sys.exit(0)