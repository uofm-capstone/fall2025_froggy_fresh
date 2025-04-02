import os
import io
import logging
import base64
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import h5py
from config import (
    ROBOFLOW_CONFIDENCE, ROBOFLOW_OVERLAP, DATA_DIR, DETECTION_CONFIDENCE,
    MODEL_ID  # Add MODEL_ID to imports
)

# Ensure MODEL_ID is defined with a fallback
try:
    from config import MODEL_ID
except ImportError:
    # Try to get from environment
    MODEL_ID = os.environ.get('MODEL_ID', 'frog-detector/1')  # Default fallback value
    logging.warning(f"MODEL_ID not found in config, using: {MODEL_ID}")

# Initialize variables for YOLO models
yolo_model = None
MODEL_DIR = os.path.join("test", "models")  # Directory for models

def get_available_models():
    """Retrieve available models (YOLO, .h5, and Roboflow API)."""
    models = []

    # Check for YOLO models in the directory
    for file in os.listdir(MODEL_DIR):
        if file.endswith('.pt'):  # YOLO models typically use .pt extension
            models.append({'id': file, 'name': f"YOLO - {file}"})

    # Check for .h5 models in the directory
    for file in os.listdir(MODEL_DIR):
        if file.endswith('.h5'):
            models.append({'id': file, 'name': f"H5 Model - {file}"})

    # Add Roboflow API as a model option
    models.append({'id': 'roboflow', 'name': 'Roboflow API'})

    return models

def load_model(model_id):
    """Load a model (YOLO, .h5, or Roboflow API)."""
    global yolo_model

    if model_id == 'roboflow':
        return None  # Roboflow API doesn't require a local model

    if model_id.endswith('.h5'):
        model_path = os.path.join(MODEL_DIR, model_id)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"H5 model not found at {model_path}")
        return tf.keras.models.load_model(model_path)

    # Handle YOLO models
    model_path = os.path.join(MODEL_DIR, model_id) if model_id.endswith('.pt') else "yolov8n.pt"
    if not os.path.exists(model_path):
        logging.warning(f"YOLO model not found at {model_path}, falling back to yolov8n.pt")
        model_path = "yolov8n.pt"

    if yolo_model is None or getattr(yolo_model, 'model_id', None) != model_id:
        logging.info(f"Loading YOLO model from {model_path}")
        yolo_model = YOLO(model_path)
        yolo_model.model_id = model_id

    return yolo_model

def process_image(image_file, client, model_id):
    """Process an image using the selected model (YOLO, .h5, or Roboflow API)."""
    try:
        if model_id == 'roboflow':
            # For Roboflow API
            from config import MODEL_ID
            
            # Log what we're doing
            logging.info(f"Processing with Roboflow API, model: {MODEL_ID}")
            
            # Try with file path
            try:
                result = client.infer(image_file, model_id=MODEL_ID)
                
                # Format Roboflow predictions
                predictions = []
                if 'predictions' in result:
                    for pred in result['predictions']:
                        predictions.append({
                            "x": pred.get('x', 0),
                            "y": pred.get('y', 0),
                            "width": pred.get('width', 0),
                            "height": pred.get('height', 0),
                            "confidence": pred.get('confidence', 0),
                            "class": pred.get('class', 'unknown'),
                            "class_id": 0
                        })
                
                return {'success': True, 'predictions': predictions}
            except Exception as e:
                logging.error(f"Roboflow API error: {str(e)}")
                return {'success': False, 'error': str(e)}
                
        elif model_id.endswith('.h5'):
            # For TensorFlow/Keras models (.h5)
            model = load_model(model_id)
            
            # Process the image for the model
            img = Image.open(image_file)
            img = img.resize((224, 224))  # Typical input size for classification models
            img_array = np.array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Predict without 'conf' parameter
            predictions_raw = model.predict(img_array)
            
            # Format predictions
            predictions = []
            confidence = float(predictions_raw[0][0])
            predictions.append({
                "confidence": confidence,
                "class": "frog" if confidence > 0.5 else "not_frog",
                "x": 0,
                "y": 0,
                "width": 0,
                "height": 0,
                "class_id": 0
            })
            
            return {'success': True, 'predictions': predictions}
        else:
            # For other models, use process_image_yolo
            return process_image_yolo(image_file, model_id)
            
    except Exception as e:
        logging.error(f"Error in process_image: {str(e)}")
        return {'success': False, 'error': str(e)}

def format_yolo_predictions(results, model):
    """Format YOLO predictions into a structured format."""
    predictions = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf)
            class_id = int(box.cls)
            class_name = model.names[class_id]
            predictions.append({
                "x": float(x1),
                "y": float(y1),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "confidence": confidence,
                "class": class_name,
                "class_id": class_id
            })
    return predictions

def run_batch_test(test_dir, client, model_id):
    """Run batch testing on a directory of images."""
    try:
        image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            raise ValueError("No image files found in the test directory.")

        total_images = len(image_files)
        detected_frogs = 0
        total_confidence = 0.0
        missing_detections = []

        # If using Roboflow, ensure we're using the correct version
        if model_id == 'roboflow':
            from config import MODEL_ID
            roboflow_model_id = MODEL_ID
            logging.info(f"Using Roboflow model ID: {roboflow_model_id}")
        else:
            roboflow_model_id = None
            # Pre-load the model to avoid loading it for each image
            model = load_model(model_id)

        for image_file in image_files:
            image_path = os.path.join(test_dir, image_file)
            
            if model_id == 'roboflow':
                # For Roboflow API
                try:
                    result = client.infer(image_path, model_id=roboflow_model_id)
                    
                    # Process predictions
                    predictions = []
                    if 'predictions' in result:
                        for pred in result['predictions']:
                            predictions.append({
                                "confidence": pred.get('confidence', 0),
                                "class": pred.get('class', 'unknown')
                            })
                except Exception as e:
                    logging.error(f"Error processing {image_file} with Roboflow: {str(e)}")
                    predictions = []
            elif model_id.endswith('.h5'):
                # For TensorFlow/Keras models (.h5)
                try:
                    img = Image.open(image_path)
                    img = img.resize((224, 224))  # Typical input size for many classification models
                    img_array = np.array(img) / 255.0  # Normalize
                    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                    
                    # Call the model without 'conf' parameter
                    predictions_raw = model.predict(img_array)
                    
                    # Format predictions for compatibility
                    predictions = []
                    # Assuming the first class is 'frog' for simplicity
                    confidence = float(predictions_raw[0][0])
                    predictions.append({
                        "confidence": confidence,
                        "class": "frog" if confidence > 0.5 else "not_frog"
                    })
                except Exception as e:
                    logging.error(f"Error processing {image_file} with h5 model: {str(e)}")
                    predictions = []
            else:
                # For YOLO models
                try:
                    img = Image.open(image_path)
                    # Use DETECTION_CONFIDENCE from config for YOLO models
                    results = model(img, conf=DETECTION_CONFIDENCE)
                    predictions = format_yolo_predictions(results, model)
                except Exception as e:
                    logging.error(f"Error processing {image_file} with YOLO: {str(e)}")
                    predictions = []

            # Check if any frog was detected
            frog_detected = False
            max_confidence = 0.0
            
            for pred in predictions:
                if pred["class"].lower() == "frog":
                    frog_detected = True
                    max_confidence = max(max_confidence, pred["confidence"])
            
            if frog_detected:
                detected_frogs += 1
                total_confidence += max_confidence
            else:
                missing_detections.append(image_file)

        # Calculate statistics
        accuracy = (detected_frogs / total_images) * 100 if total_images > 0 else 0
        avg_confidence = (total_confidence / detected_frogs) * 100 if detected_frogs > 0 else 0
        
        return {
            "total": total_images,
            "detected": detected_frogs,
            "accuracy": round(accuracy, 2),
            "avg_confidence": round(avg_confidence, 2),
            "missing": missing_detections
        }
                
    except Exception as e:
        logging.error(f"Error in batch test: {str(e)}")
        raise

def get_class_name_from_h5(model_path, class_id):
    """Retrieve class names from the .h5 model's metadata."""
    try:
        with h5py.File(model_path, 'r') as f:
            class_names = f.attrs.get('class_names')
            if class_names:
                return class_names[class_id].decode('utf-8')
    except Exception as e:
        logging.warning(f"Could not retrieve class names from {model_path}: {str(e)}")
    return f"Class {class_id}"

def get_yolo_model(model_id=None):
    """Load a YOLO model by ID."""
    return load_model(model_id)

def process_image_yolo(image_file, model_id=None):
    """Process an image using a YOLO model."""
    try:
        img = Image.open(image_file)
        model = load_model(model_id)
        
        # Skip processing if the model is None (happens with Roboflow API)
        if model is None:
            return {'success': False, 'error': 'Model not found'}
            
        results = model(img, conf=DETECTION_CONFIDENCE)
        buffered = io.BytesIO()
        results_plotted = results[0].plot()
        Image.fromarray(results_plotted).save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        predictions = format_yolo_predictions(results, model)
        
        return {
            'success': True,
            'image_data': image_data,
            'predictions': predictions,
            'model_name': getattr(model, 'model_id', 'yolo')
        }
    except Exception as e:
        logging.error(f"YOLO inference error: {str(e)}")
        return {'success': False, 'error': str(e)}

def get_model_info(model_id=None):
    """Get information about a model."""
    try:
        model = load_model(model_id)
        
        # Handle Roboflow API
        if model_id == 'roboflow':
            return {
                "success": True,
                "model_info": {
                    "model_type": "Roboflow API",
                    "task": "Object Detection",
                    "num_classes": 1,
                    "class_names": {0: "frog"},
                    "input_size": "variable",
                    "model_id": model_id,
                    "description": "Roboflow hosted model for object detection"
                }
            }
        
        # Handle YOLO models
        if model_id and (model_id.endswith('.pt') or model_id == 'yolov8n'):
            info = {
                "model_type": "YOLOv8",
                "task": "Object Detection",
                "num_classes": len(model.names) if hasattr(model, 'names') else 1,
                "class_names": model.names if hasattr(model, 'names') else {0: "frog"},
                "input_size": 640,
                "model_id": model_id,
                "description": "YOLOv8 model for object detection"
            }
            return {"success": True, "model_info": info}
        
        # Handle .h5 models
        if model_id and model_id.endswith('.h5'):
            try:
                import h5py
                MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
                model_path = os.path.join(MODEL_DIR, model_id)
                class_names = {}
                
                try:
                    with h5py.File(model_path, 'r') as f:
                        if 'class_names' in f.attrs:
                            names = f.attrs['class_names']
                            for i, name in enumerate(names):
                                class_names[i] = name.decode('utf-8')
                except Exception:
                    class_names = {0: "frog"}  # Default if no class names found
            except ImportError:
                class_names = {0: "frog"}  # Fallback if h5py not available
                
            info = {
                "model_type": "TensorFlow/Keras",
                "task": "Classification",
                "num_classes": len(class_names) or 1,
                "class_names": class_names,
                "input_size": [224, 224, 3],
                "model_id": model_id,
                "description": "TensorFlow/Keras model for classification"
            }
            return {"success": True, "model_info": info}
        
        # Unknown model type
        return {"success": False, "error": "Unsupported model type"}
        
    except Exception as e:
        logging.error(f"Error retrieving model info: {str(e)}")
        return {"success": False, "error": str(e)}