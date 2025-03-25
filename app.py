import os
import logging
import warnings
import traceback
import tempfile
import functools
import time
import shutil
import json
import random
import base64
import sys

# Apply these settings BEFORE any TensorFlow imports
# Suppress TensorFlow and CUDA warnings (must be set before importing TF)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU if not needed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL

# Only suppress warnings for specific packages, not everything
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Silence specific noisy loggers but NOT werkzeug
for module in ["tensorflow", "PIL.TiffImagePlugin", "matplotlib"]:
    logging.getLogger(module).setLevel(logging.ERROR)

# Allow Werkzeug to show port access info
logging.getLogger('werkzeug').setLevel(logging.INFO)

# Import libraries
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from flask import Flask, request, render_template_string, redirect, url_for, g, jsonify
from PIL import Image
from dotenv import load_dotenv

# Import our modules
from models import (
    model_manager, process_image, get_base64_image, process_test_image,
    fine_tune_roboflow_model, train_yolo_model, train_tensorflow_model, optimize_model_performance
)
# Import templates
from templates import HTML_TEMPLATE, DASHBOARD_TEMPLATE, TRAINING_TEMPLATE
# Import data organization helpers
from data_organization import create_train_test_split, count_images, get_category_images, validate_image_dir

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Performance optimization: Add simple cache
cache = {}
CACHE_TIMEOUT = 300  # 5 minutes

# Helper: count images in a directory
def count_images(directory, extensions=('.jpg', '.jpeg', '.png')):
    if not os.path.exists(directory):
        return 0
        
    # Performance: Cache directory listings
    cache_key = f"dir_count_{directory}"
    if cache_key in cache and cache[cache_key]['time'] > time.time() - CACHE_TIMEOUT:
        return cache[cache_key]['value']
    
    count = sum(1 for file in os.listdir(directory) if file.lower().endswith(extensions))
    cache[cache_key] = {'value': count, 'time': time.time()}
    return count

# Count images by prefix (needed for new directory structure)
def count_images_by_prefix(directory, prefix, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """Count image files with specific prefix in a directory."""
    if not os.path.exists(directory):
        return 0
    
    return sum(1 for file in os.listdir(directory) 
              if file.lower().endswith(extensions) and file.startswith(prefix))

# Performance: Template caching decorator
def cached_template(timeout=CACHE_TIMEOUT):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            cache_key = f"{f.__name__}_{str(args)}_{str(kwargs)}"
            if cache_key in cache and cache[cache_key]['time'] > time.time() - timeout:
                return cache[cache_key]['value']
            result = f(*args, **kwargs)
            cache[cache_key] = {'value': result, 'time': time.time()}
            return result
        return wrapper
    return decorator

def debug_models():
    models = model_manager.get_available_models()
    logging.info(f"Available models: {models}")
    return models

@app.route("/", methods=["GET", "POST"])
def index():
    # Get available models and ensure they're passed to the template
    available_models = model_manager.get_available_models()
    default_model = model_manager.default_model or (available_models[0] if available_models else "")
    
    logging.info(f"Models for dropdown: {available_models}, default: {default_model}")
    
    # Rest of your code...
    return render_template_string(
        HTML_TEMPLATE, 
        prediction="Upload an image to detect frogs", 
        confidence_text="", 
        output_image="",
        model_options=available_models,  # Make sure this is passed and named correctly
        selected_model=default_model
    )

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    
    # Get selected model
    model_id = request.form.get("model", None)
    try:
        model = model_manager.get_model(model_id)
    except ValueError:
        return "Invalid model selected", 400
    
    # Performance: Use a thread pool for image processing
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            
            # Create a permanent copy for feedback
            saved_dir = "uploads"
            os.makedirs(saved_dir, exist_ok=True)
            saved_path = os.path.join(saved_dir, f"{int(time.time())}_{file.filename}")
            shutil.copy(tmp_path, saved_path)
            
            # Process the image asynchronously
            future = executor.submit(process_image, tmp_path, model)
            annotated_img, prediction_label, confidence_text = future.result(timeout=30)
            
            # Clean up temporary file
            os.remove(tmp_path)
            
            # Convert annotated image to base64
            output_image = get_base64_image(annotated_img)
            
            # Return the result
            available_models = debug_models()
            return render_template_string(
                HTML_TEMPLATE,
                prediction=prediction_label,
                confidence_text=confidence_text,
                output_image=output_image,
                model_options=available_models,  # Change to match the variable name in index route
                selected_model=model_id,
                saved_path=saved_path
            )
        except Exception as e:
            logging.error("Error during prediction: %s", traceback.format_exc())
            return "Error processing image", 500

@app.route("/feedback", methods=["POST"])
def record_feedback():
    """Record user feedback about a prediction"""
    try:
        data = request.json
        if not data or "image_path" not in data or "actual_class" not in data:
            return jsonify({"success": False, "error": "Missing data"}), 400
        
        image_path = data["image_path"]
        actual_class = data["actual_class"]
        
        # Use subfolders in test directory
        test_dir = os.environ.get("TEST_DIR", "frog_images/test")
        target_folder = os.path.join(test_dir, actual_class)
        os.makedirs(target_folder, exist_ok=True)
        
        # Copy the file to the appropriate subfolder
        filename = f"{int(time.time())}_{os.path.basename(image_path)}"
        target_path = os.path.join(target_folder, filename)
        
        # Copy the file
        shutil.copy(image_path, target_path)
        
        # Clear cache to ensure dashboard updates
        keys_to_delete = [k for k in cache.keys() if k.startswith("perf_") or k.startswith("dashboard")]
        for key in keys_to_delete:
            if key in cache:
                del cache[key]
        
        return jsonify({"success": True})
    except Exception as e:
        logging.error(f"Error recording feedback: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/clear-dashboard-cache", methods=["POST"])
def clear_dashboard_cache():
    """Clear dashboard cache to force refresh"""
    keys_to_delete = [k for k in cache.keys() if k.startswith("perf_") or k.startswith("dashboard")]
    for key in keys_to_delete:
        if key in cache:
            del cache[key]
    return jsonify({"success": True})

# Performance: Use ProcessPoolExecutor for compute-intensive tasks
def compute_performance_parallel(model, test_dir):
    """Compute model performance metrics using parallel processing"""
    
    # Performance: Cache results based on model and directories
    cache_key = f"perf_{model.get_model_name()}_{test_dir}"
    if cache_key in cache and cache[cache_key]['time'] > time.time() - CACHE_TIMEOUT:
        return cache[cache_key]['value']
    
    try:
        # Get category subdirectories 
        frog_dir = os.path.join(test_dir, "frog")
        not_frog_dir = os.path.join(test_dir, "not_frog")
        
        # Get images by category
        frog_images = []
        if os.path.exists(frog_dir):
            frog_images = [os.path.join(frog_dir, f) for f in os.listdir(frog_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
        not_frog_images = []
        if os.path.exists(not_frog_dir):
            not_frog_images = [os.path.join(not_frog_dir, f) for f in os.listdir(not_frog_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Prepare arguments for parallel processing
        test_cases = [(img_path, "frog") for img_path in frog_images]
        test_cases.extend([(img_path, "not_frog") for img_path in not_frog_images])
        
        max_workers = min(os.cpu_count() or 4, 8)
        
        # Process images in parallel
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_test_image, img_path, expected, model): (img_path, expected) 
                      for img_path, expected in test_cases}
            
            # Process results as they come in
            for future in as_completed(futures):
                img_path, expected = futures[future]
                try:
                    is_correct, confidence, status = future.result()
                    if status == "ok":
                        results.append({
                            "path": img_path,
                            "expected": expected,
                            "correct": is_correct,
                            "confidence": confidence
                        })
                except Exception as exc:
                    logging.error(f"Error processing {img_path}: {exc}")
        
        # Calculate metrics
        frog_results = [r for r in results if r["expected"] == "frog"]
        not_frog_results = [r for r in results if r["expected"] == "not_frog"]
        
        # Count correct predictions
        correct_frog = sum(1 for r in frog_results if r["correct"])
        correct_not_frog = sum(1 for r in not_frog_results if r["correct"])
        
        # Calculate accuracy
        total_frog = len(frog_results)
        total_not_frog = len(not_frog_results)
        
        frog_accuracy = round((correct_frog / total_frog * 100) if total_frog > 0 else 0, 1)
        not_frog_accuracy = round((correct_not_frog / total_not_frog * 100) if total_not_frog > 0 else 0, 1)
        
        overall_accuracy = round(((correct_frog + correct_not_frog) / (total_frog + total_not_frog) * 100)
                                if (total_frog + total_not_frog) > 0 else 0, 1)
        
        # Calculate confidence
        frog_confidences = [r["confidence"] for r in frog_results if r["correct"]]
        not_frog_confidences = [r["confidence"] for r in not_frog_results if r["correct"]]
        
        avg_confidence = round((sum(frog_confidences + not_frog_confidences) / 
                              len(frog_confidences + not_frog_confidences) * 100)
                              if (frog_confidences + not_frog_confidences) else 0, 1)
        
        # Get recent image paths for display (up to 8 each)
        recent_frog_images = []
        for result in sorted(frog_results, key=lambda x: os.path.getmtime(x["path"]), reverse=True)[:8]:
            img_path = result["path"]
            # Create a data URL for the image
            with open(img_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                recent_frog_images.append({
                    "url": f"data:image/jpeg;base64,{img_data}",
                    "correct": result["correct"]
                })
                
        recent_not_frog_images = []
        for result in sorted(not_frog_results, key=lambda x: os.path.getmtime(x["path"]), reverse=True)[:8]:
            img_path = result["path"]
            # Create a data URL for the image
            with open(img_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                recent_not_frog_images.append({
                    "url": f"data:image/jpeg;base64,{img_data}",
                    "correct": result["correct"]
                })
        
        metrics = {
            "total_frog": total_frog,
            "total_not_frog": total_not_frog,
            "correct_frog": correct_frog,
            "correct_not_frog": correct_not_frog,
            "missed_frog": total_frog - correct_frog,
            "false_positive": total_not_frog - correct_not_frog,
            "frog_accuracy": frog_accuracy,
            "not_frog_accuracy": not_frog_accuracy,
            "accuracy": overall_accuracy,
            "avg_confidence": avg_confidence,
            "true_positive_rate": frog_accuracy,
            "true_negative_rate": not_frog_accuracy,
            "recent_frog_images": recent_frog_images,
            "recent_not_frog_images": recent_not_frog_images
        }
        
        # Cache the results
        cache[cache_key] = {'value': metrics, 'time': time.time()}
        return metrics
    
    except Exception as e:
        logging.error(f"Error computing metrics: {e}")
        return {
            "total_frog": 0, "total_not_frog": 0, "correct_frog": 0,
            "correct_not_frog": 0, "missed_frog": 0, "false_positive": 0,
            "frog_accuracy": 0, "not_frog_accuracy": 0, "accuracy": 0,
            "avg_confidence": 0, "true_positive_rate": 0, "true_negative_rate": 0,
            "recent_frog_images": [], "recent_not_frog_images": []
        }

@app.route("/dashboard", methods=["GET"])
def dashboard():
    # Get model selection
    model_id = request.args.get("model", None)
    
    try:
        model = model_manager.get_model(model_id)
    except ValueError:
        # If invalid model, redirect to dashboard with default model
        return redirect(url_for("dashboard"))
    
    # Get available models for dropdown
    available_models = debug_models()
    
    # Define directories
    train_dir = os.environ.get("TRAIN_DIR", "frog_images/train")
    test_dir = os.environ.get("TEST_DIR", "frog_images/test")
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Count training images
    train_frog_dir = os.path.join(train_dir, "frog")
    train_not_frog_dir = os.path.join(train_dir, "not_frog")
    
    train_frog_count = count_images(train_frog_dir)
    train_not_frog_count = count_images(train_not_frog_dir)
    train_total = train_frog_count + train_not_frog_count
    
    # Check if test directory has images
    if count_images(test_dir) == 0 and not request.args.get('error'):
        return render_template_string(
            DASHBOARD_TEMPLATE,
            available_models=available_models,
            selected_model=model_id or model_manager.default_model,
            accuracy="N/A",
            true_positive_rate="N/A",
            true_negative_rate="N/A",
            avg_confidence="N/A",
            total_frog=0,
            total_not_frog=0,
            correct_frog=0,
            correct_not_frog=0,
            missed_frog=0,
            false_positive=0,
            frog_accuracy="N/A",
            not_frog_accuracy="N/A",
            recent_frog_images=[],
            recent_not_frog_images=[],
            train_frog_count=train_frog_count,
            train_not_frog_count=train_not_frog_count,
            train_total=train_total
        )
    
    # Compute performance metrics (uses caching)
    metrics = compute_performance_parallel(model, test_dir)
    
    # Add training counts to metrics
    metrics["train_frog_count"] = train_frog_count
    metrics["train_not_frog_count"] = train_not_frog_count
    metrics["train_total"] = train_total
    
    # Render dashboard with metrics
    return render_template_string(
        DASHBOARD_TEMPLATE,
        available_models=available_models,
        selected_model=model_id or model_manager.default_model,
        **metrics
    )

@app.route("/training", methods=["GET"])
def training_options():
    """Show training options page"""
    available_models = debug_models()
    
    train_dir = os.environ.get("TRAIN_DIR", "frog_images/train")
    
    # Count available training images in subdirectories
    train_frog_dir = os.path.join(train_dir, "frog")
    train_not_frog_dir = os.path.join(train_dir, "not_frog")
    
    frog_count = count_images(train_frog_dir)
    not_frog_count = count_images(train_not_frog_dir)
    
    return render_template_string(
        TRAINING_TEMPLATE,
        available_models=available_models,
        frog_count=frog_count,
        not_frog_count=not_frog_count,
        count_images=count_images
    )

@app.route("/train-roboflow", methods=["POST"])
def train_roboflow():
    """Train or fine-tune Roboflow model with feedback images"""
    epochs = int(request.form.get("epochs", 10))
    
    train_dir = os.environ.get("TRAIN_DIR", "frog_images/train")
    train_frog_dir = os.path.join(train_dir, "frog")
    train_not_frog_dir = os.path.join(train_dir, "not_frog")
    
    try:
        # Check if directories have images
        frog_count = count_images(train_frog_dir)
        not_frog_count = count_images(train_not_frog_dir)
        
        if frog_count == 0 or not_frog_count == 0:
            return f"Error: Training directories must contain images. Found {frog_count} frogs and {not_frog_count} not-frogs", 400
        
        # Run the training in a thread to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fine_tune_roboflow_model, 
                                    train_frog_dir, 
                                    train_not_frog_dir, 
                                    epochs=epochs)
            # Wait for the result, but add a timeout
            new_model_id = future.result(timeout=60*30)  # 30-minute timeout
            
        # Clear caches
        for key in list(cache.keys()):
            del cache[key]
            
        # Redirect to dashboard with new model
        return redirect(url_for("dashboard", model=new_model_id))
    except Exception as e:
        logging.error(f"Error training Roboflow model: {e}")
        return f"Error training model: {str(e)}", 500

@app.route("/train-yolo", methods=["POST"])
def train_yolo():
    """Train a YOLO model with feedback images"""
    epochs = int(request.form.get("epochs", 50))
    
    train_dir = os.environ.get("TRAIN_DIR", "frog_images/train")
    train_frog_dir = os.path.join(train_dir, "frog")
    train_not_frog_dir = os.path.join(train_dir, "not_frog")
    
    try:
        # Check if directories have images
        frog_count = count_images(train_frog_dir)
        not_frog_count = count_images(train_not_frog_dir)
        
        if frog_count == 0 or not_frog_count == 0:
            return f"Error: Training directories must contain images. Found {frog_count} frogs and {not_frog_count} not-frogs", 400
        
        # Run the training in a thread to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(train_yolo_model, 
                                    train_frog_dir, 
                                    train_not_frog_dir, 
                                    epochs=epochs)
            # Wait for the result, but add a timeout
            model_path = future.result(timeout=60*60)  # 1-hour timeout
            
        # Clear caches
        for key in list(cache.keys()):
            del cache[key]
            
        # Redirect to dashboard with new model
        return redirect(url_for("dashboard", model="yolo"))
    except Exception as e:
        logging.error(f"Error training YOLO model: {e}")
        return f"Error training model: {str(e)}", 500

@app.route("/train-tensorflow", methods=["POST"])
def train_tensorflow():
    """Train a TensorFlow model with feedback images"""
    epochs = int(request.form.get("epochs", 20))
    
    train_dir = os.environ.get("TRAIN_DIR", "frog_images/train")
    train_frog_dir = os.path.join(train_dir, "frog")
    train_not_frog_dir = os.path.join(train_dir, "not_frog")
    
    try:
        # Check if directories have images
        frog_count = count_images(train_frog_dir)
        not_frog_count = count_images(train_not_frog_dir)
        
        if frog_count == 0 or not_frog_count == 0:
            return f"Error: Training directories must contain images. Found {frog_count} frogs and {not_frog_count} not-frogs", 400
        
        # Run the training in a thread to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(train_tensorflow_model, 
                                    train_frog_dir, 
                                    train_not_frog_dir, 
                                    epochs=epochs)
            # Wait for the result, but add a timeout
            model_path = future.result(timeout=60*40)  # 40-minute timeout
            
        # Clear caches
        for key in list(cache.keys()):
            del cache[key]
            
        # Redirect to dashboard with new model
        return redirect(url_for("dashboard", model="tensorflow"))
    except Exception as e:
        logging.error(f"Error training TensorFlow model: {e}")
        return f"Error training model: {str(e)}", 500

@app.route("/optimize-model", methods=["POST"])
def optimize_model():
    """Optimize model performance"""
    try:
        # Get model ID from form
        model_id = request.form.get("model_id")
        if not model_id:
            return "Error: No model selected", 400
        
        # Get the model
        model = model_manager.get_model(model_id)
        
        # Run optimization in a thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(optimize_model_performance, model)
            # Wait for the result
            result = future.result(timeout=300)  # 5-minute timeout            
            
        # Clear caches
        for key in list(cache.keys()):
            del cache[key]
            
        return redirect(url_for("dashboard", model=model_id))
    except Exception as e:
        logging.error(f"Error optimizing model: {e}")
        return f"Error optimizing model: {str(e)}", 500

@app.route("/create-split", methods=["POST"])
def create_split():
    """Create a train/test split from existing images"""
    try:
        # Get parameters from form
        split_ratio = float(request.form.get("split_ratio", "0.8"))
        source_dir = request.form.get("source_dir", "frog_images")
        
        # Validate the source directory
        valid, message = validate_image_dir(source_dir)
        
        if not valid:
            return redirect(url_for("training_options", error=message))
        
        # Create the split
        data_dirs = create_train_test_split(source_dir=source_dir, split_ratio=split_ratio)
        
        if not data_dirs:
            return redirect(url_for("training_options", error="Failed to create train/test split"))
            
        # Update environment variables to use the new directories
        os.environ["TRAIN_DIR"] = data_dirs["train_dir"]
        os.environ["TEST_DIR"] = data_dirs["test_dir"]
        
        # Clear caches
        for key in list(cache.keys()):
            del cache[key]
            
        # Success message
        message = f"Split complete! Train: {count_images(data_dirs['train_dir'])} images, Test: {count_images(data_dirs['test_dir'])} images"
        return redirect(url_for("training_options", message=message))
    except Exception as e:
        logging.error(f"Error creating train/test split: {e}")
        return redirect(url_for("training_options", error=f"Error: {str(e)}"))

@app.route("/evaluate-model", methods=["POST"])
def evaluate_model():
    """Evaluate model performance on test data"""
    try:
        # Get model ID from form
        model_id = request.form.get("model_id")
        
        # Get the model
        model = model_manager.get_model(model_id)
        
        # Get test directory
        test_dir = os.environ.get("TEST_DIR", "frog_images/test")
        
        # Check if test directory exists and has images
        if not os.path.exists(test_dir):
            return redirect(url_for("dashboard", model=model_id, error="Test directory not found"))
            
        frog_dir = os.path.join(test_dir, "frog")
        not_frog_dir = os.path.join(test_dir, "not_frog")
        
        frog_count = count_images(frog_dir)
        not_frog_count = count_images(not_frog_dir)
        
        if frog_count == 0 and not_frog_count == 0:
            return redirect(url_for("dashboard", model=model_id, 
                                     error="No test images found. Create a train/test split first."))
        
        # Compute performance metrics
        metrics = compute_performance_parallel(model, test_dir)
        
        # Clear dashboard cache to ensure updated metrics show up
        keys_to_delete = [k for k in cache.keys() if k.startswith("perf_") or k.startswith("dashboard")]
        for key in keys_to_delete:
            if key in cache:
                del cache[key]
                
        # Redirect to dashboard with the metrics
        return redirect(url_for("dashboard", model=model_id, evaluated=True))
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        return redirect(url_for("dashboard", model=model_id, error=f"Error: {str(e)}"))

# Performance: Clean up cache periodically
@app.before_request
def cleanup_cache():
    # Run cleanup every 100 requests (approximately)
    if random.randint(1, 100) == 1:
        current_time = time.time()
        keys_to_delete = [k for k, v in cache.items() if v['time'] < current_time - (CACHE_TIMEOUT * 2)]
        for k in keys_to_delete:
            del cache[k]

# Update the main function to create the train/test split
if __name__ == "__main__":
    # Just create the necessary directories if they don't exist
    os.makedirs("frog_images/train/frog", exist_ok=True)
    os.makedirs("frog_images/train/not_frog", exist_ok=True)
    os.makedirs("frog_images/test/frog", exist_ok=True)
    os.makedirs("frog_images/test/not_frog", exist_ok=True)
    
    # Default environment variables for directories
    if "TRAIN_DIR" not in os.environ:
        os.environ["TRAIN_DIR"] = "frog_images/train"
    if "TEST_DIR" not in os.environ:
        os.environ["TEST_DIR"] = "frog_images/test"
    
    # Create uploads directory
    os.makedirs("uploads", exist_ok=True)
    
    app.run(debug=True, threaded=True)