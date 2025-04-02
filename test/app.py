import os
import logging
import base64
import io
from PIL import Image
import numpy as np
import json
from flask import Flask, request, render_template_string, jsonify, session
from inference_sdk import InferenceHTTPClient

# Import from our modular components
from config import (
    API_KEY, MODEL_ID, DEFAULT_TEST_DIR,
    DETECTION_CONFIDENCE, ROBOFLOW_API_URL, ROBOFLOW_CONFIDENCE,
    ROBOFLOW_OVERLAP, DEBUG, FLASK_PORT, FLASK_HOST
)
from templates import HTML_TEMPLATE
from services import (
    get_available_models, load_model, format_yolo_predictions,
    run_batch_test, process_image, process_image_yolo, get_model_info
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Configure logging from config
if DEBUG:
    app.debug = True

# Initialize the Roboflow client - with explicit API URL
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",  # Hardcode to ensure correct URL
    api_key=API_KEY
)

@app.route('/')
def index():
    """Render the main page."""
    return render_template_string(
        HTML_TEMPLATE,
        available_models=get_available_models(),
        user_prefs=get_user_prefs()
    )

@app.route('/infer', methods=['POST'])
def infer_image():
    """Process an uploaded image for object detection."""
    if 'image' not in request.files:
        return render_template_string(HTML_TEMPLATE,
                                      error="No image file provided",
                                      available_models=get_available_models(),
                                      user_prefs=get_user_prefs())

    # Get the uploaded image and selected model ID
    image_file = request.files['image']
    model_id = request.form.get('model_id', 'yolov8n')
    
    # Save the selected model to session
    session['selected_model'] = model_id
    
    # Save the image to a temporary location
    temp_path = os.path.join(os.path.dirname(__file__), "temp_img.jpg")
    image_file.save(temp_path)
    
    try:
        # For YOLO models, use process_image_yolo to get annotated image
        if model_id != 'roboflow' and not model_id.endswith('.h5'):
            result = process_image_yolo(temp_path, model_id)
        else:
            # For Roboflow or .h5 models, use process_image
            result = process_image(temp_path, CLIENT, model_id)
            
            # If no image_data in result, add it (for Roboflow and .h5 models)
            if 'image_data' not in result and result.get('success', False):
                # Open the original image and convert to base64
                with open(temp_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                result['image_data'] = img_data
                
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Render the result
        if result['success']:
            return render_template_string(HTML_TEMPLATE,
                                        image_data=result.get('image_data'),
                                        predictions=result.get('predictions', []),
                                        available_models=get_available_models(),
                                        user_prefs=get_user_prefs())
        else:
            return render_template_string(HTML_TEMPLATE,
                                        error=result.get('error', 'Unknown error occurred'),
                                        available_models=get_available_models(),
                                        user_prefs=get_user_prefs())
    
    except Exception as e:
        # Clean up temporary file in case of exception
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        logging.error(f"Error in inference: {str(e)}")
        return render_template_string(HTML_TEMPLATE,
                                    error=f"Error processing image: {str(e)}",
                                    available_models=get_available_models(),
                                    user_prefs=get_user_prefs())

@app.route('/batch_test', methods=['POST'])
def batch_test():
    """Run batch testing on a directory of images."""
    try:
        # Get test directory from form or use default
        test_dir = request.form.get('test_dir', '').strip()
        
        # Handle folder uploads - this is a fallback, as the JS should set the text input
        if 'folderInput' in request.files:
            # This is a simplified approach - in production you'd need more complex logic
            test_dir = DEFAULT_TEST_DIR
        
        # Use default if no directory provided
        if not test_dir:
            test_dir = DEFAULT_TEST_DIR
            
        model_id = request.form.get('model_id', 'yolov8n')

        # Save user selections to session
        session['test_dir'] = test_dir
        session['selected_model'] = model_id

        # Validate directory exists
        if not os.path.exists(test_dir):
            return render_template_string(HTML_TEMPLATE,
                                        error=f"Test directory not found: {test_dir}",
                                        available_models=get_available_models(),
                                        user_prefs=get_user_prefs())

        # Check if directory has images
        image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            return render_template_string(HTML_TEMPLATE,
                                        error=f"No image files found in {test_dir}",
                                        available_models=get_available_models(),
                                        user_prefs=get_user_prefs())

        # Determine if we're using a YOLO model or Roboflow API
        if model_id == 'roboflow':
            # Run batch test with Roboflow API
            batch_results = run_batch_test(test_dir, CLIENT, 'roboflow')  # Use 'roboflow' as the identifier
            model_type = 'Roboflow API'
        else:
            # Run batch test with YOLO model
            batch_results = run_batch_test(test_dir, CLIENT, model_id)
            model_type = model_id

        # Return results
        return render_template_string(
            HTML_TEMPLATE,
            batch_results=batch_results,
            model_used=model_type,
            available_models=get_available_models(),
            user_prefs=get_user_prefs(),
            success=f"Successfully processed {batch_results['total']} images"
        )
    except Exception as e:
        logging.error(f"Batch test error: {str(e)}")
        return render_template_string(HTML_TEMPLATE,
                                    error=f"Batch test failed: {str(e)}",
                                    available_models=get_available_models(),
                                    user_prefs=get_user_prefs())

@app.route('/model_info', methods=['GET'])
def model_info_endpoint():
    """Display information about the selected YOLO model."""
    try:
        # Get the requested model ID
        model_id = request.args.get('model_id', None)

        # Update session with selected model for info
        session['info_model_id'] = model_id

        # Get model info from our service
        model_info = get_model_info(model_id)

        # Ensure the data is JSON serializable
        if not model_info.get("success", False):
            return jsonify(model_info)  # Return error as is

        # Convert potentially non-serializable objects to basic types
        if "model_info" in model_info:
            info = model_info["model_info"]
            # Convert class names to a simple dictionary of strings
            if "class_names" in info and info["class_names"] is not None:
                try:
                    # Try to convert to a dict of strings if it's not already
                    info["class_names"] = {str(k): str(v) for k, v in info["class_names"].items()}
                except (AttributeError, TypeError):
                    # If conversion fails, provide a simplified version
                    info["class_names"] = {"0": "frog"}

            # Convert other potentially complex fields to strings
            for key in ["model_type", "task"]:
                if key in info and not isinstance(info[key], (str, int, float, bool, type(None))):
                    info[key] = str(info[key])

        return jsonify(model_info)
    except Exception as e:
        logging.error(f"Model info error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/debug_config')
def debug_config():
    """Show current configuration values (excluding sensitive data)."""
    from config import MODEL_ID, ROBOFLOW_API_URL
    
    # Don't show the full API key for security
    api_key_preview = API_KEY[:4] + "***" if API_KEY else "Not set"
    
    debug_info = {
        "MODEL_ID": MODEL_ID,
        "API_KEY_PREVIEW": api_key_preview,
        "ROBOFLOW_API_URL": ROBOFLOW_API_URL,
        "DEFAULT_TEST_DIR": DEFAULT_TEST_DIR
    }
    
    return jsonify(debug_info)

# Helper function to get user preferences from session
def get_user_prefs():
    """Get user preferences from session with defaults."""
    available_models = get_available_models()
    return {
        'selected_model': session.get('selected_model', available_models[0]['id'] if available_models else 'yolov8n'),
        'test_dir': session.get('test_dir', DEFAULT_TEST_DIR),
        'info_model_id': session.get('info_model_id', None)
    }

if __name__ == '__main__':
    # For development only - in production use a proper WSGI server
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG)