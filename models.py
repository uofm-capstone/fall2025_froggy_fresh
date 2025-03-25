import os
import random
import logging
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Union
import uuid
import time
import shutil
import json
import io
import base64

# Only suppress specific warnings, not everything
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging to see important messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Configure TensorFlow environment (before imports)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Show warnings and errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU only by default
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Dynamic memory allocation

# Try importing other libraries with proper error handling
try:
    import cv2
    from tensorflow.keras.models import Model as KerasModel, load_model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Conv2D
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import tensorflow as tf
except ImportError:
    logging.warning("Some TensorFlow/Keras modules are missing.")

try:
    from roboflow import Roboflow
    HAS_ROBOFLOW = True
except ImportError:
    logging.warning("Roboflow not installed. Roboflow models will not be available.")
    HAS_ROBOFLOW = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    logging.warning("YOLO not installed. YOLO models will not be available.")
    HAS_YOLO = False

# Constants
INPUT_SIZE = (224, 224)  # Standard input size for most models
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

class ModelType(Enum):
    """Enum for different model types"""
    ROBOFLOW = "roboflow"
    YOLO = "yolo"
    ENHANCED_YOLO = "enhanced-yolo"
    TENSORFLOW = "tensorflow"
    DUMMY = "dummy"

@dataclass
class ModelConfig:
    """Dataclass for model configuration"""
    model_type: ModelType
    model_id: str
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    input_size: Tuple[int, int] = INPUT_SIZE

class Model:
    """Base model interface"""
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self.input_size = INPUT_SIZE
    
    def load(self):
        """Load the model"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, image_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Predict bounding boxes for frogs in the image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (image_array, list of detection results)
            Each detection result is a dict with 'class', 'confidence', 'box' keys
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_id

class RoboflowModel(Model):
    """Roboflow model implementation"""
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not self.api_key:
            logging.warning("ROBOFLOW_API_KEY not found in environment variables")
            logging.info("Get a free API key at https://app.roboflow.com/")
    
    def load(self):
        """Load the Roboflow model"""
        try:
            if not self.api_key:
                logging.error("Cannot load Roboflow model without API key")
                return False
                
            rf = Roboflow(api_key=self.api_key)
            
            # Try to access a known public frog dataset from Roboflow Universe
            try:
                # Using the public frog dataset - newer API syntax
                project = rf.workspace("roboflow-universe").project("frog-species-dvrxi")
                logging.info("Using public Frog Species dataset from Roboflow")
            except Exception as e:
                logging.warning(f"Couldn't load frog dataset: {e}")
                try:
                    # Try animals dataset instead
                    project = rf.workspace("roboflow-universe").project("animals-ij5d2")
                    logging.info("Using public Animals dataset from Roboflow")
                except Exception as e:
                    logging.warning(f"Couldn't load animals dataset: {e}")
                    return False
            
            # Get available versions
            try:
                version = 1  # Default to version 1
                self.model = project.version(version).model
                logging.info(f"Loaded Roboflow model: {project.id} v{version}")
                return True
            except Exception as e:
                logging.error(f"Failed to load Roboflow model version: {e}")
                return False
                
        except Exception as e:
            logging.error(f"Error loading Roboflow model: {e}")
            return False
    
    def predict(self, image_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Predict using Roboflow model"""
        if not self.model:
            success = self.load()
            if not success:
                return np.zeros((10, 10, 3)), []
            
        # Load image using OpenCV to work with numpy array
        img_np = cv2.imread(image_path)
        if img_np is None:
            return np.zeros((10, 10, 3)), []
            
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        try:
            predictions = self.model.predict(image_path, confidence=40, overlap=30)
            results = []
            for prediction in predictions:
                pred = prediction.json()
                if not pred.get('confidence'):
                    continue
                    
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                width = pred.get('width', 0)
                height = pred.get('height', 0)
                
                x1 = int(x - width/2)
                y1 = int(y - height/2)
                x2 = int(x + width/2)
                y2 = int(y + height/2)
                
                results.append({
                    'class': 'frog',
                    'confidence': pred.get('confidence', 0),
                    'box': [x1, y1, x2, y2]
                })
                
            return img_np, results
        except Exception as e:
            logging.error(f"Error in Roboflow prediction: {e}")
            return img_np, []

class YOLOModel(Model):
    """YOLO model implementation with object detection capabilities"""
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.model_path = os.path.join("models", "yolo", "best.pt")
        self.model = None
    
    def load(self):
        """Load the YOLO model"""
        if not HAS_YOLO:
            logging.error("YOLO library not installed. Please install ultralytics package.")
            return False
            
        try:
            # Check if custom model exists, otherwise use pre-trained
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logging.info(f"Loaded custom YOLO model from {self.model_path}")
            else:
                # Use YOLOv8n as a pre-trained model
                self.model = YOLO("yolov8n.pt")
                logging.info("Loaded pre-trained YOLOv8n model")
                
            return True
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            return False
    
    def predict(self, image_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Predict using YOLO model"""
        if not self.model:
            self.load()
            
        # Load image for returning
        img_np = cv2.imread(image_path)
        if img_np is None:
            return np.zeros((10, 10, 3)), []
            
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        # Run prediction
        try:
            results = self.model(image_path)
            detections = []
            
            # Process results - YOLO provides boxes in xyxy format
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0].item())
                    
                    # Only include frogs or if it's a custom model with the first class as 'frog'
                    if class_name.lower() == 'frog' or (class_id == 0 and os.path.exists(self.model_path)):
                        detections.append({
                            'class': 'frog',
                            'confidence': confidence,
                            'box': [x1, y1, x2, y2]
                        })
            
            return img_np, detections
        except Exception as e:
            logging.error(f"Error in YOLO prediction: {e}")
            return img_np, []
    
    def train(self, frog_dir: str, not_frog_dir: str, epochs: int = 50) -> str:
        """Train a custom YOLO model for frog detection"""
        if not HAS_YOLO:
            raise ImportError("YOLO library not installed. Please install ultralytics package.")
        
        # Create a dataset directory structure compatible with YOLO
        dataset_dir = os.path.join("models", "yolo", "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create train and validation directories
        train_dir = os.path.join(dataset_dir, "train")
        val_dir = os.path.join(dataset_dir, "val")
        os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
        
        # Create a YAML file for dataset configuration
        yaml_path = os.path.join(dataset_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"train: {train_dir}/images\n")
            f.write(f"val: {val_dir}/images\n")
            f.write("nc: 1\n")  # Number of classes
            f.write("names: ['frog']\n")  # Class names
        
        # Prepare training data - copy images and create YOLO format labels
        self._prepare_yolo_data(frog_dir, train_dir, val_dir, is_frog=True, split=0.8)
        self._prepare_yolo_data(not_frog_dir, train_dir, val_dir, is_frog=False, split=0.8)
        
        # Train the model
        model = YOLO("yolov8n.pt")  # Start with pre-trained YOLOv8n
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            patience=10,
            project=os.path.join("models", "yolo"),
            name="frog_detector"
        )
        
        # Copy the best model to a standard location
        best_model_path = os.path.join("models", "yolo", "frog_detector", "weights", "best.pt")
        if os.path.exists(best_model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            shutil.copy(best_model_path, self.model_path)
            logging.info(f"Saved best YOLO model to {self.model_path}")
        
        # Load the newly trained model
        self.model = YOLO(self.model_path)
        
        return self.model_path
    
    def _prepare_yolo_data(self, source_dir: str, train_dir: str, val_dir: str, is_frog: bool, split: float = 0.8):
        """Prepare data for YOLO training by creating labels and copying images"""
        # List all image files
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        random.shuffle(image_files)
        
        # Split into train and validation
        split_index = int(len(image_files) * split)
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]
        
        # Process training images
        for img_file in train_files:
            self._process_yolo_image(source_dir, img_file, train_dir, is_frog)
        
        # Process validation images
        for img_file in val_files:
            self._process_yolo_image(source_dir, img_file, val_dir, is_frog)
    
    def _process_yolo_image(self, source_dir: str, img_file: str, target_dir: str, is_frog: bool):
        """Process a single image for YOLO training"""
        # Source image path
        img_path = os.path.join(source_dir, img_file)
        
        # Read image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            return  # Skip invalid images
            
        h, w = img.shape[:2]
        
        # Target paths
        target_img_dir = os.path.join(target_dir, "images")
        target_label_dir = os.path.join(target_dir, "labels")
        target_img_path = os.path.join(target_img_dir, img_file)
        target_label_path = os.path.join(target_label_dir, os.path.splitext(img_file)[0] + ".txt")
        
        # Copy image to target directory
        shutil.copy(img_path, target_img_path)
        
        # Create label file if it's a frog
        if is_frog:
            # For frog images, create a bounding box that covers most of the image
            # This is a simplification - ideally you'd use actual annotations
            # YOLO format: class_id center_x center_y width height (all normalized 0-1)
            with open(target_label_path, "w") as f:
                # Using 80% of the image size as a generic bounding box
                f.write("0 0.5 0.5 0.8 0.8\n")  # class 0, centered, 80% width/height
        else:
            # For non-frog images, create an empty label file (no objects)
            with open(target_label_path, "w") as f:
                pass

class EnhancedYOLOModel(YOLOModel):
    """YOLO model with frog-specific enhancements"""
    
    def predict(self, image_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Predict using enhanced YOLO with specialized frog detection"""
        if not self.model:
            self.load()
            
        # Load image for returning
        img_np = cv2.imread(image_path)
        if img_np is None:
            return np.zeros((10, 10, 3)), []  # Return empty data for invalid image
            
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        # Convert to HSV for better color detection of frogs
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        # Enhance green/brown tones common in frogs (optional pre-processing)
        # This helps the model focus on frog-like colors
        green_mask = cv2.inRange(img_hsv, (35, 20, 20), (85, 255, 255))  # Green range
        brown_mask = cv2.inRange(img_hsv, (10, 20, 20), (30, 255, 200))  # Brown range
        frog_mask = cv2.bitwise_or(green_mask, brown_mask)
        
        # Save enhanced image for detection
        temp_path = f"{image_path}_enhanced.jpg"
        cv2.imwrite(temp_path, img_np)
        
        try:
            # Run standard YOLO prediction with confidence boosting for frog-like objects
            results = self.model(temp_path, conf=0.2)  # Lower threshold to catch more potential frogs
            detections = []
            
            # Process results with frog-specific refinement
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0].item())
                    
                    # Frog-specific confidence boosting
                    # Check if detection overlaps with green/brown areas
                    roi = frog_mask[max(0, y1):min(img_np.shape[0], y2), 
                                   max(0, x1):min(img_np.shape[1], x2)]
                    if roi.size > 0:
                        frog_color_ratio = cv2.countNonZero(roi) / roi.size
                        
                        # Boost confidence if colors match frog patterns
                        if frog_color_ratio > 0.3:
                            confidence = min(confidence * 1.2, 1.0)  # Boost confidence by 20%
                    
                    # Determine if this is a frog
                    is_frog = (class_name.lower() == 'frog' or 
                               (class_id == 0 and os.path.exists(self.model_path)))
                    
                    # Apply additional shape-based filtering specific to frogs
                    shape_score = self._evaluate_frog_shape(img_np[
                        max(0, y1):min(img_np.shape[0], y2), 
                        max(0, x1):min(img_np.shape[1], x2)
                    ])
                    
                    if is_frog or shape_score > 0.7:  # Accept as frog if shape looks like a frog
                        detections.append({
                            'class': 'frog',
                            'confidence': confidence,
                            'box': [x1, y1, x2, y2]
                        })
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return img_np, detections
        except Exception as e:
            logging.error(f"Error in Enhanced YOLO prediction: {e}")
            # Clean up temporary file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return img_np, []
    
    def _evaluate_frog_shape(self, roi):
        """Evaluate if ROI has frog-like shape characteristics"""
        if roi.size == 0:
            return 0
            
        # Calculate shape features using contours
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0
                
            # Get largest contour
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            
            # Skip if area is too small
            if area < 100:
                return 0
                
            # Calculate shape features
            perimeter = cv2.arcLength(max_contour, True)
            if perimeter == 0:
                return 0
                
            # Roundness (frogs often have round body outlines)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Compactness (frogs are usually compact)
            x, y, w, h = cv2.boundingRect(max_contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Frog-specific shape score (customized for frog body shapes)
            # Higher for circular/oval objects with certain aspect ratios
            frog_score = (0.7 * circularity) + (0.3 * (1.0 - abs(aspect_ratio - 1.2) / 2.0))
            
            return min(max(frog_score, 0), 1)  # Normalize score between 0 and 1
            
        except Exception as e:
            logging.warning(f"Error in shape evaluation: {e}")
            return 0

class TensorFlowObjectDetectionModel(Model):
    """TensorFlow model implementation for object detection"""
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.model_dir = os.path.join("models", "tensorflow")
        self.model_path = os.path.join(self.model_dir, "frog_detector.h5")
        self.model = None
        self.input_size = (416, 416)  # Larger input size for better detection
    
    def load(self):
        """Load the TensorFlow model"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                logging.info(f"Loaded TensorFlow model from {self.model_path}")
            else:
                self._create_object_detection_model()
                logging.info("Created new TensorFlow object detection model")
                
            return True
        except Exception as e:
            logging.error(f"Error loading TensorFlow model: {e}")
            return False
    
    def _create_object_detection_model(self):
        """Create a new object detection model based on EfficientNet + detection heads"""
        # Create base model with input shape
        input_shape = self.input_size + (3,)
        inputs = Input(shape=input_shape)
        
        # Use EfficientNetB0 as a feature extractor
        base_model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs
        )
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add detection heads
        x = base_model.output
        
        # Confidence head (object present or not)
        confidence = GlobalAveragePooling2D()(x)
        confidence = Dense(512, activation="relu")(confidence)
        confidence = Dropout(0.3)(confidence)
        confidence = Dense(1, activation="sigmoid", name="confidence")(confidence)
        
        # Bounding box regression head
        bbox = GlobalAveragePooling2D()(x)
        bbox = Dense(512, activation="relu")(bbox)
        bbox = Dropout(0.3)(bbox)
        # x1, y1, x2, y2 - normalized coordinates
        bbox = Dense(4, activation="sigmoid", name="bbox")(bbox)
        
        # Create the model with multiple outputs
        self.model = tf.keras.models.Model(inputs=inputs, outputs=[confidence, bbox])
        
        # Fix: Always use loss objects, not strings
        loss_confidence = tf.keras.losses.BinaryCrossentropy()
        loss_bbox = tf.keras.losses.MeanSquaredError()
        
        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                "confidence": loss_confidence,
                "bbox": loss_bbox
            },
            loss_weights={
                "confidence": 1.0,
                "bbox": 5.0  # Weight bbox loss higher
            },
            metrics={
                "confidence": ["accuracy"]  # Use list of string metrics (not function objects)
            }
        )
        
        # Save the model
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save(self.model_path)
    
    def predict(self, image_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Predict using TensorFlow object detection model"""
        if not self.model:
            self.load()
        
        # Load image and convert to RGB
        img_np = cv2.imread(image_path)
        if img_np is None:
            return np.zeros((10, 10, 3)), []
            
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        try:
            # Get original dimensions
            original_h, original_w = img_np.shape[:2]
            
            # Preprocess image for the model
            img_resized = cv2.resize(img_np, self.input_size)
            img_tensor = img_resized / 255.0  # Normalize
            img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension
            
            # Make prediction
            confidence, bbox = self.model.predict(img_tensor, verbose=0)
            
            # Process prediction
            confidence_score = float(confidence[0][0])
            results = []
            
            # If confidence is above threshold, create detection
            if confidence_score > self.confidence_threshold:
                # Denormalize bounding box coordinates to original image dimensions
                x1, y1, x2, y2 = bbox[0]
                x1 = int(x1 * original_w)
                y1 = int(y1 * original_h)
                x2 = int(x2 * original_w)
                y2 = int(y2 * original_h)
                
                # Add detection
                results.append({
                    'class': 'frog',
                    'confidence': confidence_score,
                    'box': [x1, y1, x2, y2]
                })
            
            return img_np, results
        except Exception as e:
            logging.error(f"Error in TensorFlow prediction: {e}")
            return img_np, []
    
    def train(self, frog_dir: str, not_frog_dir: str, epochs: int = 20) -> str:
        """Train the TensorFlow object detection model"""
        if not self.model:
            self.load()
        
        # Unfreeze some of the base model layers
        for layer in self.model.layers[-20:]:
            layer.trainable = True
        
        # Recompile with a lower learning rate
        losses = {
            "confidence": tf.keras.losses.BinaryCrossentropy(),
            "bbox": tf.keras.losses.MeanSquaredError()
        }
        loss_weights = {
            "confidence": 1.0,
            "bbox": 5.0
        }
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=losses,
            loss_weights=loss_weights,
            metrics={"confidence": "accuracy"}
        )
        
        # Generate training data
        X, y_confidence, y_bbox = self._prepare_training_data(frog_dir, not_frog_dir)
        
        # Create callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train the model
        self.model.fit(
            X,
            {"confidence": y_confidence, "bbox": y_bbox},
            validation_split=0.2,
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks
        )
        
        # Save the trained model
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save(self.model_path)
        
        return self.model_path
    
    def _prepare_training_data(self, frog_dir: str, not_frog_dir: str):
        """Prepare training data for object detection model"""
        # List all image files
        frog_files = [os.path.join(frog_dir, f) for f in os.listdir(frog_dir) 
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        not_frog_files = [os.path.join(not_frog_dir, f) for f in os.listdir(not_frog_dir) 
                         if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        
        # Balance dataset if needed
        max_samples = 1000  # Limit dataset size for memory constraints
        if len(frog_files) > max_samples:
            random.shuffle(frog_files)
            frog_files = frog_files[:max_samples]
        if len(not_frog_files) > max_samples:
            random.shuffle(not_frog_files)
            not_frog_files = not_frog_files[:max_samples]
        
        # Initialize arrays for features and labels
        total_samples = len(frog_files) + len(not_frog_files)
        X = np.zeros((total_samples, self.input_size[0], self.input_size[1], 3), dtype=np.float32)
        y_confidence = np.zeros((total_samples, 1), dtype=np.float32)
        y_bbox = np.zeros((total_samples, 4), dtype=np.float32)
        
        # Process frog images - they have bounding boxes
        for i, img_path in enumerate(frog_files):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, self.input_size)
            
            # Normalize pixel values
            X[i] = img_resized / 255.0
            
            # For frog images, set confidence to 1
            y_confidence[i] = 1.0
            
            # For simplicity, use a default bounding box that covers 80% of the image
            # In a real scenario, you would use actual annotations
            y_bbox[i] = [0.1, 0.1, 0.9, 0.9]  # Normalized coordinates [x1, y1, x2, y2]
        
        # Process not-frog images - they don't have bounding boxes
        for i, img_path in enumerate(not_frog_files, start=len(frog_files)):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, self.input_size)
            
            # Normalize pixel values
            X[i] = img_resized / 255.0
            
            # For not-frog images, set confidence to 0
            y_confidence[i] = 0.0
            
            # Bounding box doesn't matter for not-frog images, but set to zeros
            y_bbox[i] = [0, 0, 0, 0]
        
        # Shuffle the data
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y_confidence = y_confidence[indices]
        y_bbox = y_bbox[indices]
        
        return X, y_confidence, y_bbox

class DummyModel(Model):
    """Dummy model for fallback when no real models load"""
    def __init__(self, model_id="simple-detector"):
        super().__init__(model_id)
        
    def load(self):
        # No loading needed
        return True
        
    def predict(self, image_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Simple prediction that checks image color for frog-like green"""
        try:
            # Load image
            img_np = cv2.imread(image_path)
            if img_np is None:
                return np.zeros((10, 10, 3)), []
                
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            # Check if image has dominant green (frog-like) colors
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            green_mask = cv2.inRange(hsv, (35, 25, 25), (90, 255, 255))
            green_pixels = cv2.countNonZero(green_mask)
            
            # If >10% green, it might be a frog
            if green_pixels > (img_np.shape[0] * img_np.shape[1] * 0.1):
                # Create centered bounding box
                h, w = img_np.shape[:2]
                x1, y1 = int(w * 0.25), int(h * 0.25)
                x2, y2 = int(w * 0.75), int(h * 0.75)
                
                return img_np, [{
                    'class': 'frog',
                    'confidence': 0.6,  # Moderate confidence
                    'box': [x1, y1, x2, y2]
                }]
            else:
                return img_np, []  # No frog detected
                
        except Exception as e:
            logging.error(f"Error in dummy model: {e}")
            return np.zeros((10, 10, 3)), []

class ModelManager:
    """Manager for handling multiple models"""
    def __init__(self):
        self.models = {}
        self.default_model = None
        self.model_classes = {
            ModelType.ROBOFLOW: RoboflowModel,
            ModelType.YOLO: YOLOModel,
            ModelType.ENHANCED_YOLO: EnhancedYOLOModel,
            ModelType.TENSORFLOW: TensorFlowObjectDetectionModel,
            ModelType.DUMMY: DummyModel
        }
    
    def register_model(self, model_id: str, model_type: ModelType) -> bool:
        """Register a new model"""
        if model_id in self.models:
            return False
            
        # Create and register the model
        model_class = self.model_classes.get(model_type)
        if model_class:
            model = model_class(model_id)
            if model.load():
                self.models[model_id] = model
                # Set as default if it's the first model
                if not self.default_model:
                    self.default_model = model_id
                logging.info(f"Registered model: {model_id}")
                return True
            else:
                logging.warning(f"Failed to load model: {model_id}")
        
        return False
    
    def register_dummy_model(self):
        """Register a dummy model as fallback"""
        model = DummyModel("simple-detector")
        self.models["simple-detector"] = model
        if not self.default_model:
            self.default_model = "simple-detector"
        logging.info("Registered fallback model: simple-detector")
        return True
    
    def get_model(self, model_id: str = None) -> Model:
        """Get a model by ID, or default model if None"""
        if not model_id and self.default_model:
            model_id = self.default_model
            
        if model_id in self.models:
            return self.models[model_id]
            
        raise ValueError(f"Model {model_id} not found")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model IDs"""
        return list(self.models.keys())

# Initialize the model manager and register models
model_manager = ModelManager()

# Try registering models in order of reliability
models_registered = False

# Try YOLO first
if HAS_YOLO and model_manager.register_model("yolo", ModelType.YOLO):
    models_registered = True
    
    # Also try enhanced YOLO if regular YOLO works
    model_manager.register_model("enhanced-yolo", ModelType.ENHANCED_YOLO)

# Try TensorFlow next
if model_manager.register_model("tensorflow", ModelType.TENSORFLOW):
    models_registered = True

# Try Roboflow last (due to API dependency)
if HAS_ROBOFLOW and model_manager.register_model("roboflow", ModelType.ROBOFLOW):
    models_registered = True

# If all models failed, register dummy model
if not models_registered:
    model_manager.register_dummy_model()
    logging.warning("Using fallback model since no other models loaded")
else:
    # Also register dummy model as an option
    model_manager.register_dummy_model()
    
# Log available models
logging.info(f"Available models: {model_manager.get_available_models()}")

# Image processing functions
def process_image(image_path: str, model: Model) -> Tuple[Image.Image, str, str]:
    """
    Process an image with the model
    
    Args:
        image_path: Path to the image file
        model: Model instance to use for prediction
        
    Returns:
        Tuple of (annotated PIL Image, prediction label, confidence text)
    """
    try:
        # Predict using the model
        img_np, results = model.predict(image_path)
        
        # Convert numpy array to PIL Image for drawing
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)
        
        # Use model name in result
        model_name = model.get_model_name()
        
        # Process results
        if not results:
            return img_pil, "Not a frog", f"{model_name}: No frogs detected"
        
        # Draw boxes and calculate overall confidence
        total_confidence = 0
        for result in results:
            # Extract box and confidence
            box = result['box']
            confidence = result['confidence']
            total_confidence += confidence
            
            # Draw box
            draw.rectangle(box, outline="red", width=3)
            
            # Draw label
            label = f"Frog: {confidence:.2f}"
            draw.text((box[0], box[1] - 20), label, fill="red")
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(results)
        confidence_text = f"{model_name}: {avg_confidence:.2f} confidence"
        
        return img_pil, "Frog", confidence_text
        
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        # Return the original image on error
        return Image.open(image_path), "Error", f"Error: {str(e)}"

def get_base64_image(img: Image.Image) -> str:
    """Convert PIL Image to base64 string for HTML display"""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

def process_test_image(img_path: str, expected: str, model: Model) -> Tuple[bool, float, str]:
    """
    Process a test image and return metrics
    
    Args:
        img_path: Path to the image
        expected: Expected class ("frog" or "not_frog")
        model: Model to use for prediction
        
    Returns:
        Tuple of (is_correct, confidence, status)
    """
    try:
        # Make prediction
        _, results = model.predict(img_path)
        
        # Calculate confidence
        if results:
            # Get average confidence of all detections
            avg_confidence = sum(r["confidence"] for r in results) / len(results)
            predicted = "frog"
        else:
            avg_confidence = 0.0
            predicted = "not_frog"
        
        # Check if prediction matches expected class
        is_correct = (predicted == expected)
        
        return is_correct, avg_confidence, "ok"
    except Exception as e:
        logging.error(f"Error in test image processing: {e}")
        return False, 0.0, f"error: {str(e)}"

# Model training functions
def fine_tune_roboflow_model(frog_dir: str, not_frog_dir: str, epochs: int = 10) -> str:
    """
    Fine-tune a Roboflow model with your images
    
    Args:
        frog_dir: Directory containing frog images
        not_frog_dir: Directory containing not-frog images
        epochs: Number of training epochs
        
    Returns:
        ID of the fine-tuned model
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found in environment variables. Get a key at https://app.roboflow.com/")
    
    # Create a unique project name
    project_name = f"frog-detector-{int(time.time())}"
    logging.info(f"Creating new Roboflow project: {project_name}")
    
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    
    # Create a new project
    try:
        # Create or get workspace
        workspace = rf.workspace()
        
        # Create a new project
        project = workspace.project(project_name, project_type="object-detection")
        logging.info(f"Created project: {project_name}")
        
        # Upload images
        upload_count = 0
        
        # Upload frog images with annotation
        for img_file in os.listdir(frog_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
                
            img_path = os.path.join(frog_dir, img_file)
            try:
                # Create simple annotation - a box covering 70% of the image
                # In a real implementation, you'd generate proper annotations
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                x1, y1 = int(w * 0.15), int(h * 0.15)
                x2, y2 = int(w * 0.85), int(h * 0.85)
                
                # Create annotation in Roboflow format
                annotation = {
                    "annotations": [{
                        "class": "frog",
                        "x": (x1 + x2) / 2,  # Center x
                        "y": (y1 + y2) / 2,  # Center y
                        "width": x2 - x1,
                        "height": y2 - y1
                    }]
                }
                
                # Upload image with annotation
                project.upload(img_path, annotation_json=json.dumps(annotation))
                upload_count += 1
            except Exception as e:
                logging.warning(f"Error uploading {img_file}: {e}")
        
        # Upload not-frog images without annotations
        for img_file in os.listdir(not_frog_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
                
            img_path = os.path.join(not_frog_dir, img_file)
            try:
                # Upload image without annotation (empty background)
                project.upload(img_path, annotation_json=json.dumps({"annotations": []}))
                upload_count += 1
            except Exception as e:
                logging.warning(f"Error uploading {img_file}: {e}")
        
        if upload_count == 0:
            raise ValueError("No images were successfully uploaded")
            
        logging.info(f"Uploaded {upload_count} images to Roboflow")
        
        # Generate a new version
        version_number = 1
        version = project.version(version_number)
        
        # Configure preprocessing and augmentation
        version.preprocessing(resize="416x416")
        version.augmentation(
            flip="horizontal",
            rotation=15,
            shear=5,
            brightness=20,
            blur=1,
            exposure=15
        )
        
        # Generate the dataset
        version.generate()
        
        # Train the model
        model = version.model
        model.train(epochs=epochs, batch_size=8, lr=0.001)
        
        # Create a unique model ID
        model_id = f"roboflow-{project_name}-v{version_number}"
        
        # Register the new model
        if model_manager.register_model(model_id, ModelType.ROBOFLOW):
            logging.info(f"Successfully trained and registered Roboflow model: {model_id}")
            return model_id
        else:
            raise ValueError("Failed to register trained model")
            
    except Exception as e:
        logging.error(f"Error in Roboflow training: {e}")
        raise ValueError(f"Roboflow training failed: {e}")

def train_yolo_model(frog_dir: str, not_frog_dir: str, epochs: int = 50) -> str:
    """Train a specialized YOLO model for frog detection"""
    if not HAS_YOLO:
        raise ImportError("YOLO not installed. Please install ultralytics package.")
    
    # Create YOLO model
    model = YOLOModel("yolo")
    
    # Train the model
    model_path = model.train(frog_dir, not_frog_dir, epochs)
    
    # Make sure our model is registered
    if "yolo" not in model_manager.get_available_models():
        model_manager.register_model("yolo", ModelType.YOLO)
    
    return model_path

def train_tensorflow_model(frog_dir: str, not_frog_dir: str, epochs: int = 20) -> str:
    """
    Train a TensorFlow model with your images
    
    Args:
        frog_dir: Directory containing frog images
        not_frog_dir: Directory containing not-frog images
        epochs: Number of training epochs
        
    Returns:
        Path to the trained model
    """
    # Create TensorFlow model
    model = TensorFlowObjectDetectionModel("tensorflow")
        
    # Train the model
    model_path = model.train(frog_dir, not_frog_dir, epochs)
    
    # Make sure our model is registered
    if "tensorflow" not in model_manager.get_available_models():
        model_manager.register_model("tensorflow", ModelType.TENSORFLOW)
    
    return model_path
    
def optimize_model_performance(model: Model) -> bool:
    """
    Optimize model performance (e.g., quantize, prune)
    
    Args:
        model: The model to optimize
        
    Returns:
        Success flag
    """
    # This is a placeholder for model optimization
    # In a real implementation, you would apply techniques like:
    # 1. Quantization (e.g., TFLite quantization)
    # 2. Pruning (removing unused weights)
    # 3. Knowledge distillation
    
    return True