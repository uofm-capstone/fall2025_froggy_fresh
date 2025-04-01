import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime

def process_images(folder_path):
    model = load_model(os.path.join(".", "backend", "frog_detector.h5"))

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
        "currentFile": last_file
    }

    # Save this run’s stats to a file for later retrieval.
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