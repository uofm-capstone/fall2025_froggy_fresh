import os
import sys
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime

def process_images(folder_path, model_path):
    model = load_model(model_path)

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
            last_file = img_path

            current_file_data = {
                "name": os.path.basename(img_path),
                "absolutePath": img_path,
                "classification": label,
                "confidence": round(file_conf * 100)  # store as percentage integer
            }
            processed_files.append(current_file_data)
            update_data = {
                "currentFile": current_file_data,
                "progress": {
                    "frogs": frog_count,
                    "notFrogs": not_frog_count,
                    "averageConfidence": round((confidence_total * 100) / len(processed_files)) if len(processed_files) != 0 else 0,
                    "processedImageCount": len(processed_files),
                    "totalImageCount": len(image_files),
                }
            }
            print(json.dumps(update_data))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    total_processed = frog_count + not_frog_count
    average_confidence = round((confidence_total / total_processed) * 100) if total_processed > 0 else 0

    run_time = datetime.now().strftime("%Y-%m-%dT%H_%M_%S") # 2025-04-10T17_54_30 (ISO 8601)
    stats = {
        "runDate": run_time,
        "frogs": frog_count,
        "notFrogs": not_frog_count,
        "averageConfidence": average_confidence,
        "results": processed_files,
        "totalFiles": f"{total_processed}",
    }

    # Save this run’s stats to a file for later retrieval.
    runs_folder = os.path.join(os.path.expanduser("~"), "Documents", "Leapfrog", "runs")
    os.makedirs(runs_folder, exist_ok=True) # creates parent folders if they dont exist

    new_run_path = os.path.join(runs_folder, f"{run_time}.json")
    with open(new_run_path, "w") as f:
        json.dump(stats, f)

    return stats

if __name__ == "__main__":
    model_path = sys.argv[1]
    folder_path = sys.argv[2]
    process_images(folder_path, model_path)