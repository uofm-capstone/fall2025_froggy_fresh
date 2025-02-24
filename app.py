import os
import logging
import warnings
import traceback
import tempfile
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, render_template_string
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

# Load environment variables from .env file.
load_dotenv()

# Suppress extra logs and warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Import the Roboflow InferenceHTTPClient.
from inference_sdk import InferenceHTTPClient

# Initialize the Roboflow client.
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", 
    api_key=ROBOFLOW_API_KEY
)

app = Flask(__name__)

# Helper: count images in a directory.
def count_images(directory, extensions=('.jpg', '.jpeg', '.png')):
    return sum(1 for file in os.listdir(directory) if file.lower().endswith(extensions))

# Home / Predict Page Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Frog Identifier</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap CSS -->
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
  <style>
    body {
      background: #e9ecef;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }
    .container {
      margin-top: 60px;
    }
    .card {
      border: none;
      border-radius: 1rem;
      box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .card-header {
      background-color: #007bff;
      color: white;
      border-top-left-radius: 1rem;
      border-top-right-radius: 1rem;
      text-align: center;
    }
    .result-img {
      max-width: 100%;
      height: auto;
      border-radius: 0.5rem;
      margin-top: 15px;
    }
    .custom-file-label::after {
      content: "Browse";
    }
    /* Loading overlay for home page */
    #loading {
      display: none; 
      position: fixed; 
      top: 0; left: 0; 
      width: 100%; height: 100%; 
      background: rgba(255,255,255,0.8); 
      z-index: 9999; 
      text-align: center; 
      padding-top: 20%;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="card mx-auto">
      <div class="card-header">
        <h2>{{ prediction }}</h2>
        {% if confidence_text %}
          <p class="mb-0">{{ confidence_text }}</p>
        {% endif %}
      </div>
      <div class="card-body">
        <form action="/predict" method="post" enctype="multipart/form-data">
          <div class="custom-file mb-3">
            <input type="file" class="custom-file-input" id="file" name="file" accept="image/*" required>
            <label class="custom-file-label" for="file">Choose image</label>
          </div>
          <button type="submit" class="btn btn-primary btn-block">Upload and Identify</button>
        </form>
        {% if output_image %}
          <hr>
          <img class="result-img" src="data:image/jpeg;base64,{{ output_image }}" alt="Result Image">
        {% endif %}
      </div>
    </div>
    <br>
    <div class="text-center">
      <a href="/dashboard" class="btn btn-secondary">View Performance Dashboard</a>
    </div>
  </div>
  <!-- Loading overlay for home page -->
  <div id="loading">
    <div class="spinner-border text-primary" role="status">
      <span class="sr-only">Loading Dashboard...</span>
    </div>
    <p>Loading Dashboard...</p>
  </div>
  <!-- jQuery and Bootstrap JS -->
  <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
  <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
  <script>
    $(".custom-file-input").on("change", function(){
      var fileName = $(this).val().split("\\\\").pop();
      $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
    });
    // Show loading overlay on dashboard link click.
    document.addEventListener("DOMContentLoaded", function(){
      var dashboardLink = document.querySelector('a[href="/dashboard"]');
      if(dashboardLink){
        dashboardLink.addEventListener("click", function(){
          document.getElementById("loading").style.display = "block";
        });
      }
    });
  </script>
</body>
</html>
'''

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, prediction="Frog Identifier", confidence_text="", output_image="")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    try:
        img_bytes = file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        logging.info("Image loaded.")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        result = CLIENT.infer(tmp_path, model_id="frog-ukiu5/1")
        logging.info("Roboflow result: %s", result)
        os.remove(tmp_path)
        predictions = result.get("predictions", [])
        if predictions:
            best_pred = max(predictions, key=lambda p: p.get("confidence", 0))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", size=16)
            except Exception:
                font = ImageFont.load_default()
            x_center = best_pred.get("x")
            y_center = best_pred.get("y")
            width = best_pred.get("width")
            height = best_pred.get("height")
            left = x_center - width / 2
            top = y_center - height / 2
            right = x_center + width / 2
            bottom = y_center + height / 2
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            label = best_pred.get("class", "N/A")
            confidence = best_pred.get("confidence", 0)
            text = f"{label} {confidence*100:.2f}%"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle([left, top - text_height, left + text_width, top], fill="red")
            draw.text((left, top - text_height), text, fill="white", font=font)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            output_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            if best_pred.get("class", "").lower() == "frog":
                prediction_label = "Frog Detected"
                confidence_text = f"Confidence: {confidence*100:.2f}%"
            else:
                prediction_label = "No Frog Detected"
                confidence_text = ""
            
            return render_template_string(
                HTML_TEMPLATE,
                prediction=prediction_label,
                confidence_text=confidence_text,
                output_image=output_image
            )
        else:
            logging.info("No predictions returned.")
            return render_template_string(
                HTML_TEMPLATE,
                prediction="No predictions available. Please try again.",
                confidence_text="",
                output_image=""
            )
    except Exception as e:
        logging.error("Error during prediction: %s", traceback.format_exc())
        return "Error processing image", 500

# Helper: Process one image for performance dashboard.
def process_image(img_path, expected_label):
    """
    Run inference on one image and return:
      (is_correct, confidence, status)
    where status is "ok" if processed successfully, or "error".
    """
    try:
        result = CLIENT.infer(img_path, model_id="frog-ukiu5/1")
        preds = result.get("predictions", [])
        if preds:
            best = max(preds, key=lambda p: p.get("confidence", 0))
            pred_label = best.get("class", "").lower()
            confidence = best.get("confidence", 0)
        else:
            pred_label = "nothing"
            confidence = 0
        status = "ok"
    except Exception as ex:
        logging.error("Error processing %s: %s", img_path, ex)
        return None, None, "error"
    if expected_label == "frog":
        is_correct = (pred_label == "frog")
    else:
        is_correct = (pred_label != "frog")
    return is_correct, confidence, status

# Compute performance metrics in parallel.
def compute_performance_parallel(directory, expected_label, max_workers=8):
    error_files = []
    image_files = [f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    total_images = len(image_files)
    if total_images == 0:
        return 0, 0, 0, error_files
    correct = 0
    confidences = []
    processed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, os.path.join(directory, filename), expected_label): filename for filename in image_files}
        for future in as_completed(futures):
            filename = futures[future]
            try:
                is_correct, conf, status = future.result()
                if status == "error":
                    error_files.append(filename)
                else:
                    if is_correct:
                        correct += 1
                    confidences.append(conf)
                    processed += 1
            except Exception as e:
                logging.error("Error processing %s: %s", filename, e)
                error_files.append(filename)
    accuracy = (correct / processed * 100) if processed > 0 else 0
    avg_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0
    return accuracy, avg_confidence, correct, error_files

# Performance Dashboard Route with modern UI, full-screen loading overlay,
# updated chart labels showing classified/total counts, and non-closable errors.
@app.route("/dashboard", methods=["GET"])
def dashboard():
    frog_dir = "frog_images/frog/"
    not_frog_dir = "frog_images/not_frog/"
    
    # Compute performance metrics in parallel.
    frog_accuracy, frog_avg_conf, correct_frog, frog_errors = compute_performance_parallel(frog_dir, expected_label="frog")
    not_frog_accuracy, not_frog_avg_conf, correct_not_frog, not_frog_errors = compute_performance_parallel(not_frog_dir, expected_label="nothing")
    
    # Get image counts.
    frog_count = count_images(frog_dir)
    not_frog_count = count_images(not_frog_dir)
    total_count = frog_count + not_frog_count

    # Build a non-closable error alert.
    errors_html = ""
    if frog_errors or not_frog_errors:
        errors_html = (
            "<div class='alert alert-danger mt-3' role='alert'>"
            "<h5>Error processing the following images:</h5><ul>"
        )
        for fname in frog_errors + not_frog_errors:
            errors_html += f"<li>{fname}</li>"
        errors_html += "</ul></div>"
    
    dashboard_html = f"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <title>Performance Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <!-- Chart.js -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
          body {{
            background: #e9ecef;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
          }}
          .container {{
            margin-top: 40px;
          }}
          /* Full-screen loading overlay style */
          #loading {{
            display: block;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255,255,255,0.9);
            z-index: 9999;
            text-align: center;
            padding-top: 20%;
          }}
          .stats-card {{
            margin-top: 20px;
          }}
        </style>
      </head>
      <body>
        <div id="loading">
          <div class="spinner-border text-primary" role="status">
            <span class="sr-only">Loading...</span>
          </div>
          <h4 class="mt-3">Loading Dashboard...</h4>
        </div>
        <div class="container" id="dashboardContent" style="display:none;">
          <div class="card shadow-sm">
            <div class="card-header text-center bg-primary text-white">
              <h3>Performance Dashboard</h3>
            </div>
            <div class="card-body">
              <div class="row mb-3">
                <div class="col-md-4">
                  <div class="card text-center stats-card">
                    <div class="card-body">
                      <h5 class="card-title">Frog Images</h5>
                      <p class="card-text">{correct_frog} / {frog_count}</p>
                    </div>
                  </div>
                </div>
                <div class="col-md-4">
                  <div class="card text-center stats-card">
                    <div class="card-body">
                      <h5 class="card-title">Not-Frog Images</h5>
                      <p class="card-text">{correct_not_frog} / {not_frog_count}</p>
                    </div>
                  </div>
                </div>
                <div class="col-md-4">
                  <div class="card text-center stats-card">
                    <div class="card-body">
                      <h5 class="card-title">Total Images</h5>
                      <p class="card-text">{total_count}</p>
                    </div>
                  </div>
                </div>
              </div>
              <div class="row">
                <div class="col-md-6">
                  <canvas id="accuracyChart" width="400" height="300"></canvas>
                </div>
                <div class="col-md-6">
                  <canvas id="confidenceChart" width="400" height="300"></canvas>
                </div>
              </div>
              {errors_html}
              <div class="text-center mt-4">
                <a href="/" class="btn btn-primary">Go Back</a>
              </div>
            </div>
          </div>
        </div>
        <script>
          // When window loads, hide loading overlay and show dashboard.
          window.addEventListener("load", function() {{
            document.getElementById("loading").style.display = "none";
            document.getElementById("dashboardContent").style.display = "block";
          }});
          
          // Accuracy Chart with updated labels showing classified/total.
          var ctx = document.getElementById('accuracyChart').getContext('2d');
          var accuracyChart = new Chart(ctx, {{
            type: 'bar',
            data: {{
              labels: [
                `Frog Images ({correct_frog}/{frog_count})`, 
                `Not-Frog Images ({correct_not_frog}/{not_frog_count})`
              ],
              datasets: [{{
                label: 'Accuracy (%)',
                data: [{frog_accuracy:.2f}, {not_frog_accuracy:.2f}],
                backgroundColor: ['rgba(40, 167, 69, 0.2)','rgba(220, 53, 69, 0.2)'],
                borderColor: ['rgba(40, 167, 69, 1)','rgba(220, 53, 69, 1)'],
                borderWidth: 1
              }}]
            }},
            options: {{
              scales: {{
                y: {{
                  beginAtZero: true,
                  max: 100
                }}
              }}
            }}
          }});
          
          // Confidence Chart with updated labels showing classified/total.
          var ctx2 = document.getElementById('confidenceChart').getContext('2d');
          var confidenceChart = new Chart(ctx2, {{
            type: 'bar',
            data: {{
              labels: [
                `Frog Images ({correct_frog}/{frog_count})`, 
                `Not-Frog Images ({correct_not_frog}/{not_frog_count})`
              ],
              datasets: [{{
                label: 'Avg Confidence (%)',
                data: [{frog_avg_conf:.2f}, {not_frog_avg_conf:.2f}],
                backgroundColor: ['rgba(23, 162, 184, 0.2)','rgba(255, 193, 7, 0.2)'],
                borderColor: ['rgba(23, 162, 184, 1)','rgba(255, 193, 7, 1)'],
                borderWidth: 1
              }}]
            }},
            options: {{
              scales: {{
                y: {{
                  beginAtZero: true,
                  max: 100
                }}
              }}
            }}
          }});
        </script>
      </body>
    </html>
    """
    return dashboard_html

if __name__ == "__main__":
    app.run(debug=True)