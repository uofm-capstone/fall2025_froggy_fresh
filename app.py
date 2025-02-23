import os
import logging
import warnings
from flask import Flask, request, render_template_string
from dotenv import load_dotenv

# Load environment variables from .env file.
load_dotenv()

# Suppress extra logs and warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Import the Roboflow InferenceHTTPClient.
from inference_sdk import InferenceHTTPClient

# Initialize the Roboflow client with API key from environment.
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", 
    api_key=ROBOFLOW_API_KEY  # API key loaded from .env file.
)

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Frog Identifier</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap CSS (v4.5) -->
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
  <style>
    body {
      background: #f1f5f9;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }
    .container {
      margin-top: 50px;
    }
    .card {
      border: none;
      border-radius: 1rem;
      box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .card-header {
      background-color: #4a90e2;
      color: white;
      border-top-left-radius: 1rem;
      border-top-right-radius: 1rem;
      text-align: center;
    }
    .card-body {
      padding: 2rem;
    }
    .custom-file-label::after {
      content: "Browse";
    }
    .result-img {
      max-width: 100%;
      height: auto;
      border-radius: 0.5rem;
      margin-top: 15px;
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
      <a href="/metrics" class="btn btn-secondary">View Metrics</a>
    </div>
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
  </script>
</body>
</html>
'''

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, prediction="Frog Identifier", confidence_text="", output_image="")