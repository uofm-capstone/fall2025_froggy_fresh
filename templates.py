"""HTML templates for Frog Detector application"""

# Main application template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Leapfrog - Frog Detection App</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css">
  <style>
    /* Base styles */
    body {
      padding: 20px;
      transition: background-color 0.3s, color 0.3s;
    }
    
    /* Light mode (default) */
    :root {
      --bg-color: #f8f9fa;
      --text-color: #212529;
      --card-bg: #ffffff;
      --card-border: #dee2e6;
      --btn-primary: #007bff;
      --btn-primary-hover: #0069d9;
      --input-bg: #ffffff;
      --input-border: #ced4da;
    }
    
    /* Dark mode */
    [data-theme="dark"] {
      --bg-color: #222;
      --text-color: #eee;
      --card-bg: #333;
      --card-border: #444;
      --btn-primary: #0069d9;
      --btn-primary-hover: #004ea6;
      --input-bg: #444;
      --input-border: #555;
    }
    
    /* Apply theme variables */
    body {
      background-color: var(--bg-color);
      color: var(--text-color);
    }
    
    .card {
      background-color: var(--card-bg);
      border-color: var(--card-border);
    }
    
    .form-control {
      background-color: var(--input-bg);
      border-color: var(--input-border);
      color: var(--text-color);
    }
    
    .btn-primary {
      background-color: var(--btn-primary);
      border-color: var(--btn-primary);
    }
    
    .btn-primary:hover {
      background-color: var(--btn-primary-hover);
      border-color: var(--btn-primary-hover);
    }
    
    /* Dark mode toggle */
    .theme-switch {
      position: fixed;
      top: 10px;
      right: 10px;
      z-index: 1000;
    }
    
    .theme-toggle-btn {
      width: 40px;
      height: 40px;
      border-radius: 50%;
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 20px;
      box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    /* Result image container */
    .result-image-container {
      max-width: 100%;
      margin: 0 auto;
      border: 1px solid var(--card-border);
      padding: 5px;
      background-color: var(--card-bg);
    }
    
    .result-image {
      max-width: 100%;
      max-height: 600px;
    }
    
    /* Feedback buttons */
    .feedback-buttons {
      margin-top: 15px;
      padding: 10px;
      background-color: var(--card-bg);
      border-radius: 4px;
    }
    
    /* Dashboard link */
    .dashboard-link, .training-link {
      margin-top: 20px;
    }
  </style>
</head>
<body>
  <!-- Dark mode toggle button with moon icon -->
  <div class="theme-switch">
    <button id="theme-toggle" class="theme-toggle-btn">
      <span id="theme-icon">🌙</span>
    </button>
  </div>
  
  <div class="container">
    <div class="card mx-auto">
      <div class="card-header">
        <h2 class="text-center">Frog Detector</h2>
      </div>
      <div class="card-body">
        <form action="/predict" method="post" enctype="multipart/form-data">
          <!-- Model selection dropdown -->
          <div class="form-group model-selector">
            <label for="model">Select Model:</label>
            <select id="model-selector" name="model" class="form-control">
              {% for model in model_options %}
                <option value="{{ model }}" {% if model == selected_model %}selected{% endif %}>{{ model }}</option>
              {% endfor %}
            </select>
          </div>
          
          <div class="form-group">
            <label for="file">Upload an image:</label>
            <input type="file" class="form-control" id="file" name="file" accept="image/*">
          </div>
          <button type="submit" class="btn btn-success upload-btn">Detect Frogs</button>
        </form>
        
        {% if output_image %}
          <div class="text-center mt-4">
            <p class="prediction-text">{{ prediction }}</p>
            <p class="confidence-text">{{ confidence_text }}</p>
            <div class="result-image-container">
              <img src="{{ output_image }}" class="result-image" alt="Result">
            </div>
            
            <!-- Feedback buttons to improve the model -->
            <div class="feedback-buttons mt-3">
              <p class="mb-2">Was this prediction correct?</p>
              <button type="button" class="btn btn-sm btn-success mr-2" onclick="provideFeedback('frog')">It's a Frog</button>
              <button type="button" class="btn btn-sm btn-danger" onclick="provideFeedback('not_frog')">Not a Frog</button>
            </div>
          </div>
          {% if saved_path %}
          <input type="hidden" id="imagePath" value="{{ saved_path }}">
          {% endif %}
          <script>
            function provideFeedback(actualClass) {
              const imagePath = document.getElementById('imagePath').value;
              
              fetch('/feedback', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                  image_path: imagePath,
                  actual_class: actualClass
                })
              })
              .then(response => response.json())
              .then(data => {
                if (data.success) {
                  alert('Thank you for your feedback! The dashboard has been updated.');
                  // Optional: redirect to dashboard
                  // window.location.href = '/dashboard';
                }
              })
              .catch(error => {
                console.error('Error:', error);
                alert('There was an error submitting your feedback.');
              });
            }
          </script>
        {% else %}
          <div class="text-center mt-4">
            <p>{{ prediction }}</p>
          </div>
        {% endif %}
      </div>
    </div>
    
    <div class="text-center dashboard-link">
      <a href="/dashboard" class="btn btn-outline-primary">View Model Performance Dashboard</a>
    </div>
    
    <div class="text-center mt-3 training-link">
      <a href="/training" class="btn btn-outline-secondary">Model Training Options</a>
    </div>
  </div>
  
  <!-- Theme switcher JavaScript -->
  <script>
    // Theme switcher
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const html = document.documentElement;
    
    // Check for saved theme preference or prefer-color-scheme
    const savedTheme = localStorage.getItem('theme') || 
        (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    
    // Apply saved theme on page load
    if (savedTheme === 'dark') {
        html.setAttribute('data-theme', 'dark');
        themeIcon.innerText = '☀️';
    }
    
    // Handle theme toggle click
    themeToggle.addEventListener('click', () => {
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        themeIcon.innerText = newTheme === 'dark' ? '☀️' : '🌙';
    });
  </script>
</body>
</html>
'''

# Dashboard template
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Model Performance Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    /* Light mode (default) */
    :root {
      --bg-color: #f8f9fa;
      --text-color: #212529;
      --card-bg: #ffffff;
      --card-border: #dee2e6;
      --card-header: #0d6efd;
      --btn-primary: #0d6efd;
    }
    
    /* Dark mode */
    [data-theme="dark"] {
      --bg-color: #222;
      --text-color: #eee;
      --card-bg: #333;
      --card-border: #444;
      --card-header: #0a58ca;
      --btn-primary: #0a58ca;
    }
    
    /* Apply theme variables */
    body {
      background-color: var(--bg-color);
      color: var(--text-color);
      padding-top: 20px;
      transition: background-color 0.3s, color 0.3s;
    }
    
    .card {
      margin-bottom: 20px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
      background-color: var(--card-bg);
      border-color: var(--card-border);
    }
    
    .card-header {
      background-color: var(--card-header);
      color: white;
      font-weight: bold;
    }
    
    .stat-card {
      text-align: center;
      padding: 15px;
    }
    
    .stat-value {
      font-size: 32px;
      font-weight: bold;
    }
    
    .stat-label {
      font-size: 14px;
      color: #6c757d;
      text-transform: uppercase;
    }
    
    .back-link {
      margin-top: 20px;
    }
    
    .recent-image {
      width: 100px;
      height: 100px;
      object-fit: cover;
      border-radius: 4px;
      margin: 5px;
      border: 2px solid #ddd;
    }
    
    .correct {
      border-color: #198754;
    }
    
    .incorrect {
      border-color: #dc3545;
    }
    
    .refresh-btn {
      margin-left: 10px;
    }
    
    .form-control {
      background-color: var(--card-bg);
      color: var(--text-color);
      border-color: var(--card-border);
    }
    
    .theme-switch {
      position: fixed;
      top: 10px;
      right: 10px;
      z-index: 1000;
    }
    
    .theme-toggle-btn {
      width: 40px;
      height: 40px;
      border-radius: 50%;
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 20px;
      box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
  </style>
</head>
<body>
  <!-- Dark mode toggle button with moon icon -->
  <div class="theme-switch">
    <button id="theme-toggle" class="theme-toggle-btn">
      <span id="theme-icon">🌙</span>
    </button>
  </div>

  <div class="container">
    <div class="card">
      <div class="card-header">
        <div class="d-flex justify-content-between align-items-center">
          <h2>Model Performance Dashboard</h2>
          <div class="d-flex">
            <div class="model-selector">
              <form action="/dashboard" method="get" class="d-flex" id="modelForm">
                <select class="form-control me-2" id="model" name="model" onchange="this.form.submit()">
                  {% for model in available_models %}
                    <option value="{{ model }}" {% if model == selected_model %}selected{% endif %}>
                      {{ model }}
                    </option>
                  {% endfor %}
                </select>
              </form>
            </div>
            <button class="btn btn-outline-light refresh-btn" onclick="refreshDashboard()">
              <i class="bi bi-arrow-clockwise"></i> Refresh
            </button>
          </div>
        </div>
      </div>
      <div class="card-body">
        <div class="row">
          <div class="col-md-3">
            <div class="card stat-card">
              <div class="stat-value">{{ accuracy }}{{ "%" if accuracy != "N/A" else "" }}</div>
              <div class="stat-label">Overall Accuracy</div>
            </div>
          </div>
          <div class="col-md-3">
            <div class="card stat-card">
              <div class="stat-value">{{ true_positive_rate }}{{ "%" if true_positive_rate != "N/A" else "" }}</div>
              <div class="stat-label">Frog Detection Rate</div>
            </div>
          </div>
          <div class="col-md-3">
            <div class="card stat-card">
              <div class="stat-value">{{ true_negative_rate }}{{ "%" if true_negative_rate != "N/A" else "" }}</div>
              <div class="stat-label">Non-Frog Detection Rate</div>
            </div>
          </div>
          <div class="col-md-3">
            <div class="card stat-card">
              <div class="stat-value">{{ avg_confidence }}{{ "%" if avg_confidence != "N/A" else "" }}</div>
              <div class="stat-label">Avg. Confidence</div>
            </div>
          </div>
        </div>
        
        <div class="row mt-4">
          <div class="col-md-6">
            <div class="card">
              <div class="card-header">Frog Test Images ({{ total_frog }})</div>
              <div class="card-body">
                <table class="table table-striped">
                  <thead>
                    <tr>
                      <th>Result</th>
                      <th>Count</th>
                      <th>Percentage</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr class="table-success">
                      <td>Correctly identified</td>
                      <td>{{ correct_frog }}</td>
                      <td>{{ frog_accuracy }}{{ "%" if frog_accuracy != "N/A" else "" }}</td>
                    </tr>
                    <tr class="table-danger">
                      <td>Missed</td>
                      <td>{{ missed_frog }}</td>
                      <td>{{ (100 - frog_accuracy) if frog_accuracy != "N/A" else "N/A" }}{{ "%" if frog_accuracy != "N/A" else "" }}</td>
                    </tr>
                  </tbody>
                </table>
                
                <!-- Recent frog images -->
                <div class="mt-3">
                  <h6>Recent Frog Images:</h6>
                  <div class="d-flex flex-wrap">
                    {% for img in recent_frog_images %}
                      <img src="{{ img.url }}" class="recent-image {{ 'correct' if img.correct else 'incorrect' }}" 
                           title="{{ 'Correctly identified' if img.correct else 'Incorrectly identified' }}">
                    {% endfor %}
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div class="col-md-6">
            <div class="card">
              <div class="card-header">Non-Frog Test Images ({{ total_not_frog }})</div>
              <div class="card-body">
                <table class="table table-striped">
                  <thead>
                    <tr>
                      <th>Result</th>
                      <th>Count</th>
                      <th>Percentage</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr class="table-success">
                      <td>Correctly identified</td>
                      <td>{{ correct_not_frog }}</td>
                      <td>{{ not_frog_accuracy }}{{ "%" if not_frog_accuracy != "N/A" else "" }}</td>
                    </tr>
                    <tr class="table-danger">
                      <td>False positives</td>
                      <td>{{ false_positive }}</td>
                      <td>{{ (100 - not_frog_accuracy) if not_frog_accuracy != "N/A" else "N/A" }}{{ "%" if not_frog_accuracy != "N/A" else "" }}</td>
                    </tr>
                  </tbody>
                </table>
                    
                <!-- Recent non-frog images -->
                <div class="mt-3">
                  <h6>Recent Non-Frog Images:</h6>
                  <div class="d-flex flex-wrap">
                    {% for img in recent_not_frog_images %}
                      <img src="{{ img.url }}" class="recent-image {{ 'correct' if img.correct else 'incorrect' }}" 
                           title="{{ 'Correctly identified' if img.correct else 'Incorrectly identified' }}">
                    {% endfor %}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="card mb-3">
      <div class="card-header bg-info text-white">
        <h4>Model Evaluation Stats</h4>
      </div>
      <div class="card-body">
        <div class="alert alert-info">
          <strong>Images in Test Set:</strong> {{ total_frog }} frogs, {{ total_not_frog }} non-frogs
          <br>
          <small>These images are used for evaluation only, not for training.</small>
        </div>
      </div>
    </div>
    
    <div class="card mb-3">
      <div class="card-header bg-info text-white">
        <h4>Dataset Information</h4>
      </div>
      <div class="card-body">
        <div class="row">
          <div class="col-md-6">
            <div class="card mb-2">
              <div class="card-header bg-success text-white">Training Images</div>
              <div class="card-body">
                <p><strong>Total:</strong> {{ train_total }} images</p>
                <ul>
                  <li>Frogs: {{ train_frog_count }}</li>
                  <li>Not-Frogs: {{ train_not_frog_count }}</li>
                </ul>
                <small class="text-muted">Used for training models</small>
              </div>
            </div>
          </div>
          <div class="col-md-6">
            <div class="card mb-2">
              <div class="card-header bg-warning text-dark">Test Images</div>
              <div class="card-body">
                <p><strong>Total:</strong> {{ total_frog + total_not_frog }} images</p>
                <ul>
                  <li>Frogs: {{ total_frog }}</li>
                  <li>Not-Frogs: {{ total_not_frog }}</li>
                </ul>
                <small class="text-muted">Used for evaluation only</small>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="row">
      <div class="col-md-12">
        <div class="card">
          <div class="card-header">
            <h3>Model Improvement Options</h3>
          </div>
          <div class="card-body">
            <div class="row">
              <div class="col-md-6">
                <div class="card">
                  <div class="card-body">
                    <h5 class="card-title">Train with Current Dataset</h5>
                    <p class="card-text">Use the feedback-based dataset to improve this model.</p>
                    <a href="/train-model?model={{ selected_model }}" class="btn btn-primary">Start Training</a>
                  </div>
                </div>
              </div>
              <div class="col-md-6">
                <div class="card">
                  <div class="card-body">
                    <h5 class="card-title">Optimize Model Performance</h5>
                    <p class="card-text">Apply techniques to speed up and improve the current model.</p>
                    <a href="/optimize-model?model={{ selected_model }}" class="btn btn-success">Optimize Model</a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="card mb-4">
      <div class="card-header bg-primary text-white">
          <h3>Model Evaluation</h3>
      </div>
      <div class="card-body">
          {% if request.args.get('error') %}
          <div class="alert alert-danger">
              {{ request.args.get('error') }}
          </div>
          {% endif %}
          
          {% if request.args.get('evaluated') %}
          <div class="alert alert-success">
              <strong>Evaluation Complete!</strong> Results from testing on {{ total_frog + total_not_frog }} images:
              <ul>
                  <li>Overall Accuracy: {{ accuracy }}%</li>
                  <li>Frog Detection Accuracy: {{ frog_accuracy }}%</li>
                  <li>Non-Frog Detection Accuracy: {{ not_frog_accuracy }}%</li>
              </ul>
          </div>
          {% endif %}
          
          <p>Evaluate the selected model's performance on test images:</p>
          
          <form action="/evaluate-model" method="post">
              <input type="hidden" name="model_id" value="{{ selected_model }}">
              <button type="submit" class="btn btn-primary">
                  <i class="fas fa-chart-bar"></i> Evaluate Model on Test Data
              </button>
          </form>
          
          <p class="mt-3 text-muted">
              <small>This will run the model on all images in the test directory 
              and compute accuracy metrics.</small>
          </p>
      </div>
    </div>
    
    <div class="text-center back-link">
      <a href="/" class="btn btn-outline-primary">Back to Detector</a>
    </div>
  </div>
  
  <script>
    // Theme switcher
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const html = document.documentElement;
    
    // Check for saved theme preference or prefer-color-scheme
    const savedTheme = localStorage.getItem('theme') || 
        (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    
    // Apply saved theme on page load
    if (savedTheme === 'dark') {
        html.setAttribute('data-theme', 'dark');
        themeIcon.innerText = '☀️';
    }
    
    // Handle theme toggle click
    themeToggle.addEventListener('click', () => {
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        themeIcon.innerText = newTheme === 'dark' ? '☀️' : '🌙';
    });
    
    function refreshDashboard() {
      // Clear cache and reload
      fetch('/clear-dashboard-cache', {
        method: 'POST'
      }).then(() => {
        location.reload();
      });
    }
  </script>
</body>
</html>
'''

# Training template
TRAINING_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Model Training Options</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    /* Light mode (default) */
    :root {
      --bg-color: #f8f9fa;
      --text-color: #212529;
      --card-bg: #ffffff;
      --card-border: #dee2e6;
      --card-header: #6610f2;
    }
    
    /* Dark mode */
    [data-theme="dark"] {
      --bg-color: #222;
      --text-color: #eee;
      --card-bg: #333;
      --card-border: #444;
      --card-header: #5a0bc2;
    }
    
    /* Apply theme variables */
    body {
      background-color: var(--bg-color);
      color: var(--text-color);
      padding-top: 20px;
      transition: background-color 0.3s, color 0.3s;
    }
    
    .card {
      margin-bottom: 20px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
      background-color: var(--card-bg);
      border-color: var(--card-border);
    }
    
    .card-header {
      background-color: var(--card-header);
      color: white;
      font-weight: bold;
    }
    
    .form-control {
      background-color: var(--card-bg);
      color: var(--text-color);
      border-color: var(--card-border);
    }
    
    .back-link {
      margin-top: 20px;
    }
    
    .theme-switch {
      position: fixed;
      top: 10px;
      right: 10px;
      z-index: 1000;
    }
    
    .theme-toggle-btn {
      width: 40px;
      height: 40px;
      border-radius: 50%;
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 20px;
      box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
  </style>
</head>
<body>
  <!-- Dark mode toggle button with moon icon -->
  <div class="theme-switch">
    <button id="theme-toggle" class="theme-toggle-btn">
      <span id="theme-icon">🌙</span>
    </button>
  </div>

  <div class="container">
    <div class="card">
      <div class="card-header">
        <h2>Model Training Options</h2>
      </div>
      <div class="card-body">
        <div class="row">
          <div class="col-md-6">
            <div class="card">
              <div class="card-header">Train Roboflow Model</div>
              <div class="card-body">
                <p>Fine-tune the Roboflow model with your images.</p>
                <p>Images in training directories: {{ frog_count }} frogs, {{ not_frog_count }} non-frogs</p>
                <form action="/train-roboflow" method="post">
                  <div class="mb-3">
                    <label for="epochs" class="form-label">Training Epochs</label>
                    <input type="number" class="form-control" id="epochs" name="epochs" value="10" min="1" max="100">
                  </div>
                  <button type="submit" class="btn btn-primary">Start Training</button>
                </form>
              </div>
            </div>
          </div>
          
          <div class="col-md-6">
            <div class="card">
              <div class="card-header">Train Local YOLO Model</div>
              <div class="card-body">
                <p>Train a YOLO model on your local machine.</p>
                <form action="/train-yolo" method="post">
                  <div class="mb-3">
                    <label for="yolo_epochs" class="form-label">Training Epochs</label>
                    <input type="number" class="form-control" id="yolo_epochs" name="epochs" value="50" min="1" max="500">
                  </div>
                  <button type="submit" class="btn btn-success">Start Training</button>
                </form>
              </div>
            </div>
          </div>
        </div>
        
        <div class="row mt-4">
          <div class="col-md-6">
            <div class="card">
              <div class="card-header">Train TensorFlow Model</div>
              <div class="card-body">
                <p>Train a TensorFlow classification model.</p>
                <form action="/train-tensorflow" method="post">
                  <div class="mb-3">
                    <label for="tf_epochs" class="form-label">Training Epochs</label>
                    <input type="number" class="form-control" id="tf_epochs" name="epochs" value="20" min="1" max="200">
                  </div>
                  <button type="submit" class="btn btn-info text-white">Start Training</button>
                </form>
              </div>
            </div>
          </div>
          
          <div class="col-md-6">
            <div class="card">
              <div class="card-header">Model Optimization</div>
              <div class="card-body">
                <p>Optimize existing models for better performance.</p>
                <form action="/optimize-model" method="post">
                  <div class="mb-3">
                    <label for="model_to_optimize" class="form-label">Select Model</label>
                    <select class="form-control" id="model_to_optimize" name="model_id">
                      {% for model in available_models %}
                        <option value="{{ model }}">{{ model }}</option>
                      {% endfor %}
                    </select>
                  </div>
                  <button type="submit" class="btn btn-warning text-dark">Optimize Model</button>
                </form>
              </div>
            </div>
          </div>
        </div>
        
        <div class="card mb-4">
          <div class="card-header bg-info text-white">
              <h3>Data Organization</h3>
          </div>
          <div class="card-body">
              {% if request.args.get('message') %}
              <div class="alert alert-success">{{ request.args.get('message') }}</div>
              {% endif %}
              
              {% if request.args.get('error') %}
              <div class="alert alert-danger">{{ request.args.get('error') }}</div>
              {% endif %}

              <p>A train/test split helps evaluate your model's performance on unseen data.</p>
              <p><strong>Important:</strong> Your source directory must have "frog" and "not_frog" subdirectories.</p>
              
              <form action="/create-split" method="post">
                  <div class="form-group mb-3">
                      <label for="source_dir">Source Images Directory:</label>
                      <input type="text" name="source_dir" id="source_dir" class="form-control" 
                             value="frog_images" placeholder="Directory with frog/not_frog subdirectories">
                      <small class="form-text text-muted">Default: frog_images</small>
                  </div>
                    
                  <div class="form-group mb-3">
                      <label for="split_ratio">Train/Test Split Ratio:</label>
                      <select name="split_ratio" id="split_ratio" class="form-control">
                          <option value="0.8">80% Train / 20% Test</option>
                          <option value="0.7">70% Train / 30% Test</option>
                          <option value="0.75">75% Train / 25% Test</option>
                          <option value="0.9">90% Train / 10% Test</option>
                      </select>
                  </div>
                  <div class="alert alert-warning">
                      <strong>Warning:</strong> This will move images from the source folders to train/test directories!
                  </div>
                  <button type="submit" class="btn btn-primary">Create Train/Test Split</button>
              </form>
          </div>
        </div>
      </div>
    </div>
    
    <div class="text-center back-link">
      <a href="/" class="btn btn-outline-primary">Back to Detector</a>
      <a href="/dashboard" class="btn btn-outline-secondary ms-2">View Dashboard</a>
    </div>
  </div>
  
  <script>
    // Theme switcher
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const html = document.documentElement;
    
    // Check for saved theme preference or prefer-color-scheme
    const savedTheme = localStorage.getItem('theme') || 
        (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    
    // Apply saved theme on page load
    if (savedTheme === 'dark') {
        html.setAttribute('data-theme', 'dark');
        themeIcon.innerText = '☀️';
    }
    
    // Handle theme toggle click
    themeToggle.addEventListener('click', () => {
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        themeIcon.innerText = newTheme === 'dark' ? '☀️' : '🌙';
    });
  </script>
</body>
</html>
'''