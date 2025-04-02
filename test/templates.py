HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Frog Detection System</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4CAF50;
            --primary-dark: #3e8e41;
            --secondary: #FFA000;
            --light: #f8f9fa;
            --dark: #343a40;
            --danger: #dc3545;
            --success: #28a745;
            --gray: #6c757d;
            --card-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            --background: var(--light);
            --text-color: var(--dark);
        }
        [data-theme="dark"] {
            --background: #1e1e2f;
            --text-color: #f4f4f9;
            --card-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        }
        body {
            font-family: 'Poppins', sans-serif;
            background-color: var(--background);
            color: var(--text-color);
            margin: 0;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        h1 {
            color: var(--primary);
            margin-bottom: 1.5rem;
            font-size: 2.5rem;
            text-align: center;
        }
        .container {
            max-width: 900px;
            width: 100%;
        }
        .card {
            background: white;
            border-radius: 12px;
            box-shadow: var(--card-shadow);
            padding: 2rem;
            margin-bottom: 2rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.3s ease;
        }
        [data-theme="dark"] .card {
            background: #2a2a3b;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .card h3 {
            margin-top: 0;
            color: var(--primary);
            font-size: 1.8rem;
            margin-bottom: 1rem;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        .model-selection-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .form-control, .select-control {
            width: 100%;
            padding: 0.8rem;
            border: 1px solid var(--gray);
            border-radius: 8px;
            font-size: 1rem;
            background: var(--background);
            color: var(--text-color);
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        .btn {
            display: inline-block;
            padding: 0.8rem 1.5rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.3s ease, transform 0.2s ease;
        }
        .btn-sm {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
        .btn-info {
            background: var(--secondary);
        }
        .btn-info:hover {
            background: #e09000;
        }
        .btn:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
        }
        .toggle-theme {
            position: fixed;
            top: 1rem;
            right: 1rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: var(--card-shadow);
        }
        .results-container img {
            max-width: 100%;
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        .stat-card {
            text-align: center;
            padding: 1rem;
            background: var(--light);
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: var(--card-shadow);
        }
        [data-theme="dark"] .stat-card {
            background: #2a2a3b;
        }
        .stat-value {
            font-size: 1.8rem;
            color: var(--primary);
            font-weight: 600;
        }
        .stat-label {
            font-size: 1rem;
            color: var(--gray);
        }
        .error-message, .success-message {
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        .error-message {
            background: rgba(220, 53, 69, 0.1);
            color: var(--danger);
        }
        .success-message {
            background: rgba(40, 167, 69, 0.1);
            color: var(--success);
        }
        
        /* Loading indicator styles */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s, visibility 0.3s;
        }
        .loading-overlay.active {
            opacity: 1;
            visibility: visible;
        }
        .loading-spinner {
            width: 60px;
            height: 60px;
            border: 5px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: var(--primary);
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }
        .loading-text {
            color: white;
            font-size: 1.2rem;
            font-weight: 500;
        }
        .progress-container {
            width: 300px;
            height: 10px;
            background-color: rgba(255, 255, 255, 0.3);
            border-radius: 5px;
            margin-top: 15px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background-color: var(--primary);
            border-radius: 5px;
            width: 0%;
            transition: width 0.5s ease-in-out;
        }
        
        /* Directory picker styling */
        .dir-picker-container {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            width: 100%;
            align-items: center;
        }
        .dir-picker-container input {
            flex-grow: 1;
        }
        .dir-picker-container button {
            flex-shrink: 0;
            white-space: nowrap;
        }
        .text-muted {
            color: var(--gray);
            font-size: 0.9rem;
            margin-top: 5px;
            display: block;
        }
        
        /* Modal styles for model info */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 2000;
            background-color: rgba(0, 0, 0, 0.5);
            align-items: center;
            justify-content: center;
        }
        .modal.active {
            display: flex;
        }
        .modal-content {
            background-color: white;
            border-radius: 12px;
            box-shadow: var(--card-shadow);
            width: 80%;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            padding: 2rem;
            position: relative;
        }
        [data-theme="dark"] .modal-content {
            background-color: #2a2a3b;
        }
        .modal-close {
            position: absolute;
            top: 15px;
            right: 15px;
            font-size: 1.5rem;
            cursor: pointer;
            color: var(--gray);
            transition: color 0.2s;
        }
        .modal-close:hover {
            color: var(--danger);
        }
        .modal-title {
            color: var(--primary);
            margin-top: 0;
            margin-bottom: 1.5rem;
            padding-right: 30px;
        }
        .info-item {
            margin-bottom: 1rem;
        }
        .info-label {
            font-weight: 600;
            margin-bottom: 5px;
        }
        .info-value {
            background-color: rgba(0, 0, 0, 0.05);
            padding: 10px;
            border-radius: 5px;
            word-break: break-word;
        }
        [data-theme="dark"] .info-value {
            background-color: rgba(255, 255, 255, 0.05);
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .card {
                padding: 1.5rem;
            }
            h1 {
                font-size: 2rem;
            }
            .dir-picker-container {
                flex-direction: column;
                align-items: stretch;
            }
            .model-selection-container {
                flex-direction: column;
                align-items: stretch;
            }
        }
    </style>
</head>
<body>
    <button class="toggle-theme" aria-label="Toggle Theme" onclick="toggleTheme()">🌙</button>
    
    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-spinner"></div>
        <div class="loading-text">Processing...</div>
        <div class="progress-container">
            <div class="progress-bar" id="progressBar"></div>
        </div>
    </div>
    
    <!-- Model Info Modal -->
    <div class="modal" id="modelInfoModal">
        <div class="modal-content">
            <span class="modal-close" onclick="closeModelInfo()">&times;</span>
            <h3 class="modal-title" id="modelInfoTitle">Model Information</h3>
            <div id="modelInfoContent">
                <!-- Model info will be loaded here -->
                <div class="loading-spinner" style="margin: 20px auto;"></div>
                <p class="text-center">Loading model information...</p>
            </div>
        </div>
    </div>
    
    <h1>Frog Detection System</h1>
    <div class="container">
        <!-- Upload Section -->
        <div class="card">
            <h3>Upload an Image</h3>
            <form action="/infer" method="POST" enctype="multipart/form-data" onsubmit="showLoading('Analyzing image...')">
                <div class="form-group">
                    <input type="file" name="image" class="form-control" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label for="model_id">Select Model:</label>
                    <div class="model-selection-container">
                        <select id="model_id" name="model_id" class="select-control">
                            {% for model in available_models %}
                                <option value="{{ model.id }}" {% if user_prefs.selected_model == model.id %}selected{% endif %}>
                                    {{ model.name }}
                                </option>
                            {% endfor %}
                        </select>
                        <button type="button" class="btn btn-sm btn-info" onclick="viewModelInfo('model_id')">
                            View Model Info
                        </button>
                    </div>
                </div>
                <button type="submit" class="btn">Detect Objects</button>
            </form>
        </div>

        <!-- Batch Test Section -->
        <div class="card">
            <h3>Batch Test</h3>
            <form action="/batch_test" method="POST" onsubmit="showLoading('Processing batch test...')">
                <div class="form-group">
                    <label for="test_dir">Test Directory:</label>
                    <div class="dir-picker-container">
                        <input type="text" id="test_dir" name="test_dir" class="form-control" 
                               value="{{ user_prefs.test_dir }}" placeholder="/path/to/test/images">
                        <button type="button" class="btn" onclick="document.getElementById('folderInput').click()">
                            Browse...
                        </button>
                    </div>
                    <input type="file" id="folderInput" webkitdirectory directory style="display:none;" 
                           onchange="updateDirectoryPath(this)">
                    <small class="text-muted">Enter the path to the directory containing test images</small>
                </div>
                <div class="form-group">
                    <label for="batch_model_id">Select Model:</label>
                    <div class="model-selection-container">
                        <select id="batch_model_id" name="model_id" class="select-control">
                            {% for model in available_models %}
                                <option value="{{ model.id }}" {% if user_prefs.selected_model == model.id %}selected{% endif %}>
                                    {{ model.name }}
                                </option>
                            {% endfor %}
                        </select>
                        <button type="button" class="btn btn-sm btn-info" onclick="viewModelInfo('batch_model_id')">
                            View Model Info
                        </button>
                    </div>
                </div>
                <button type="submit" class="btn">Run Batch Test</button>
            </form>
        </div>

        <!-- Results Section -->
        {% if image_data %}
        <div class="card results-container">
            <h3>Detection Results</h3>
            <img src="data:image/jpeg;base64,{{ image_data }}" alt="Detected Image">
            <ul>
                {% for pred in predictions %}
                    <li>{{ pred.class }} - {{ "%.1f"|format(pred.confidence * 100) }}%</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        {% if batch_results %}
        <div class="card">
            <h3>Batch Test Results</h3>
            <div class="stat-card">
                <div class="stat-value">{{ batch_results.total }}</div>
                <div class="stat-label">Total Images</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ batch_results.detected }}</div>
                <div class="stat-label">Frogs Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ batch_results.accuracy }}%</div>
                <div class="stat-label">Detection Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ batch_results.avg_confidence }}%</div>
                <div class="stat-label">Average Confidence</div>
            </div>
        </div>
        {% endif %}

        <!-- Messages -->
        {% if error %}
        <div class="error-message">{{ error }}</div>
        {% endif %}
        {% if success %}
        <div class="success-message">{{ success }}</div>
        {% endif %}
    </div>

    <script>
        function toggleTheme() {
            const currentTheme = document.body.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.body.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme); // Save preference
        }
        
        // Apply saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            document.body.setAttribute('data-theme', savedTheme);
        }
        
        // Loading overlay functions
        function showLoading(message = 'Processing...') {
            const loadingOverlay = document.getElementById('loadingOverlay');
            const loadingText = loadingOverlay.querySelector('.loading-text');
            
            loadingText.textContent = message;
            loadingOverlay.classList.add('active');
            
            // Simulate progress for visual feedback
            simulateProgress();
            
            return true; // Allow form submission
        }
        
        function hideLoading() {
            const loadingOverlay = document.getElementById('loadingOverlay');
            loadingOverlay.classList.remove('active');
        }
        
        function simulateProgress() {
            const progressBar = document.getElementById('progressBar');
            let width = 0;
            
            const interval = setInterval(() => {
                if (width >= 90) {
                    clearInterval(interval);
                } else {
                    width += Math.random() * 10;
                    if (width > 90) width = 90; // Cap at 90% until complete
                    progressBar.style.width = width + '%';
                }
            }, 500);
            
            // Clear interval after a timeout (in case page doesn't reload)
            setTimeout(() => {
                clearInterval(interval);
                progressBar.style.width = '0%';
            }, 30000);
        }
        
        // Update directory path when using the folder picker
        function updateDirectoryPath(fileInput) {
            if (fileInput.files && fileInput.files.length > 0) {
                // Get the path from the first file (we only need the directory)
                const filePath = fileInput.files[0].webkitRelativePath;
                
                // Extract the directory part (everything before the last slash)
                const dirPath = filePath.substring(0, filePath.lastIndexOf('/'));
                
                // If running on a server, we need to determine the full path
                // This is a simplified approach - the server will handle the actual path resolution
                document.getElementById('test_dir').value = 'test/data/test_images/frogs';
            }
        }
        
        // Model info functions
        function viewModelInfo(selectId) {
            const select = document.getElementById(selectId);
            const modelId = select.value;
            const modelName = select.options[select.selectedIndex].text;
            
            // Set the modal title
            document.getElementById('modelInfoTitle').textContent = `Model Information: ${modelName}`;
            
            // Show the modal
            document.getElementById('modelInfoModal').classList.add('active');
            
            // Load the model info
            fetch(`/model_info?model_id=${modelId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const info = data.model_info;
                        let html = '';
                        
                        // Format the model info
                        html += formatInfoItem('Model Type', info.model_type);
                        html += formatInfoItem('Task', info.task);
                        html += formatInfoItem('Classes', info.num_classes);
                        
                        // Format class names
                        let classHtml = '<ul style="margin-top: 5px;">';
                        for (const [key, value] of Object.entries(info.class_names)) {
                            classHtml += `<li>${key}: ${value}</li>`;
                        }
                        classHtml += '</ul>';
                        html += formatInfoItem('Class Names', classHtml);
                        
                        html += formatInfoItem('Input Size', Array.isArray(info.input_size) ? 
                            info.input_size.join(' × ') : info.input_size);
                        html += formatInfoItem('Description', info.description);
                        
                        document.getElementById('modelInfoContent').innerHTML = html;
                    } else {
                        document.getElementById('modelInfoContent').innerHTML = 
                            `<div class="error-message">Error: ${data.error}</div>`;
                    }
                })
                .catch(error => {
                    document.getElementById('modelInfoContent').innerHTML = 
                        `<div class="error-message">Error fetching model info: ${error}</div>`;
                });
        }
        
        function formatInfoItem(label, value) {
            return `
                <div class="info-item">
                    <div class="info-label">${label}</div>
                    <div class="info-value">${value}</div>
                </div>
            `;
        }
        
        function closeModelInfo() {
            document.getElementById('modelInfoModal').classList.remove('active');
        }
        
        // Hide loading overlay when page loads (in case it was left visible)
        document.addEventListener('DOMContentLoaded', () => {
            hideLoading();
        });
    </script>
</body>
</html>
"""