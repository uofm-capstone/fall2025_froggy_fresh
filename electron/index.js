const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

let mainWindow;

// Create models directory if it doesn't exist
const userModelsDir = path.join(__dirname, '../backend/models');
if (!fs.existsSync(userModelsDir)) {
  fs.mkdirSync(userModelsDir, { recursive: true });
}

app.whenReady().then(() => {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    }
  });

  // Load React app from Vite's dev server or built files
  const devServerURL = 'http://localhost:5173'; // Change this if your Vite dev server uses a different port
  mainWindow.loadURL(devServerURL);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
});

// Handle the folder picker request
ipcMain.handle('open-directory-dialog', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
  });
  return result.filePaths[0] || null; // Return folder path or null if canceled
});

// Handle model file upload
ipcMain.handle('upload-model', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [
      { name: 'Model Files', extensions: ['h5', 'pt', 'onnx'] }
    ]
  });
  
  if (result.canceled || !result.filePaths[0]) {
    return { success: false, message: 'Upload canceled' };
  }
  
  try {
    const modelPath = result.filePaths[0];
    const modelName = path.basename(modelPath);
    const destPath = path.join(__dirname, '../backend/models', modelName);
    
    // Copy the model file to backend/models directory
    fs.copyFileSync(modelPath, destPath);
    
    // Register the model with backend
    const pythonProcess = spawn('python', [
      path.join(__dirname, '../backend/process_images.py'),
      '--register-model',
      destPath,
      '--model-name',
      modelName.split('.')[0] // Use filename without extension as model name
    ]);
    
    return new Promise((resolve) => {
      let stdoutData = '';
      let stderrData = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdoutData += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderrData += data.toString();
      });
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdoutData);
            resolve({ success: true, ...result });
          } catch (e) {
            resolve({ success: true, message: 'Model uploaded' });
          }
        } else {
          resolve({ success: false, error: stderrData || 'Unknown error' });
        }
      });
    });
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Replace the list-models handler with this improved version
ipcMain.handle('list-models', async () => {
  try {
    console.log('Spawning Python process for list-models...');
    const pythonProcess = spawn('python', [
      path.join(__dirname, '../backend/process_images.py'),
      '--list-models'
    ]);
    
    return new Promise((resolve) => {
      let stdoutData = '';
      let stderrData = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdoutData += data.toString();
        console.log('Python stdout:', data.toString());
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderrData += data.toString();
        console.error('Python stderr:', data.toString());
      });
      
      pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        console.log('Full stdout:', stdoutData);
        
        if (code === 0) {
          try {
            const result = JSON.parse(stdoutData);
            console.log('Parsed result:', result);
            resolve(result);
          } catch (e) {
            console.error('JSON parse error:', e);
            // More descriptive error
            resolve({ 
              success: false, 
              error: `Failed to parse models list: ${e.message}`,
              rawData: stdoutData.substring(0, 200) // Include first 200 chars of output
            });
          }
        } else {
          console.error('Python process error:', stderrData);
          resolve({ success: false, error: stderrData || `Process exited with code ${code}` });
        }
      });
    });
  } catch (error) {
    console.error('Spawn error:', error);
    return { success: false, error: `Failed to spawn Python process: ${error.message}` };
  }
});

// Switch active model
ipcMain.handle('switch-model', async (event, modelId) => {
  try {
    const pythonProcess = spawn('python', [
      path.join(__dirname, '../backend/process_images.py'),
      '--set-model',
      modelId
    ]);
    
    return new Promise((resolve) => {
      let stdoutData = '';
      let stderrData = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdoutData += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderrData += data.toString();
      });
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdoutData);
            resolve(result);
          } catch (e) {
            resolve({ success: true, message: `Switched to model: ${modelId}` });
          }
        } else {
          resolve({ success: false, error: stderrData || 'Unknown error' });
        }
      });
    });
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Delete model handler
ipcMain.handle('delete-model', async (event, modelId) => {
  try {
    console.log(`Deleting model: ${modelId}`);
    const pythonProcess = spawn('python', [
      path.join(__dirname, '../backend/process_images.py'),
      '--delete-model',
      modelId
    ]);
    
    return new Promise((resolve) => {
      let stdoutData = '';
      let stderrData = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdoutData += data.toString();
        console.log('Python stdout:', data.toString());
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderrData += data.toString();
        console.error('Python stderr:', data.toString());
      });
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdoutData);
            console.log('Delete result:', result);
            resolve(result);
          } catch (e) {
            console.error('JSON parse error:', e);
            resolve({ 
              success: false, 
              error: `Failed to parse response: ${e.message}`
            });
          }
        } else {
          console.error('Python process error:', stderrData);
          resolve({ 
            success: false, 
            error: stderrData || `Process exited with code ${code}` 
          });
        }
      });
    });
  } catch (error) {
    console.error('Error deleting model:', error);
    return { success: false, error: error.message };
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
