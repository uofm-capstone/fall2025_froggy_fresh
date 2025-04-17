const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const os = require("os");
const path = require("path");
const { spawn } = require("child_process");

const isPackaged = app.isPackaged;

const resourcesFolder = isPackaged ? process.resourcesPath : path.join(__dirname, "..");
const pythonPath =
  os.platform() === "win32"
    ? path.resolve(resourcesFolder, "backend", ".venv", "Scripts", "python.exe")
    : path.resolve(resourcesFolder, "backend", ".venv", "bin", "python");

let mainWindow;

app.whenReady().then(() => {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  // Load React app from Vite's dev server or built files
  if (!isPackaged) {
    const devServerURL = "http://localhost:5173"; // Change this if your Vite dev server uses a different port
    mainWindow.loadURL(devServerURL);
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
});

// Handle the folder picker request
ipcMain.handle("open-directory-dialog", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openDirectory"],
  });
  return result.filePaths[0] || null; // Return folder path or null if canceled
});

let backendProcess;

app.on("ready", () => {
  if (isPackaged) {
    const appScriptPath = path.join(resourcesFolder, "backend", "app.py");
  
    backendProcess = spawn(pythonPath, ['-u', appScriptPath]);
  
    // Handle backend output
    backendProcess.stdout.on('data', (data) => {
      console.log(`Backend: ${data.toString()}`);
    });
  
    backendProcess.stderr.on('data', (error) => {
      console.error(`Backend error: ${error.toString()}`);
    });
  
    backendProcess.on('close', (code) => {
      console.log(`Backend process exited with code ${code}`);
    });
  }
})

ipcMain.on("run-process-images", (event, folderPath) => {
  const processImagesScriptPath = path.resolve(resourcesFolder, "backend", "process_images.py");
  const modelPath = path.resolve(resourcesFolder, "backend", "frog_detector.h5");
  const processImagesProcess = spawn(pythonPath, ["-u", processImagesScriptPath, modelPath, folderPath], {
    cwd: path.resolve(resourcesFolder, "backend"),
  });

  // send stdout to event listener in SortView.tsx
  processImagesProcess.stdout.on("data", (data) => {
    const output = data.toString(); // Convert Buffer to string
    console.log(`got: ${output}`);
    event.sender.send("process-images-output", output); // Send to renderer
  });

  // Capture and send stderr data to the renderer process
  processImagesProcess.stderr.on("data", (error) => {
    const errorMessage = error.toString(); // Convert Buffer to string
    console.error(`stderr: ${errorMessage}`);
    event.sender.send("process-images-error", errorMessage); // Send to renderer
  });

  // Notify renderer process when the Python script finishes
  processImagesProcess.on("close", (code) => {
    console.log(`Python process exited with code ${code}`);
    event.sender.send("process-images-done", code); // Send exit code to renderer
  });
});

app.on('before-quit', () => {
  if (isPackaged) {
    if (backendProcess) {
      backendProcess.kill(); // Clean up the backend process when the app exits
    }
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
