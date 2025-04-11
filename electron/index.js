const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const os = require("os");
const path = require("path");
const { spawn } = require("child_process");

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
  const devServerURL = "http://localhost:5173"; // Change this if your Vite dev server uses a different port
  mainWindow.loadURL(devServerURL);

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

ipcMain.on("run-process-images", (event, folderPath) => {
  const pythonPath =
    os.platform() === "win32"
      // ? path.join(".", ".venv", "Scripts", "python")
      ? path.resolve(__dirname, "..", ".venv", "Scripts", "python")
      : path.resolve(__dirname, "..", ".venv", "bin", "python");
      // : path.join(".", ".venv", "bin", "python");

  const processImagesScriptPath = path.resolve(__dirname, "..", "backend", "process_images.py");
  const processImagesProcess = spawn(pythonPath, ["-u", processImagesScriptPath, folderPath], {
    cwd: path.resolve(__dirname, ".."),
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

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
