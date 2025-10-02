// You must run `npm run build` to compile this into `electron/main.js`
// before running `npm run dev` or `npm run package`

import { IpcMainEvent, IpcMainInvokeEvent } from "electron";
import { app, BrowserWindow, ipcMain, dialog, shell } from "electron";
import * as os from "os";
import * as path from "path";
import { promises as fs } from "fs";
import { spawn } from "child_process";

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

ipcMain.on("open-folder", (event: IpcMainEvent, path: string, showInFolder: boolean) => {
  if (showInFolder) {
    shell.showItemInFolder(path);
  } else {
    shell.openPath(path);
  }
})

function convertRunDataToCsv(runData: RunData): string {
  const headers = ["Override", "ImagePath", "Classification", "Confidence", "Camera"];
  const rows = runData.results.map(({ override, imagePath, classification, confidence, camera }) => {
    return [
      override ? "Y" : "N",
      `"${imagePath}"`, // so paths with commas don't break csv
      override ? "NA" : classification,
      confidence,
      camera
    ].join(",");
  });
  return [headers.join(","), ...rows].join("\n");
}

ipcMain.handle("save-csv-dialog", async (event: IpcMainInvokeEvent, runData: RunData) => {
  const result = await dialog.showSaveDialog({
    title: "Export results to .csv",
    defaultPath: `${runData.runDate}.csv`,
    filters: [{
      name: "CSV Files", extensions: ["csv"]
    }]
  });

  if (result.canceled || !result.filePath) {
    return null; // user decided not to save file/cancel
  }

  const csvData = convertRunDataToCsv(runData);

  try {
    await fs.writeFile(result.filePath, csvData, "utf-8");
    const filePath = result.filePath;
    shell.showItemInFolder(filePath);
    return result.filePath;
  } catch (error) {
    console.error("Failed to save CSV:", error);
    return null;
  }
});

ipcMain.handle("update-image-classification", async (event: IpcMainInvokeEvent, runData: RunData) => {
  try {
    // Simply overwrite the existing file with updated runData
    await fs.writeFile(runData.filePath, JSON.stringify(runData, null, 2), "utf-8");
    return { success: true, filePath: runData.filePath };
  } catch (error: any) {
    console.error("Failed to update image classification:", error);
    return { success: false, error: error.message };
  }
});

// Handle listing all runs from Documents/Leapfrog/runs dir
ipcMain.handle("list-runs", async () => {
  try {
    const runsFolder = path.join(os.homedir(), "Documents", "Leapfrog", "runs");
    // make runsFolder if it doesn't exist
    await fs.mkdir(runsFolder, { recursive: true });
    const filenames: string[] = await fs.readdir(runsFolder);

    // extract date, time, and filePath from each filename in runs folder
    const processedRuns = filenames
      .filter((filename) => filename.endsWith(".json"))
      .map((filename) => {
        const match = filename.match(/^(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})\.json$/);
        if (!match) return null;
  
        const [_, date, hour, minute] = match; // Extract date and time components
  
        // Convert hour to US time format (AM/PM)
        let hourNumber = parseInt(hour, 10);
        const isPM = hourNumber >= 12;
        if (hourNumber > 12) hourNumber -= 12;
        if (hourNumber === 0) hourNumber = 12;
  
        const formattedTime = `${hourNumber}:${minute} ${isPM ? "PM" : "AM"}`;
  
        // Construct the full file path
        const filePath = path.join(runsFolder, filename);
  
        return { date, time: formattedTime, filePath };
      })
      .filter((entry) => entry !== null) as { date: string; time: string; filePath: string }[];
  
    return processedRuns;
  } catch (error) {
    console.error("Error processing runs:", error);
    return [];
  }
});

interface RunData {
  filePath: string;
  runDate: string;
  frogs: number;
  notFrogs: number;
  averageConfidence: number;
  results: Array<ImageResultData>;
}

interface ImageResultData {
  name: string;
  imagePath: string;
  classification: "FROG" | "NOT FROG";
  confidence: number;
  override: boolean;
  camera: boolean;
}

function isValidImageResult(result: any): result is ImageResultData {
  return (
    typeof result.imagePath === "string" &&
    typeof result.name === "string" &&
    (result.classification === "FROG" || result.classification === "NOT FROG") &&
    typeof result.confidence === "number" &&
    typeof result.override === "boolean"
  );
}

function isValidRunData(data: any): data is RunData {
  return (
    typeof data.runDate === "string" &&
    typeof data.frogs === "number" &&
    typeof data.notFrogs === "number" &&
    typeof data.filePath === "string" &&
    Array.isArray(data.results) &&
    data.results.every(isValidImageResult)
  );
}

function sortRunDataResultsByConfidence(data: Array<ImageResultData>): Array<ImageResultData> {
  return data.sort((a, b) => a.confidence - b.confidence);
}

ipcMain.handle("get-run-data", async (event: IpcMainInvokeEvent, runResultPath: string): Promise<RunData | null> => {
  try {
    const rawData = await fs.readFile(runResultPath, "utf-8");
    const data = JSON.parse(rawData);

    if (!isValidRunData(data)) {
      console.error("Invalid RunData structure:", data);
      return null;
    }
    data.results = sortRunDataResultsByConfidence(data.results);
    return data;
  } catch (error) {
    console.error("Error reading or parsing file:", error);
    return null;
  }
});

ipcMain.handle("get-raw-image-data", async (event: IpcMainInvokeEvent, imagePath: string) => {
  try {
    const rawImageData = await fs.readFile(imagePath);
    return { success: true, data: rawImageData.toString("base64") };
  } catch (err) {
    return { success: false, message: "unable to read image" };
  }
})

ipcMain.on("run-process-images", (event: IpcMainEvent, folderPath: string, cameraNumber: number) => {
  const processImagesScriptPath = path.resolve(resourcesFolder, "backend", "process_images.py");
  const modelPath = path.resolve(resourcesFolder, "backend", "frog_detector.h5");
  const cameraString = cameraNumber.toString();
  const processImagesProcess = spawn(pythonPath, ["-u", processImagesScriptPath, modelPath, folderPath, cameraString], {
    cwd: path.resolve(resourcesFolder, "backend"),
  });

  // send stdout to event listener in SortView.tsx
  processImagesProcess.stdout.on("data", (data: Buffer) => {
    const output = data.toString(); // Convert Buffer to string
    console.log(`got: ${output}`);
    event.sender.send("process-images-output", output); // Send to renderer
  });

  // Capture and send stderr data to the renderer process
  processImagesProcess.stderr.on("data", (error: Buffer) => {
    const errorMessage = error.toString(); // Convert Buffer to string
    console.error(`stderr: ${errorMessage}`);
    event.sender.send("process-images-error", errorMessage); // Send to renderer
  });

  // Notify renderer process when the Python script finishes
  processImagesProcess.on("close", (code: string) => {
    console.log(`Python process exited with code ${code}`);
    event.sender.send("process-images-done", code); // Send exit code to renderer
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
