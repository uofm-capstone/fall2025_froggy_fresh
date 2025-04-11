import React, { useState, useRef } from "react";
import BackButton from "./BackButton";
const { ipcRenderer } = window.require("electron");
import { IpcRendererEvent } from "electron";

// Define SortViewProps interface
interface SortViewProps {
    onBack: () => void;
    onSortComplete: (result: {
        frogs: number;
        notFrogs: number;
        confidence: number;
        files: any[];
        totalFiles: string;
        currentFile: string;
    }) => void;
}

export default function SortView({ onBack, onSortComplete }: SortViewProps) {
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [folderPath, setFolderPath] = useState<string>("");
  const [isFolderSelected, setIsFolderSelected] = useState<boolean>(false);
  // New state to store backend stats
  const [stats, setStats] = useState<{
    frogs: number;
    notFrogs: number;
    averageConfidence: number;
    totalFileCount: number;
    progressFileCount: number;
  } | null>(null);
  
  
  // Handle browse button click
  const handleBrowse = async () => {
    const newFolderPath = await ipcRenderer.invoke('open-directory-dialog');
    if (newFolderPath) {
      console.log('Selected folder:', newFolderPath);
      setFolderPath(newFolderPath);
      setIsFolderSelected(true);
    } else {
      console.log('No folder selected');
      setIsFolderSelected(false);
    }
  };

  type FileData = {
    name: string; // The base name of the file
    absolutePath: string; // The full path to the file
    classification: string; // The classification result (e.g., "frog", "not frog")
    confidence: number; // Confidence level as a percentage (integer)
  };
  
  type ProgressData = {
    frogs: number; // Count of files classified as "frogs"
    notFrogs: number; // Count of files classified as "not frogs"
    averageConfidence: number; // Average confidence across processed files
    processedImageCount: number; // Number of images processed so far
    totalImageCount: number; // Total number of images to process
  };
  
  // type of the json printed by python process_images
  type UpdateData = {
    currentFile: FileData; // Data about the current file being processed
    progress: ProgressData; // Progress tracking data
  };
  
  const [lastFiveResults, setLastFiveResults] = useState<FileData[]>([]);
  // Handle start button click: send path of input folder to backend
  const handleStart = async () => {
    console.log(`handleStart says folder path '${folderPath}'`);

    ipcRenderer.send("run-process-images", folderPath);
    setIsRunning(true);
    const updateOutput = (event: IpcRendererEvent, line: string) => {
      try {
        // Parse the json string into the expected UpdateData type
        const updateData: UpdateData = JSON.parse(line);
        const currentFile = updateData.currentFile;
    
        setLastFiveResults((prev) => {
          const updatedResults = [...prev, currentFile];
          return updatedResults.slice(-5); // keep only last 5 results
        });

        const progress = updateData.progress;
        setStats({
          frogs: progress.frogs,
          notFrogs: progress.notFrogs,
          progressFileCount: progress.processedImageCount,
          totalFileCount: progress.totalImageCount,
          averageConfidence: progress.averageConfidence,
        });

        if (progress.processedImageCount === progress.totalImageCount) {
          setIsRunning(false);
        }
    
      } catch (error) {
        // Handle cases where the JSON parsing fails (e.g., invalid input)
        // console.error("Failed to parse JSON:", error, "Line received:", line);
      }
    };    

    ipcRenderer.on("process-images-output", updateOutput);
    
    // old code
    // const response = await fetch("http://127.0.0.1:5000/upload", {
    //   method: "POST",
    //   headers: {
    //     "Content-Type": "application/json",
    //   },
    //   body: JSON.stringify({
    //     folderPath: folderPath,
    //   })
    // });

  };

  // Ensure cleanup when SortView unmounts
  // React.useEffect(() => {
  //   return () => {
  //     ipcRenderer.removeAllListeners("process-images-output");
  //   };
  // }, []);

  return (
    <div>
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-8">
          <BackButton onClick={onBack} />
          <h2 className="text-3xl font-semibold text-[var(--apple-text)]">Sort</h2>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <div className="mb-6">
            <div className="flex gap-2">
              <input
                type="text"
                value={folderPath}
                className="flex-1 rounded-xl border border-[var(--apple-border)] bg-[#f5f5f5] dark:bg-[#1c1c1e] px-4 py-2 text-[var(--apple-text)] focus:outline-none focus:ring-2 focus:ring-[var(--apple-accent)] focus:border-transparent transition-all duration-200"
                placeholder="Select folder"
                readOnly
              />
              <button onClick={handleBrowse} className="apple-button-secondary">
                Browse
              </button>
            </div>
          </div>

          <button
            onClick={handleStart}
            disabled={!isFolderSelected}
            className="apple-button w-full disabled:opacity-50 rounded-xl py-3 text-lg font-medium bg-[var(--apple-accent)] text-white hover:bg-opacity-90 transition-all duration-200 shadow-sm"
          >
            {
              isRunning ? "Starting..." : "Start"
            }
          </button>
        </div>

        <div className="apple-card">
          <h3 className="text-2xl font-bold text-[var(--apple-text)] mb-6">Stats</h3>
          <div className="flex flex-col h-full">
            <div className="flex justify-between items-center mb-6">
              <div className="flex gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-[var(--apple-accent)]">
                    {stats ? stats.frogs : "-"}
                  </div>
                  <div className="text-sm text-[var(--apple-text)] uppercase">FROGS</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-[var(--apple-text)]">
                    {stats ? stats.notFrogs : "-"}
                  </div>
                  <div className="text-sm text-[var(--apple-text)] uppercase">NOT FROG</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-[var(--apple-text)]">
                    {stats ? `${stats.averageConfidence}%` : "-"}
                  </div>
                  <div className="text-sm text-[var(--apple-text)] uppercase">CONFIDENCE</div>
                </div>
              </div>
              <div className="text-center">
                <div className="text-1xl font-bold text-[var(--apple-text)]">
                  {stats ? `(${stats.progressFileCount}/${stats.totalFileCount})` : "-"}
                </div>
                <div className="text-sm text-[var(--apple-text)] uppercase">IMAGES</div>
              </div>
            </div>

            <div className="h-2 w-full bg-[#f5f5f5] dark:bg-[#1c1c1e] rounded-full mb-6">
              <div
                className="h-2 bg-[var(--apple-accent)] rounded-full"
                style={{ width: stats ? `${stats.averageConfidence}%` : "0%" }}
              ></div>
            </div>

            <div className="space-y-2 mb-6 min-h-[160px] flex items-center justify-left">
              <div className="text-[var(--apple-text-secondary)]">
                {lastFiveResults.length > 0 ? (
                  <ul className="space-y-1">
                    {lastFiveResults.map((file, index) => (
                      <li key={index}>
                        {/* <strong>{file.name}</strong> - {file.classification} ({file.confidence}%) */}
                        <strong>
                          {file.name} -{" "}
                          <span style={{ color: file.classification === "FROG" ? "limegreen" : "red" }}>
                            {file.classification}
                          </span>
                        </strong>
                        {" "}
                        ({file.confidence}%)
                      </li>
                    ))}
                  </ul>
                ) : (
                  "Select a folder to begin sorting"
                )}
              </div>
            </div>

            <button className="apple-button-secondary w-full" disabled={!isFolderSelected}>
              Open in folder
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}