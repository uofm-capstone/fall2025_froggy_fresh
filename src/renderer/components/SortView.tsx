import React, { useState, useRef } from "react";
import BackButton from "./BackButton";
import JSZip from "jszip";

import { dialog } from "electron";
const { ipcRenderer } = window.require("electron");

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
  const [folderPath, setFolderPath] = useState<string>("");
  const [isFolderSelected, setIsFolderSelected] = useState<boolean>(false);
  // New state to store backend stats
  const [stats, setStats] = useState<{
    frogs: number;
    notFrogs: number;
    confidence: number;
    files: any[];
    totalFiles: string;
    currentFile: string;
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

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

  // Handle start button click: send path of input folder to backend
  const handleStart = async () => {
    console.log(`handleStart says folder path '${folderPath}'`);
    const response = await fetch("http://127.0.0.1:5000/upload", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        folderPath: folderPath,
      })
    });
    
    if (fileInputRef.current?.files) {
      // const files = fileInputRef.current?.files;
      // Array.from(files).forEach((file) => {
      //   // console.log(file.name);
      // })
      // const zip = new JSZip();
      // // Add all files preserving folder structure
      // Array.from(fileInputRef.current.files).forEach((file) => {
      //   zip.file(file.webkitRelativePath, file);
      // });

      // try {
      //   // Generate zip blob
      //   const zipBlob = await zip.generateAsync({ type: "blob" });

      //   // Create form data to send to backend
      //   const formData = new FormData();
      //   formData.append("file", zipBlob, "uploaded_folder.zip");

      //   const response = await fetch("http://127.0.0.1:5000/upload", {
      //     method: "POST",
      //     body: formData,
      //   });

      //   if (response.ok) {
      //     const data = await response.json();
      //     console.log("Upload successful:", data);
      //     setStats(data);
      //   } else {
      //     console.error("Upload failed");
      //   }
      // } catch (error) {
      //   console.error("Error during zipping or upload:", error);
      // }
    }
  };

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
            Start
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
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-[var(--apple-text)]">
                  {stats ? `${stats.confidence}%` : "-"}
                </div>
                <div className="text-sm text-[var(--apple-text)] uppercase">CONFIDENCE</div>
              </div>
            </div>

            <div className="h-2 w-full bg-[#f5f5f5] dark:bg-[#1c1c1e] rounded-full mb-6">
              <div
                className="h-2 bg-[var(--apple-accent)] rounded-full"
                style={{ width: stats ? `${stats.confidence}%` : "0%" }}
              ></div>
            </div>

            <div className="space-y-2 mb-6 min-h-[160px] flex items-center justify-center">
              <div className="text-[var(--apple-text-secondary)]">
                {stats ? "Processing complete" : "Select a folder to begin sorting"}
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