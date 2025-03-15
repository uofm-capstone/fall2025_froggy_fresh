"use client";

import { useState, useRef } from "react";
import BackButton from "./BackButton";
import JSZip from "jszip";

// Define a proper type for the results
interface SortResults {
  frogs: number;
  notFrogs: number;
  confidence: number;
  files: Array<{
    name: string;
    classification: string;
    confidence: number;
  }>;
  totalFiles: string;
  currentFile: string;
}

interface SortViewProps {
  onBack: () => void;
  onSortComplete: (results: SortResults) => void;
}

// Extend the HTMLInputElement interface to include webkitdirectory
interface ExtendedHTMLInputElement extends HTMLInputElement {
  webkitdirectory: boolean;
}

export default function SortView({ onBack, onSortComplete }: SortViewProps) {
  const [folderPath, setFolderPath] = useState<string>("");
  const [selectedOption, setSelectedOption] = useState<string>("find-camera-folders");
  const [isFolderSelected, setIsFolderSelected] = useState<boolean>(false);
  const fileInputRef = useRef<ExtendedHTMLInputElement | null>(null);

  // Simulated data for the sort results
  const simulatedResults: SortResults = {
    frogs: 20,
    notFrogs: 122,
    confidence: 87,
    files: [
      { name: "2024-04-12-Camera1-99.jpg", classification: "NOT FROG", confidence: 98 },
      { name: "2024-04-12-Camera1-98.jpg", classification: "FROG", confidence: 58 },
      { name: "2024-04-12-Camera1-97.jpg", classification: "NOT FROG", confidence: 98 },
      { name: "2024-04-12-Camera1-96.jpg", classification: "NOT FROG", confidence: 96 },
      { name: "2024-04-12-Camera1-95.jpg", classification: "NOT FROG", confidence: 96 },
      { name: "2024-04-12-Camera1-95.jpg", classification: "NOT FROG", confidence: 96 },
      { name: "2024-04-12-Camera1-95.jpg", classification: "NOT FROG", confidence: 96 },
      { name: "2024-04-12-Camera1-95.jpg", classification: "NOT FROG", confidence: 96 },
      { name: "2024-04-12-Camera1-95.jpg", classification: "NOT FROG", confidence: 96 }
    ],
    totalFiles: "142/1281",
    currentFile: "./output/Camera1/2024-04-12-Camera1-100.jpg"
  };

  // Handle browse button click
  const handleBrowse = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Handle folder selection
  const handleFolderSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      setFolderPath(files[0].webkitRelativePath.split("/")[0]);
      setIsFolderSelected(true);
    }
  };

  // Handle start button click: zip folder and send to backend
  const handleStart = async () => {
    if (fileInputRef.current?.files) {
      const zip = new JSZip();
      // Add all files preserving folder structure
      Array.from(fileInputRef.current.files).forEach((file) => {
        // file.webkitRelativePath contains the subfolder path
        zip.file(file.webkitRelativePath, file);
      });
      
      try {
        // Generate zip blob
        const zipBlob = await zip.generateAsync({ type: "blob" });
        
        // Create form data to send to backend
        const formData = new FormData();
        formData.append("file", zipBlob, "uploaded_folder.zip");
        
        // Replace the URL if your backend is hosted elsewhere
        const response = await fetch("http://127.0.0.1:5000/upload", {
          method: "POST",
          body: formData,
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log("Upload successful:", data.message);
          // Simulate sort completion here, or update your UI as needed
          onSortComplete(simulatedResults);
        } else {
          console.error("Upload failed");
        }
      } catch (error) {
        console.error("Error during zipping or upload:", error);
      }
    }
  };

  return (
    <div>
      {/* Hidden file input for folder selection */}
      <input
        type="file"
        ref={(input) => {
          if (input) {
            input.setAttribute("webkitdirectory", "true");
            fileInputRef.current = input as ExtendedHTMLInputElement;
          }
        }}
        multiple
        onChange={handleFolderSelect}
        style={{ display: "none" }}
      />

      <div className="mb-8">
        <div className="flex items-center gap-4 mb-8">
          <BackButton onClick={onBack} />
          <h2 className="text-3xl font-semibold text-[var(--apple-text)]">Sort</h2>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Left side - folder selection and options */}
        <div>
          <div className="mb-6">
            <div className="flex gap-2">
              <input
                type="text"
                value={folderPath}
                onChange={(e) => setFolderPath(e.target.value)}
                className="flex-1 rounded-xl border border-[var(--apple-border)] bg-[#f5f5f5] dark:bg-[#1c1c1e] px-4 py-2 text-[var(--apple-text)] focus:outline-none focus:ring-2 focus:ring-[var(--apple-accent)] focus:border-transparent transition-all duration-200"
                placeholder="Select folder"
                readOnly
              />
              <button onClick={handleBrowse} className="apple-button-secondary">
                Browse
              </button>
            </div>
          </div>

          <div className="mb-8">
            <div className="apple-card">
              <h3 className="text-2xl font-bold text-[var(--apple-text)] mb-6">
                Folder Structure
              </h3>
              <div className="h-[120px] bg-[#f5f5f5] dark:bg-[#1c1c1e] flex items-center justify-center text-[var(--apple-text-secondary)] text-sm border border-[var(--apple-border)] rounded-lg">
                {/* Optionally, display the folder structure */}
              </div>
            </div>
          </div>

          <div className="space-y-4 mb-8"></div>

          <button
            onClick={handleStart}
            disabled={!isFolderSelected}
            className="apple-button w-full disabled:opacity-50 rounded-xl py-3 text-lg font-medium bg-[var(--apple-accent)] text-white hover:bg-opacity-90 transition-all duration-200 shadow-sm"
          >
            Start
          </button>
        </div>

        {/* Right side - statistics and preview */}
        <div className="apple-card">
          <h3 className="text-2xl font-bold text-[var(--apple-text)] mb-6">Stats</h3>
          {/* Render stats similarly as before */}
        </div>
      </div>
    </div>
  );
}