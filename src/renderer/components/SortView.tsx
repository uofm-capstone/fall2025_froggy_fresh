import React, { useState, useRef } from "react";
import BackButton from "./BackButton";
import JSZip from "jszip";

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
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Handle browse button click
  const handleBrowse = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Handle folder selection
  const handleFolderSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
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
        zip.file(file.webkitRelativePath, file);
      });

      try {
        // Generate zip blob
        const zipBlob = await zip.generateAsync({ type: "blob" });

        // Create form data to send to backend
        const formData = new FormData();
        formData.append("file", zipBlob, "uploaded_folder.zip");

        const response = await fetch("http://127.0.0.1:5000/upload", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          console.log("Upload successful:", data.message);
          onSortComplete({
            frogs: 20,
            notFrogs: 122,
            confidence: 87,
            files: [],
            totalFiles: "142/1281",
            currentFile: "./output/Camera1/2024-04-12-Camera1-100.jpg"
          });
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
      <input
        type="file"
        ref={fileInputRef}
        multiple
        // Use a spread and cast to any to bypass TS type checking for non-standard attributes
        {...({ webkitdirectory: "true", directory: "true" } as any)}
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
                  <div className="text-3xl font-bold text-[var(--apple-accent)]">{isFolderSelected ? "20" : "-"}</div>
                  <div className="text-sm text-[var(--apple-text)] uppercase">FROGS</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-[var(--apple-text)]">{isFolderSelected ? "122" : "-"}</div>
                  <div className="text-sm text-[var(--apple-text)] uppercase">NOT FROG</div>
                </div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-[var(--apple-text)]">{isFolderSelected ? "87%" : "-"}</div>
                <div className="text-sm text-[var(--apple-text)] uppercase">CONFIDENCE</div>
              </div>
            </div>

            <div className="h-2 w-full bg-[#f5f5f5] dark:bg-[#1c1c1e] rounded-full mb-6">
              {isFolderSelected && (
                <div className="h-2 bg-[var(--apple-accent)] rounded-full" style={{ width: "87%" }}></div>
              )}
            </div>

            <div className="space-y-2 mb-6 min-h-[160px] flex items-center justify-center">
              <div className="text-[var(--apple-text-secondary)]">
                {isFolderSelected ? "Processing files..." : "Select a folder to begin sorting"}
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