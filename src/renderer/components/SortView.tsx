"use client";

import { useState, useRef } from "react";
import BackButton from "./BackButton";

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
  const fileInputRef = useRef<ExtendedHTMLInputElement | null>(null); // Use the extended type

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
      fileInputRef.current.click(); // Trigger the file input dialog
    }
  };

  // Handle folder selection
  const handleFolderSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      setFolderPath(files[0].webkitRelativePath.split("/")[0]); // Set folder path from the first file
      setIsFolderSelected(true);
      // You can process the files here if needed
    }
  };

  // Handle start button click
  const handleStart = () => {
    // In a real app, this would start the actual processing
    onSortComplete(simulatedResults);
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
        style={{ display: "none" }} // Hide the input
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
              <button
                onClick={handleBrowse}
                className="apple-button-secondary"
              >
                Browse
              </button>
            </div>
          </div>

          <div className="mb-8">
            <div className="apple-card">
              <h3 className="text-2xl font-bold text-[var(--apple-text)] mb-6">Folder Structure</h3>
              <div className="h-[120px] bg-[#f5f5f5] dark:bg-[#1c1c1e] flex items-center justify-center text-[var(--apple-text-secondary)] text-sm border border-[var(--apple-border)] rounded-lg">
              </div>
            </div>
          </div>

          <div className="space-y-4 mb-8">
          </div>

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
          {isFolderSelected ? (
            <div className="flex flex-col h-full">
              <div className="flex justify-between items-center mb-6">
                <div className="flex gap-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-[var(--apple-accent)]">20</div>
                    <div className="text-sm text-[var(--apple-text)] uppercase">FROGS</div>
                  </div>

                  <div className="text-center">
                    <div className="text-3xl font-bold text-[var(--apple-text)]">122</div>
                    <div className="text-sm text-[var(--apple-text)] uppercase">NOT FROG</div>
                  </div>
                </div>

                <div className="text-center">
                  <div className="text-3xl font-bold text-[var(--apple-text)]">87<span className="text-lg">%</span></div>
                  <div className="text-sm text-[var(--apple-text)] uppercase">CONFIDENCE</div>
                </div>
              </div>

              <div className="h-2 w-full bg-[#f5f5f5] dark:bg-[#1c1c1e] rounded-full mb-6">
                <div className="h-2 bg-[var(--apple-accent)] rounded-full" style={{ width: "87%" }}></div>
              </div>

              <div style={{ height: '144px' }} className="mb-6 rounded-xl overflow-hidden">
                <div style={{ height: '100%', overflowY: 'auto' }} className="scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-transparent">
                  {simulatedResults.files.map((file, index) => (
                    <div key={index} style={{ height: '36px' }} className="flex justify-between items-center px-4 hover:bg-[#f5f5f5] dark:hover:bg-[#1c1c1e]">
                      <div className="text-[var(--apple-text)] truncate max-w-[180px]">{file.name}</div>
                      <div className="flex items-center gap-4 shrink-0">
                        <div className={file.classification === "FROG" ? "text-[var(--apple-accent)]" : "text-[var(--apple-red)]"}>
                          {file.classification}
                        </div>
                        <div className="text-[var(--apple-text)] w-[40px] text-right">{file.confidence}%</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="text-[var(--apple-text-secondary)] text-sm text-center mb-4">
                ({simulatedResults.totalFiles})
              </div>

              <div className="text-[var(--apple-accent)] text-sm mb-6">
                {simulatedResults.currentFile}
              </div>

              <button className="apple-button-secondary w-full">
                Open in folder
              </button>
            </div>
          ) : (
            <div className="flex flex-col h-full">
              <div className="flex justify-between items-center mb-6">
                <div className="flex gap-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-[var(--apple-accent)]">-</div>
                    <div className="text-sm text-[var(--apple-text)] uppercase">FROGS</div>
                  </div>

                  <div className="text-center">
                    <div className="text-3xl font-bold text-[var(--apple-text)]">-</div>
                    <div className="text-sm text-[var(--apple-text)] uppercase">NOT FROG</div>
                  </div>
                </div>

                <div className="text-center">
                  <div className="text-3xl font-bold text-[var(--apple-text)]">-<span className="text-lg">%</span></div>
                  <div className="text-sm text-[var(--apple-text)] uppercase">CONFIDENCE</div>
                </div>
              </div>

              <div className="h-2 w-full bg-[#f5f5f5] dark:bg-[#1c1c1e] rounded-full mb-6"></div>

              <div className="space-y-2 mb-6 min-h-[160px] flex items-center justify-center">
                <div className="text-[var(--apple-text-secondary)]">
                  Select a folder to begin sorting
                </div>
              </div>

              <div className="text-[var(--apple-text-secondary)] text-sm text-center mb-4">
                (0/0)
              </div>

              <div className="text-[var(--apple-accent)] text-sm mb-6">
                No file selected
              </div>

              <button className="apple-button-secondary w-full" disabled>
                Open in folder
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
