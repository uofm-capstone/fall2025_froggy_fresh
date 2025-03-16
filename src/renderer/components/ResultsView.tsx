"use client";

import { useState, useEffect } from "react";
import BackButton from "./BackButton";

// Define the type for a sorting run result
interface SortResults {
  runDate: string;
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

export default function ResultsView({ onBack }: { onBack: () => void }) {
  const [runs, setRuns] = useState<SortResults[]>([]);
  const [selectedRun, setSelectedRun] = useState<SortResults | null>(null);
  const [selectedImage, setSelectedImage] = useState<{
    name: string;
    classification: string;
    confidence: number;
  } | null>(null);

  useEffect(() => {
    async function fetchRuns() {
      try {
        const response = await fetch("http://127.0.0.1:5000/results");
        if (response.ok) {
          const data = await response.json();
          setRuns(data);
          // Set latest run as default if exists:
          if (data.length > 0) {
            setSelectedRun(data[data.length - 1]);
          }
        } else {
          console.error("Failed to fetch results");
        }
      } catch (error) {
        console.error("Error fetching results:", error);
      }
    }
    fetchRuns();
  }, []);

  return (
    <div>
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-8">
          <BackButton onClick={onBack} />
          <h2 className="text-3xl font-semibold text-[var(--apple-text)]">Results</h2>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Runs sidebar */}
        <div className="apple-card p-0 overflow-hidden">
          <div className="p-4 bg-[#f0f0f0] dark:bg-[#2c2c2e] font-medium text-[var(--apple-text)]">
            Runs
          </div>
          <div>
            {runs.length > 0 ? (
              runs.map((run, index) => (
                <button
                  key={index}
                  onClick={() => {
                    setSelectedRun(run);
                    setSelectedImage(null); // reset selected image on new run
                  }}
                  className={`w-full text-left p-4 text-[var(--apple-text)] hover:bg-[#f0f0f0] dark:hover:bg-[#2c2c2e] transition-colors rounded-xl ${
                    selectedRun && selectedRun.runDate === run.runDate
                      ? "bg-[var(--apple-accent)] text-white"
                      : ""
                  }`}
                >
                  {run.runDate}
                </button>
              ))
            ) : (
              <div className="p-4 text-[var(--apple-text)]">No runs available</div>
            )}
          </div>
        </div>

        {/* Main content: Show selected run details in a table */}
        <div className="lg:col-span-3">
          <div className="apple-card p-0 overflow-hidden">
            <div className="grid grid-cols-12 gap-2 p-3 bg-[#f0f0f0] dark:bg-[#2c2c2e] font-medium text-[var(--apple-text)]">
              <div className="col-span-1">#</div>
              <div className="col-span-6">File</div>
              <div className="col-span-3">Classification</div>
              <div className="col-span-2">Confidence</div>
            </div>
            <div className="max-h-[300px] overflow-y-auto scrollbar-hidden">
              {selectedRun && selectedRun.files.length > 0 ? (
                selectedRun.files.map((file, idx) => (
                  <div
                    key={idx}
                    onClick={() => setSelectedImage(file)}
                    className="grid grid-cols-12 gap-2 p-2.5 border-b border-[var(--apple-border)] text-[var(--apple-text)] hover:bg-[#f9f9f9] dark:hover:bg-[#2c2c2e]/50 cursor-pointer"
                  >
                    <div className="col-span-1">{idx + 1}</div>
                    <div className="col-span-6 truncate">{file.name}</div>
                    <div className="col-span-3">{file.classification}</div>
                    <div className="col-span-2">{file.confidence}%</div>
                  </div>
                ))
              ) : (
                <div className="p-2.5 text-[var(--apple-text)]">
                  No files available.
                </div>
              )}
            </div>
          </div>

          {/* Image preview section (always visible) */}
          <div className="mt-6">
            <div className="apple-card p-6">
              <div className="relative w-full aspect-video bg-[#1c1c1e] rounded-lg overflow-hidden mb-4">
                {selectedImage ? (
                  <>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <p className="text-white/50">{selectedImage.name}</p>
                    </div>
                    <div className="absolute bottom-2 right-2 text-xs bg-black/70 text-white px-2 py-1 rounded">
                      {selectedImage.name}
                    </div>
                  </>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <p className="text-white/50">Select a picture to view</p>
                  </div>
                )}
              </div>
              <div className="flex gap-6">
                <button className="flex-1 bg-[var(--apple-green)] hover:opacity-90 text-white text-xl py-4 rounded-xl border-0 transition-opacity">
                  FROG
                </button>
                <button className="flex-1 bg-[var(--apple-red)] hover:opacity-90 text-white text-xl py-4 rounded-xl border-0 transition-opacity">
                  NOT FROG
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
