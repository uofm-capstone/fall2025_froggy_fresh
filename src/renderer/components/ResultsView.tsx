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
      <div className="mb-6">
        <div className="flex items-center gap-4 mb-6">
          <BackButton onClick={onBack} />
          <h2 className="text-3xl font-semibold text-[var(--apple-text)]">Results</h2>
        </div>
      </div>
      
      <div className="apple-card w-full p-0 overflow-hidden">
        <div className="grid grid-cols-11 gap-0">
          {/* Header row */}
          <div className="col-span-11 grid grid-cols-11 border-b border-[var(--apple-border)]">
            <div className="col-span-2 p-4 font-medium text-[var(--apple-text)]
                        bg-[#f0f0f0] dark:bg-[#2c2c2e] border-r border-[var(--apple-border)]">
              Runs
            </div>

            <div className="col-span-5 grid grid-cols-10 p-0 bg-[#f0f0f0]
                        dark:bg-[#2c2c2e] font-medium text-[var(--apple-text)]
                        border-r border-[var(--apple-border)]">
              <div className="col-span-1 p-4 text-center">#</div>
              <div className="col-span-3 p-4">File</div>
              <div className="col-span-3 p-4">Classification</div>
              <div className="col-span-3 p-4">Confidence</div>
            </div>

            <div className="col-span-4 p-4 bg-[#f0f0f0] dark:bg-[#2c2c2e]
                        font-medium text-[var(--apple-text)]">
              Preview
            </div>
          </div>

          {/* Content area */}
          <div className="col-span-11 grid grid-cols-11">
            <div className="col-span-2 border-r border-[var(--apple-border)]">
              <div className="max-h-[calc(100vh-205px)] overflow-y-auto">
                {runs.length > 0 ? (
                  runs.map((run, index) => (
                    <button
                      key={index}
                      onClick={() => {
                        setSelectedRun(run);
                        setSelectedImage(null); 
                      }}
                      className={`w-full text-left p-4 text-[var(--apple-text)]
                                 hover:bg-[#f0f0f0] dark:hover:bg-[#2c2c2e]
                                 transition-colors border-b border-[var(--apple-border)]
                                 ${
                                   selectedRun &&
                                   selectedRun.runDate === run.runDate
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

            <div className="col-span-5 border-r border-[var(--apple-border)]">
              <div className="max-h-[calc(100vh-205px)] overflow-y-auto scrollbar-hidden">
                {selectedRun && selectedRun.files.length > 0 ? (
                  selectedRun.files.map((file, idx) => (
                    <div
                      key={idx}
                      onClick={() => setSelectedImage(file)}
                      className={`grid grid-cols-10 border-b border-[var(--apple-border)]
                                 text-[var(--apple-text)] hover:bg-[#f9f9f9]
                                 dark:hover:bg-[#2c2c2e]/50 cursor-pointer ${
                                   selectedImage && selectedImage.name === file.name
                                     ? "bg-[#e6f2ff] dark:bg-[#00366d]/30"
                                     : ""
                                 }`}
                    >
                      <div className="col-span-1 p-4 text-center">{idx + 1}</div>
                      <div className="col-span-3 p-4 truncate">{file.name}</div>
                      <div className="col-span-3 p-4">{file.classification}</div>
                      <div className="col-span-3 p-4">{file.confidence}%</div>
                    </div>
                  ))
                ) : (
                  <div className="p-4 text-[var(--apple-text)]"></div>
                )}
              </div>
            </div>

            <div className="col-span-4">
              <div className="p-4 flex flex-col h-full">
                <div className="relative w-full h-72 bg-[#1c1c1e] rounded-lg
                            overflow-hidden mb-4">
                  {selectedImage ? (
                    <>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <p className="text-white/50">{selectedImage.name}</p>
                      </div>
                      <div className="absolute bottom-2 right-2 text-xs
                                  bg-black/70 text-white px-2 py-1 rounded">
                        {selectedImage.name}
                      </div>
                    </>
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <p className="text-white/50">Select a picture to view</p>
                    </div>
                  )}
                </div>

                <div className="flex gap-4 mt-4">
                  <button className="flex-1 bg-[var(--apple-green)]
                             hover:opacity-90 text-white text-xl py-3
                             rounded-xl border-0 transition-opacity">
                    FROG
                  </button>
                  <button className="flex-1 bg-[var(--apple-red)]
                             hover:opacity-90 text-white text-xl py-3
                             rounded-xl border-0 transition-opacity">
                    NOT FROG
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}