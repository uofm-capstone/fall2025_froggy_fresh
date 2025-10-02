"use client";

import { useState, useEffect } from "react";
import BackButton from "./BackButton";
const { ipcRenderer } = window.require("electron");

interface ImageResultData {
  imagePath: string;
  rawData: any;
  name: string;
  classification: "FROG" | "NOT FROG";
  confidence: number;
  override: boolean;
  camera: number;
}

interface SelectedRunData {
  filePath: string;
  runDate: string;
  frogs: number;
  notFrogs: number;
  results: Array<ImageResultData>;
}

// represents each <date>_runs.json file in the runs folder
interface RunResultsEntry {
  date: string;
  time: string;
  filePath: string;
}

export default function ResultsView({ onBack }: { onBack: () => void }) {
  const [selectedRun, setSelectedRun] = useState<SelectedRunData | null>(null);
  const [selectedRunResultsEntry, setSelectedRunResultsEntry] = useState<RunResultsEntry | null>(null);
  const [selectedImage, setSelectedImage] = useState<{ image: ImageResultData; loading: boolean} | null>(null);
  const [runsList, setRunsList] = useState<RunResultsEntry[]>([]);

  const selectRunEntry = async (entry: RunResultsEntry) => {
    if (selectedRun?.filePath === entry.filePath) {
      // allow users to deselect current run entry
      setSelectedRun(null);
      setSelectedRunResultsEntry(null);
    } else {
      ipcRenderer.invoke("get-run-data", entry.filePath).then((runData: SelectedRunData | null) => {
        if (runData !== null) {
          setSelectedRun(runData);
          setSelectedRunResultsEntry(entry);
        };
      })
    }
  };

  const selectImage = async (image: ImageResultData) => {
    // handle update to render image somehow
    setSelectedImage({ image: image, loading: true });
    ipcRenderer.invoke("get-raw-image-data", image.imagePath).then((dataResult: { success: true; data: string }) => {
      if (dataResult.success) {
        image.rawData = dataResult.data;
      }
      setSelectedImage({image: image, loading: false });
    })
  };

  const overrideImageClassification = async (newClassification: "FROG" | "NOT FROG") => {
    if (!selectedImage || !selectedRun) return;

    // update image and results since state doesnt update dependently
    // spread operator ... copies existing fields and modifies classification/override
    const updatedImage = { ...selectedImage.image, classification: newClassification, override: true };
    const updatedResults = selectedRun.results.map(image =>
      image.imagePath === selectedImage.image.imagePath ? updatedImage : image
    );

    let updatedRun = { ...selectedRun, results: updatedResults };
    // update frog and notFrogs count without relying on arbitrary counter state
    updatedRun.frogs = updatedResults.filter(image => image.classification === "FROG").length;
    updatedRun.notFrogs = updatedResults.filter(image => image.classification === "NOT FROG").length;

    setSelectedRun(updatedRun);
    setSelectedImage({ ...selectedImage, image: updatedImage });

    await ipcRenderer.invoke("update-image-classification", updatedRun);
  };

  const exportCsv = async () => {
    try {
      const result = await ipcRenderer.invoke("save-csv-dialog", selectedRun);
      if (result) {
        console.log("CSV saved at:", result);
      } else {
        console.log("Export canceled or failed.");
      }
    } catch (error) {
      console.error("Error exporting CSV:", error);
    }
  };

  useEffect(() => {
    const fetchRunList = async () => {
      ipcRenderer.invoke("list-runs").then((newRunList: RunResultsEntry[]) => {
        setRunsList(newRunList);
      });
    };
    fetchRunList();
  }, []); // passing an empty array here means it only gets called once after initial render

  return (
    <div>

      <div className="mb-8">
        <div className="flex items-center gap-4 mb-8">
          <BackButton onClick={onBack} />
          <h2 className="text-3xl font-semibold text-[var(--apple-text)]">Results</h2>
        </div>
      </div>

      <div className="mb-4">
        <div className="apple-card w-full p-0 overflow-hidden border-b border-[var(--apple-border)]">
          <div className="grid grid-cols-[3fr,1fr,1fr,1fr,1fr,0.65fr] gap-0 w-full text-sm bg-[var(--apple-body-bg)]">
            {/* Timestamp column */}
            <div className="p-2 pl-3 font-medium text-[var(--apple-text)] bg-[var(--apple-body-bg)] border-r border-[var(--apple-border)]">
              {selectedRunResultsEntry ? (
                <div>
                  <div>{selectedRunResultsEntry.date} {selectedRunResultsEntry.time}</div>
                  <div className="text-xs text-[var(--apple-subtle-text)]">{selectedRunResultsEntry.filePath}</div>
                </div>
              ) : (
                "No run selected"
              )}
            </div>

            {/* Data columns */}
            <div className="p-2 text-center font-medium text-[var(--apple-text)] bg-[var(--apple-body-bg)] border-r border-[var(--apple-border)] flex items-center justify-center">
              {
                selectedRun ? (
                  `${selectedRun.results.length} IMAGES`
                ) : (
                  "# IMAGES"
                )
              }
            </div>
            <div className="p-2 text-center font-medium text-[var(--apple-text)] bg-[var(--apple-body-bg)] border-r border-[var(--apple-border)] flex items-center justify-center">
              {
                selectedRun ? (
                  `${selectedRun.frogs} FROG`
                ) : (
                  "# FROG"
                )
              }
            </div>
            <div className="p-2 text-center font-medium text-[var(--apple-text)] bg-[var(--apple-body-bg)] border-r border-[var(--apple-border)] flex items-center justify-center">
              {
                selectedRun ? (
                  `${selectedRun.notFrogs} NOT FROG`
                ) : (
                  "# NOT FROG"
                )
              }
            </div>
            <div className="p-2 text-center font-medium text-[var(--apple-text)] bg-[var(--apple-body-bg)] border-r border-[var(--apple-border)] flex items-center justify-center">
              {
                selectedRun ? (
                  <div>
                    <div>
                      {(100 - (selectedRun.results.filter(image => image.override).length / selectedRun.results.length * 100)).toFixed(2)}%
                    </div>
                    {/* <div className="text-xs text-[var(--apple-subtle-text)]">(not overriden/total)</div> */}
                  </div>
                ) : (
                  <div>
                    <div>
                      ACCURACY
                    </div>
                    <div className="text-xs text-[var(--apple-subtle-text)]">(not overriden/total)</div>
                  </div>
                )
              }
            </div>


            {/* Export Button in the 6th column */}
            <div 
              className={`p-2 flex items-center justify-center ${
                selectedRun ? 'bg-blue-500 cursor-pointer' : 'bg-gray-400 cursor-not-allowed'
              }`}
              onClick={selectedRun ? exportCsv : undefined}
            >
              <span className={`text-sm font-semibold ${
                selectedRun ? 'text-white' : 'text-gray-200'
              }`}>
                Export
              </span>
            </div>


          </div>
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
                {runsList.length > 0 ? (
                  runsList.map((run, index) => (
                    <button
                      key={index}
                      onClick={() => {
                        selectRunEntry(run);
                        setSelectedImage(null); 
                      }}
                      className={`w-full text-left p-4 text-[var(--apple-text)]
                                 hover:bg-[#f0f0f0] dark:hover:bg-[#2c2c2e]
                                 transition-colors border-b border-[var(--apple-border)]
                                 ${
                                   selectedRun &&
                                   selectedRun.filePath === run.filePath
                                     ? "bg-[var(--apple-accent)] text-white"
                                     : ""
                                 }`}
                    >
                      {`${run.date} ${run.time}`}
                    </button>
                  ))
                ) : (
                  <div className="p-4 text-[var(--apple-text)]">No runs available</div>
                )}
              </div>
            </div>

            <div className="col-span-5 border-r border-[var(--apple-border)]">
              <div className="max-h-[calc(100vh-205px)] overflow-y-auto scrollbar-hidden">
                {selectedRun && selectedRun.results.length > 0 ? (
                  selectedRun.results.map((file, idx) => (
                    <div
                      key={idx}
                      onClick={() => selectImage(file)}
                      className={`grid grid-cols-10 border-b border-[var(--apple-border)]
                                 text-[var(--apple-text)] hover:bg-[#f9f9f9]
                                 dark:hover:bg-[#2c2c2e]/50 cursor-pointer ${
                                   selectedImage && selectedImage.image.imagePath === file.imagePath
                                     ? "bg-[#e6f2ff] dark:bg-[#00366d]/30"
                                     : ""
                                 }`}
                    >
                      <div className="col-span-1 p-4 text-center">{idx + 1}</div>
                      <div className="col-span-3 p-4 truncate">{file.name}</div>
                      <div className="col-span-3 p-4">{file.classification}</div>
                      <div className="col-span-3 p-4">{file.override === false ? `${file.confidence}%` : "OVERRIDE"}</div>
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
                      {
                        selectedImage.loading === false ? (
                          <img
                            src={`data:image/jpeg;base64,${selectedImage.image.rawData}`} // Absolute file path for Electron
                            alt={selectedImage.image.name}
                            className="absolute inset-0 w-full h-full object-cover"
                          />
                        ) : (
                          <div className="absolute inset-0 flex items-center justify-center">
                            <p className="text-white/50">{selectedImage.image.name} (loading)</p>
                          </div>
                        )
                      }
                      <div className="absolute bottom-2 right-2 text-xs
                                  bg-black/70 text-white px-2 py-1 rounded">
                        {selectedImage.image.name}
                      </div>
                    </>
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <p className="text-white/50">Select a picture to view</p>
                    </div>
                  )}
                </div>

                <div className="flex gap-4 mt-4">
                  <button
                    className="flex-1 bg-[var(--apple-green)] hover:opacity-90 text-white text-xl py-3 rounded-xl border-0 transition-opacity"
                    onClick={() => overrideImageClassification("FROG")}
                  >
                    FROG
                  </button>
                  <button
                    className="flex-1 bg-[var(--apple-red)] hover:opacity-90 text-white text-xl py-3 rounded-xl border-0 transition-opacity"
                    onClick={() => overrideImageClassification("NOT FROG")}
                  >
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