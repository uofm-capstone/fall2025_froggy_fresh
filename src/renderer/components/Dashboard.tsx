"use client";
const { ipcRenderer } = window.require("electron");

import { useEffect, useState } from "react";
import FrogChart from "./FrogChart";
import { GraphData } from "./FrogChart";

interface DashboardProps {
  onSortClick: () => void;
  onResultsClick: () => void;
}

export default function Dashboard({ onSortClick, onResultsClick }: DashboardProps) {
  const [availableCameras, setAvailableCameras] = useState<number[]>([]);
  const [selectedCameras, setSelectedCameras] = useState<number[]>([]);
  const [graphData, setGraphData] = useState<Array<GraphData> | null | undefined>(null);

  useEffect(() => {
    loadGraphData();
  }, []);

  async function loadGraphData() {
    setGraphData(undefined); 
    ipcRenderer.invoke("get-graph-data").then((loadedData: Array<GraphData> | null) => {
      if (loadedData) {
        const cams = Array.from(new Set(loadedData.map(r => r.camera))).sort((a, b) => a - b);
        setAvailableCameras(cams);
        setGraphData(loadedData);
        // default select first camera (or all, if you prefer: setSelectedCameras(cams))
        if (cams.length > 0) setSelectedCameras([cams[0]]);
      } else {
        setGraphData([]);
        setAvailableCameras([]);
        setSelectedCameras([]);
      }
    });
  }

  function toggleCamera(cameraNumber: number) {
    setSelectedCameras(prev =>
      prev.includes(cameraNumber)
        ? prev.filter(c => c !== cameraNumber)
        : [...prev, cameraNumber].sort((a, b) => a - b)
    );
  }

  return (
    <div className="flex flex-col items-center">
      <h1 className="text-4xl font-bold mb-10 text-[var(--apple-text)]">Leapfrog</h1>

      {/* Action buttons */}
      <div className="flex gap-6 mb-12">
        <button onClick={onSortClick} className="apple-button-outline">
          Sort Frog Images
          <div className="text-[12px] text-[var(--apple-subtle-text)]">
            Select a folder and begin sorting
          </div>
        </button>

        <button
          onClick={onResultsClick}
          className="apple-button"
        >
          Results
        </button>
      </div>

      {/* Chart section */}
      <div className="w-full max-w-4xl">
        <h2 className="text-2xl font-semibold text-center mb-6 text-[var(--apple-text)]">Frogs Over Time</h2>

        {/* Multi-camera selection (checkboxes) */}
        <div className="mb-4 apple-card p-4">
          <div className="text-sm text-[var(--apple-text)] mb-2">
            {graphData === undefined ? "Loading cameras..." : "Select one or more cameras:"}
          </div>
          <div className="flex flex-wrap gap-3">
            {availableCameras.map((cam) => (
              <label key={cam} className="inline-flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  className="accent-[var(--apple-accent)]"
                  checked={selectedCameras.includes(cam)}
                  onChange={() => toggleCamera(cam)}
                />
                <span className="text-[var(--apple-text)]">Camera {cam}</span>
              </label>
            ))}
            {availableCameras.length === 0 && graphData !== undefined && (
              <span className="text-[var(--apple-text-secondary)]">No cameras found</span>
            )}
          </div>
        </div>

        <div className="apple-card h-[300px]">
          <FrogChart cameraIds={selectedCameras} graphData={graphData} />
        </div>

        <div className="flex justify-between text-[var(--apple-text-secondary)] mt-2">
          <span>Feb 2025</span>
          <span>Dec 2025</span>
        </div>
      </div>
    </div>
  );
}
