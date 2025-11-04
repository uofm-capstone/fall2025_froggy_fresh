"use client";
const { ipcRenderer } = window.require("electron");

import { useState } from "react";
import FrogChart from "./FrogChart";
import { GraphData } from "./FrogChart";

interface DashboardProps {
  onSortClick: () => void;
  onResultsClick: () => void;
}


export default function Dashboard({ onSortClick, onResultsClick }: DashboardProps) {
  const [activeCamera, setActiveCamera] = useState<string>("");
  const [availableCameras] = useState<number[]>([]);
  const [cameraOptions, setCameraOptions] = useState<JSX.Element[]>();
  const [graphData, setGraphData] = useState<Array<GraphData> | null | undefined>(null);

  async function loadGraphData() {
    setGraphData(undefined); // Indicate loading state
    ipcRenderer.invoke("get-graph-data").then((loadedData: Array<GraphData> | null) => {
      if (loadedData) {
        console.log("Graph Data from main process:", loadedData);
        for (let runData of loadedData) {
          if (!availableCameras.includes(runData.camera)) {
            availableCameras.push(runData.camera);
          }
        }
        setGraphData(loadedData);
        //Sort camera numbers
        availableCameras.sort((a, b) => a - b);
        setCameraOptions(renderCameraOptions());
      }
    });
  }

  function renderCameraOptions(): JSX.Element[] {
    console.log("Rendering option for cameras:", availableCameras);
    return availableCameras.map((cameraNumber) => (
      <option key={cameraNumber} value={`camera${cameraNumber}`}>
        {`Camera ${cameraNumber}`}
      </option>
    ));
  }

  if (graphData === null) {
    loadGraphData();
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

        <div className="mb-4">
          <select
            value={activeCamera}
            onChange={(e) => setActiveCamera(e.target.value)}
            className="apple-select"
          >
            <option key={null}>Select Camera</option>
            {cameraOptions}
          </select>
        </div>

        <div className="apple-card h-[300px]">
          <FrogChart cameraId={activeCamera} graphData={graphData} />
        </div>

        <div className="flex justify-between text-[var(--apple-text-secondary)] mt-2">
          <span>Feb 2025</span>
          <span>Dec 2025</span>
        </div>
      </div>
    </div>
  );
}
