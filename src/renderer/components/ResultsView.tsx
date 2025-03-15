"use client";

import { useState } from "react";
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

// Define a type for the image result item
interface ImageResult {
  id: number;
  filename: string;
  classification: string;
  confidence: string;
}

interface ResultsViewProps {
  onBack: () => void;
  results: SortResults | null;
}

export default function ResultsView({ onBack, results }: ResultsViewProps) {
  const [selectedCamera, setSelectedCamera] = useState<string>("Camera1");
  const [selectedImage, setSelectedImage] = useState<ImageResult>({
    id: 1,
    filename: "2024-04-12-Camera1-99.jpg",
    classification: "NOT FROG",
    confidence: "98%"
  });

  // Simulated image results data
  const imageResults: ImageResult[] = [
    { id: 1, filename: "2024-04-12-Camera1-99.jpg", classification: "NOT FROG", confidence: "98%" },
    { id: 2, filename: "2024-04-17-Camera1-0009.jpg", classification: "NOT FROG", confidence: "OVERRIDE" },
    { id: 3, filename: "2024-05-01-Camera1-123.jpg", classification: "", confidence: "" },
    { id: 4, filename: "2024-05-02-Camera1-245.jpg", classification: "", confidence: "" },
    { id: 5, filename: "2024-05-03-Camera1-367.jpg", classification: "", confidence: "" },
    { id: 6, filename: "2024-05-04-Camera1-489.jpg", classification: "", confidence: "" },
    { id: 7, filename: "2024-05-05-Camera1-511.jpg", classification: "", confidence: "" },
    { id: 8, filename: "2024-05-06-Camera1-633.jpg", classification: "", confidence: "" },
    { id: 9, filename: "2024-05-07-Camera1-755.jpg", classification: "", confidence: "" },
    { id: 10, filename: "2024-05-08-Camera1-877.jpg", classification: "", confidence: "" },
  ];

  return (
    <div>
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-8">
          <BackButton onClick={onBack} />
          <h2 className="text-3xl font-semibold text-[var(--apple-text)]">Results</h2>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Camera sidebar */}
        <div className="apple-card p-0 overflow-hidden">
          <div className="p-4 bg-[#f0f0f0] dark:bg-[#2c2c2e] font-medium text-[var(--apple-text)]">
            Cameras
          </div>

          <div>
            {["Camera1", "Camera2", "Camera3"].map((camera) => (
              <button
                key={camera}
                onClick={() => setSelectedCamera(camera)}
                className={`w-full text-left p-4 text-[var(--apple-text)] hover:bg-[#f0f0f0] dark:hover:bg-[#2c2c2e] transition-colors rounded-xl ${
                  selectedCamera === camera ? "bg-[var(--apple-accent)] text-white" : ""
                }`}
              >
                {camera}
              </button>
            ))}
          </div>
        </div>

        {/* Main content */}
        <div className="lg:col-span-3">
          {/* Results table */}
          <div className="apple-card p-0 overflow-hidden mb-6">
            <div className="grid grid-cols-12 gap-2 p-3 bg-[#f0f0f0] dark:bg-[#2c2c2e] font-medium text-[var(--apple-text)]">
              <div className="col-span-1">#</div>
              <div className="col-span-6">File</div>
              <div className="col-span-3">Classification</div>
              <div className="col-span-2">Confidence</div>
            </div>

            <div className="max-h-[300px] overflow-y-auto scrollbar-hidden">
              {imageResults.map((image) => (
                <div
                  key={image.id}
                  onClick={() => setSelectedImage(image)}
                  className={`grid grid-cols-12 gap-2 p-2.5 border-b border-[var(--apple-border)] text-[var(--apple-text)] hover:bg-[#f9f9f9] dark:hover:bg-[#2c2c2e]/50 cursor-pointer`}
                >
                  <div className="col-span-1">{image.id}</div>
                  <div className="col-span-6 truncate">{image.filename}</div>
                  <div className={`col-span-3 ${
                    image.classification === "NOT FROG" ? "text-[var(--apple-red)]" :
                    image.classification === "FROG" ? "text-[var(--apple-green)]" : ""
                  }`}>
                    {image.classification}
                  </div>
                  <div className="col-span-2">
                    {image.confidence === "OVERRIDE" ? (
                      <span className="text-amber-500">OVERRIDE</span>
                    ) : (
                      image.confidence
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Image preview and classification */}
          {selectedImage && (
            <div className="lg:col-span-3">
              <div className="apple-card p-6">
                <div className="relative w-full aspect-video bg-[#1c1c1e] rounded-lg overflow-hidden mb-4">
                  {/* Placeholder for the image */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <p className="text-white/50">Image: {selectedImage.filename}</p>
                  </div>
                  <div className="absolute bottom-2 right-2 text-xs bg-black/70 text-white px-2 py-1 rounded">
                    {selectedImage.filename}
                  </div>
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
          )}
        </div>
      </div>
    </div>
  );
}
