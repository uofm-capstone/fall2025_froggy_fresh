"use client";

import { useState } from "react";
import Dashboard from "../components/Dashboard";
import SortView from "../components/SortView";
import ResultsView from "../components/ResultsView";
import ThemeToggle from "../components/ThemeToggle";

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

export default function Home() {
  const [currentView, setCurrentView] = useState<"dashboard" | "sort" | "results">("dashboard");
  const [sortResults, setSortResults] = useState<SortResults | null>(null);

  const navigateTo = (view: "dashboard" | "sort" | "results") => {
    setCurrentView(view);
  };

  const handleSortComplete = (results: SortResults) => {
    setSortResults(results);
    navigateTo("results");
  };

  return (
    <main className="px-4 py-8">
      <ThemeToggle /> 
      
      {currentView === "dashboard" && (
        <div className="mx-auto max-w-4xl">
          <Dashboard
            onSortClick={() => navigateTo("sort")}
            onResultsClick={() => navigateTo("results")}
          />
        </div>
      )}

      {currentView === "sort" && (
        <div className="w-full max-w-none">
          <SortView
            onBack={() => navigateTo("dashboard")}
            onResults={() => navigateTo("results")}
            onSortComplete={handleSortComplete}
          />
        </div>
      )}

      {currentView === "results" && (
        <div className="w-full max-w-none">
          <ResultsView
            onBack={() => navigateTo("dashboard")}
            results={sortResults}
          />
        </div>
      )}
    </main>
  );
}