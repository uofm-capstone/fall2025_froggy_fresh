"use client";

import { useState } from "react";
import Dashboard from "../components/Dashboard";
import SortView from "../components/SortView";
import ResultsView from "../components/ResultsView";
import ThemeToggle from "../components/ThemeToggle";


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
    <main className="container mx-auto px-4 py-8">
      <ThemeToggle /> {/* Add this line to include the button */}
      
      {currentView === "dashboard" && (
        <Dashboard
          onSortClick={() => navigateTo("sort")}
          onResultsClick={() => navigateTo("results")}
        />
      )}

      {currentView === "sort" && (
        <SortView
          onBack={() => navigateTo("dashboard")}
          onSortComplete={handleSortComplete}
        />
      )}

      {currentView === "results" && (
        <ResultsView
          onBack={() => navigateTo("dashboard")}
          results={sortResults}
        />
      )}
    </main>
  );
}