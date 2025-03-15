import React from 'react';
import ReactDOM from 'react-dom/client';
import App from '../renderer/app/page';
import "../renderer/app/globals.css";
import { ThemeProvider } from "next-themes";

const root = document.getElementById('root') as HTMLElement;
ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <App />
    </ThemeProvider>
  </React.StrictMode>
);
