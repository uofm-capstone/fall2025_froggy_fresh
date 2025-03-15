import React from 'react';
import ReactDOM from 'react-dom/client';
import App from '../renderer/app/page';
import "../renderer/app/globals.css";

const root = document.getElementById('root') as HTMLElement;
ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
