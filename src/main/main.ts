import { app, BrowserWindow } from 'electron';
import * as path from 'path';

let mainWindow: BrowserWindow;

app.whenReady().then(() => {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'), // Optional if you use a preload script
      nodeIntegration: true,  // Set this to true only if you're sure of the security
    },
  });

  mainWindow.loadURL('http://localhost:5173'); // Vite's dev server URL
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
