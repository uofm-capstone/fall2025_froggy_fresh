import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';  // <-- Add this line

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: true,  // Automatically open browser
  },
  build: {
    outDir: path.resolve(__dirname, 'dist'), // Output directory
  },
  resolve: {
    alias: {
      '@renderer': path.resolve(__dirname, 'src/renderer'), // Alias for the renderer folder
      '@app': path.resolve(__dirname, 'src/renderer/app'), // Alias for the app folder
    },
  },
});
