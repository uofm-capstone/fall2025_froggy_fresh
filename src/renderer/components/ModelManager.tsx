import React, { useState, useEffect } from 'react';
const { ipcRenderer } = window.require('electron');

interface Model {
  id: string;
  name: string;
  type: string;
  source: 'default' | 'uploaded' | 'preloaded';
  file: string;
}

const ModelManager: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [activeModel, setActiveModel] = useState<string>('frog_detector');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  // Fetch available models on component mount
  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const result = await ipcRenderer.invoke('list-models');
      
      if (result.success) {
        setModels(result.models || []);
        // Set active model if available
        if (result.activeModel) {
          setActiveModel(result.activeModel);
        }
      } else {
        setError(result.error || 'Failed to fetch models');
      }
    } catch (err) {
      setError('Error communicating with backend');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelSwitch = async (modelId: string) => {
    try {
      setIsLoading(true);
      const result = await ipcRenderer.invoke('switch-model', modelId);
      
      if (result.success) {
        setActiveModel(modelId);
        setUploadStatus(`Switched to model: ${models.find(m => m.id === modelId)?.name || modelId}`);
      } else {
        setError(result.error || 'Failed to switch model');
      }
    } catch (err) {
      setError('Error switching model');
      console.error(err);
    } finally {
      setIsLoading(false);
      // Clear status after 3 seconds
      setTimeout(() => setUploadStatus(null), 3000);
    }
  };

  const handleModelUpload = async () => {
    try {
      setIsLoading(true);
      setError(null);
      setUploadStatus('Uploading model...');
      
      const result = await ipcRenderer.invoke('upload-model');
      
      if (result.success) {
        setUploadStatus('Model uploaded successfully!');
        // Refresh models list
        fetchModels();
      } else {
        setError(result.error || 'Failed to upload model');
        setUploadStatus(null);
      }
    } catch (err) {
      setError('Error uploading model');
      setUploadStatus(null);
      console.error(err);
    } finally {
      setIsLoading(false);
      // Clear status after 3 seconds
      setTimeout(() => setUploadStatus(null), 3000);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-4 mb-6">
      <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">Model Management</h2>
      
      {/* Model Selector */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Active CNN Model
        </label>
        <div className="flex gap-2">
          <select 
            value={activeModel}
            onChange={(e) => handleModelSwitch(e.target.value)}
            disabled={isLoading || models.length === 0}
            className="block w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 text-sm"
          >
            {models.length === 0 && (
              <option value="">No models available</option>
            )}
            {models.map(model => (
              <option key={model.id} value={model.id}>
                {model.name} ({model.source})
              </option>
            ))}
          </select>
        </div>
      </div>
      
      {/* Upload New Model */}
      <div className="mb-4">
        <button
          onClick={handleModelUpload}
          disabled={isLoading}
          className="px-4 py-2 bg-blue-600 dark:bg-blue-700 text-white rounded-md hover:bg-blue-700 dark:hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
        >
          {isLoading ? 'Processing...' : 'Upload Custom Model'}
        </button>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Supported formats: .h5, .pt, .onnx
        </p>
      </div>
      
      {/* Status Messages */}
      {uploadStatus && (
        <div className="mt-2 p-2 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 rounded-md text-sm">
          {uploadStatus}
        </div>
      )}
      
      {error && (
        <div className="mt-2 p-2 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 rounded-md text-sm">
          Error: {error}
        </div>
      )}
      
      {/* Model Info */}
      {activeModel && models.length > 0 && (
        <div className="mt-4 border-t pt-4 dark:border-gray-700">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Current Model Details
          </h3>
          {models.find(m => m.id === activeModel) ? (
            <div className="bg-gray-50 dark:bg-gray-900 p-3 rounded-md text-sm">
              <p><span className="font-medium">Name:</span> {models.find(m => m.id === activeModel)?.name}</p>
              <p><span className="font-medium">Type:</span> {models.find(m => m.id === activeModel)?.type}</p>
              <p><span className="font-medium">Source:</span> {models.find(m => m.id === activeModel)?.source}</p>
            </div>
          ) : (
            <p className="text-sm text-gray-500 dark:text-gray-400">Model details not available</p>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelManager;