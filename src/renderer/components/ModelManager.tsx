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
  const [isDeleting, setIsDeleting] = useState<boolean>(false);
  const [isSwitching, setIsSwitching] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  // Fetch available models on component mount
  useEffect(() => {
    fetchModels();
    
    // Try to restore last selected model
    const lastSelected = localStorage.getItem('lastSelectedModel');
    if (lastSelected) {
      setActiveModel(lastSelected);
    }
  }, []);

  useEffect(() => {
    // Remember last selected model
    if (activeModel) {
      localStorage.setItem('lastSelectedModel', activeModel);
    }
  }, [activeModel]);

  const fetchModels = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      console.log('Requesting models...');
      const result = await ipcRenderer.invoke('list-models');
      console.log('Models response received:', result);
      
      // IMPORTANT: Detailed validation to catch all possible issues
      if (!result) {
        throw new Error('Empty response received');
      }
      
      if (typeof result !== 'object') {
        throw new Error(`Invalid response type: ${typeof result}`);
      }
      
      if (result.success === false) {
        throw new Error(result.error || 'Server reported error');
      }
      
      if (!Array.isArray(result.models)) {
        throw new Error('Response missing models array');
      }
      
      // If we get here, the response is valid
      console.log(`Found ${result.models.length} models`);
      setModels(result.models);
      
      // Set active model if available
      if (result.activeModel) {
        setActiveModel(result.activeModel);
      }
    } catch (err: unknown) {
      console.error('Error in fetchModels:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelSwitch = async (modelId: string) => {
    try {
      setIsSwitching(true);
      setIsLoading(true);
      const result = await ipcRenderer.invoke('switch-model', modelId);
      
      if (result.success) {
        setActiveModel(modelId);
        setUploadStatus(`Switched to model: ${models.find(m => m.id === modelId)?.name || modelId}`);
      } else {
        setError(result.error || 'Failed to switch model');
      }
    } catch (err: unknown) {
      console.error('Error switching model:', err);
      setError(`Error switching model: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsSwitching(false);
      setIsLoading(false);
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
    } catch (err: unknown) {
      console.error('Error uploading model:', err);
      setError(`Error uploading: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setUploadStatus(null);
    } finally {
      setIsLoading(false);
      // Clear status after 3 seconds
      setTimeout(() => setUploadStatus(null), 3000);
    }
  };

  // Add this function to the ModelManager component
  const handleModelDelete = async (modelId: string) => {
    // Show confirmation dialog
    if (!confirm(`Are you sure you want to delete this model? This action cannot be undone.`)) {
      return; // User cancelled
    }

    try {
      setIsDeleting(true);
      setIsLoading(true);
      setError(null);
      setUploadStatus('Deleting model...');
      
      const result = await ipcRenderer.invoke('delete-model', modelId);
      
      if (result.success) {
        setUploadStatus('Model deleted successfully!');
        // Refresh models list
        fetchModels();
      } else {
        setError(result.error || 'Failed to delete model');
        setUploadStatus(null);
      }
    } catch (err: unknown) {
      console.error('Error deleting model:', err);
      setError(`Error deleting: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setUploadStatus(null);
    } finally {
      setIsDeleting(false);
      setIsLoading(false);
      // Clear status after 3 seconds
      setTimeout(() => setUploadStatus(null), 3000);
    }
  };

  // Add to the component
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+U or Cmd+U to upload model
      if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        if (!isLoading) handleModelUpload();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isLoading]);

  return (
    <div className="apple-card">
      {/* Model Selector */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-[var(--apple-text)] mb-2">
          Active CNN Model
        </label>
        <div className="flex gap-2">
          <select 
            value={activeModel}
            onChange={(e) => handleModelSwitch(e.target.value)}
            disabled={isLoading || models.length === 0}
            className="apple-select w-full"
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
          className="apple-button w-full"
        >
          {isLoading ? 'Processing...' : 'Upload Custom Model'}
        </button>
        <p className="mt-1 text-sm text-[var(--apple-text-secondary)]">
          Supported formats: .h5, .pt, .onnx
        </p>
      </div>
      
      {/* Status Messages */}
      {uploadStatus && (
        <div className="mt-2 p-2 bg-green-100/20 text-green-300 rounded-md text-sm">
          {uploadStatus}
        </div>
      )}
      
      {error && (
        <div className="mt-2 p-2 bg-red-100/20 text-red-300 rounded-md text-sm">
          <p>Error: {error}</p>
          <button 
            className="text-xs underline mt-1"
            onClick={() => fetchModels()}
          >
            Retry
          </button>
        </div>
      )}
      
      {/* Model Info */}
      {activeModel && models.length > 0 && (
        <div className="mt-4 border-t border-[var(--apple-border)] pt-4">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-sm font-medium text-[var(--apple-text)]">
              Current Model Details
            </h3>
            
            {/* Move delete button next to the heading for quicker access */}
            {models.find(m => m.id === activeModel)?.source === 'uploaded' && (
              <button
                onClick={() => handleModelDelete(activeModel)}
                disabled={isLoading}
                className="text-xs text-red-400 hover:text-red-300 
                            flex items-center px-2 py-1 rounded-md 
                            bg-red-500/10 border border-red-500/20
                            transition-colors duration-200 disabled:opacity-50"
                title="Delete this model"
              >
                {isDeleting ? 'Deleting...' : 'Delete'}
              </button>
            )}
          </div>
          
          {models.find(m => m.id === activeModel) ? (
            <div className="bg-[var(--apple-bg-secondary)] p-3 rounded-md text-sm text-[var(--apple-text)]">
              <div className="grid grid-cols-2 gap-y-1">
                <p><span className="font-medium">Name:</span></p>
                <p>{models.find(m => m.id === activeModel)?.name}</p>
                
                <p><span className="font-medium">Type:</span></p>
                <p>{models.find(m => m.id === activeModel)?.type}</p>
                
                <p><span className="font-medium">Source:</span></p>
                <p>{models.find(m => m.id === activeModel)?.source}</p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-[var(--apple-text-secondary)]">Model details not available</p>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelManager;