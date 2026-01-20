import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { searchAPI, controlsAPI } from './services/api';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [rescanning, setRescanning] = useState(false);
  const [toast, setToast] = useState(null);
  const [serverStatus, setServerStatus] = useState('unknown');

  // Check server health on mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await controlsAPI.healthCheck();
      console.log('Health check response:', response);
      setServerStatus('online');
    } catch (error) {
      console.error('Health check failed:', error);
      setServerStatus('offline');
      // Don't show toast on initial load if server is offline
      // User will see the status indicator
    }
  };

  const showToast = (message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 5000);
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) {
      showToast('Please enter a search query', 'error');
      return;
    }

    setLoading(true);
    try {
      const data = await searchAPI.search(query);
      console.log('Search response:', data);
      
      // Handle results - could be data.results or just data as an array
      const resultsList = data.results || (Array.isArray(data) ? data : []);
      setResults(resultsList);
      
      if (resultsList && resultsList.length > 0) {
        showToast(`✨ Found ${resultsList.length} ${resultsList.length === 1 ? 'result' : 'results'}`, 'success');
      } else {
        showToast('No results found. Try a different query!', 'info');
      }
    } catch (error) {
      console.error('Search failed:', error);
      console.error('Error details:', error.response?.data || error.message);
      
      if (error.response) {
        showToast(`Search error: ${error.response.status} - ${error.response.statusText}`, 'error');
      } else if (error.request) {
        showToast('No response from server. Is the backend running?', 'error');
      } else {
        showToast(`Search failed: ${error.message}`, 'error');
      }
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleRescan = async () => {
    setRescanning(true);
    try {
      const data = await controlsAPI.rescan();
      console.log('Rescan response:', data);
      
      // Check if we got a successful response
      if (data && data.status) {
        showToast(`✅ ${data.status} - ${data.files || 'Files processed'}`, 'success');
      } else if (data) {
        showToast(`Rescan completed! ${data.files || 'Files processed'}`, 'success');
      } else {
        showToast('Rescan completed but received unexpected response', 'info');
      }
    } catch (error) {
      console.error('Rescan failed:', error);
      console.error('Error details:', error.response?.data || error.message);
      
      // More specific error messages
      if (error.response) {
        showToast(`Rescan error: ${error.response.status} - ${error.response.statusText}`, 'error');
      } else if (error.request) {
        showToast('No response from server. Is the backend running?', 'error');
      } else {
        showToast(`Rescan failed: ${error.message}`, 'error');
      }
    } finally {
      setRescanning(false);
    }
  };

  const getFileName = (path) => {
    return path.split('/').pop() || path;
  };

  const formatScore = (score) => {
    return (score * 100).toFixed(1) + '%';
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <motion.div
          className="app-logo"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <span className="app-logo-icon">🔮</span>
        </motion.div>
        <motion.h1
          className="app-title"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          Insight
        </motion.h1>
        <motion.p
          className="app-subtitle"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          Semantic search through your photos & videos using AI
        </motion.p>
      </header>

      {/* Controls Panel */}
      <motion.div
        className="controls-panel"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <div className="controls-title">
          ⚙️ System Controls
        </div>
        <div className="controls-actions">
          <button
            className="control-button primary"
            onClick={handleRescan}
            disabled={rescanning}
          >
            {rescanning ? '⏳ Rescanning...' : '🔄 Rescan Media'}
          </button>
          <button
            className="control-button"
            onClick={checkHealth}
          >
            {serverStatus === 'online' ? '✅ Server Online' : '❌ Server Offline'}
          </button>
        </div>
        {rescanning && (
          <div className="status-badge loading" style={{ marginTop: '1rem' }}>
            <span className="status-icon">⏳</span>
            Scanning files and updating vector database...
          </div>
        )}
      </motion.div>

      {/* Search Section */}
      <motion.div
        className="search-section"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <form onSubmit={handleSearch} className="search-bar">
          <div className="search-input-wrapper">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              className="search-input"
              placeholder="Describe what you're looking for... (e.g., 'sunset over mountains', 'person walking a dog')"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
            />
            <button
              type="submit"
              className="search-button"
              disabled={loading}
            >
              {loading ? '⏳ Searching...' : '✨ Search'}
            </button>
          </div>
        </form>

        {loading && (
          <div className="status-badge loading">
            <span className="status-icon">⏳</span>
            Using AI to understand your query...
          </div>
        )}
      </motion.div>

      {/* Results Section */}
      <AnimatePresence mode="wait">
        {results.length > 0 && (
          <motion.div
            className="results-section"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="results-header">
              <h2 className="results-title">Search Results</h2>
              <span className="results-count">
                {results.length} {results.length === 1 ? 'match' : 'matches'}
              </span>
            </div>

            <div className="results-grid">
              {results.map((result, index) => (
                <motion.div
                  key={index}
                  className="result-card"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="result-image-container">
                    <img
                      src={`file://${result.payload?.path || result.path}`}
                      alt={getFileName(result.payload?.path || result.path)}
                      className="result-image"
                      onError={(e) => {
                        e.target.style.display = 'none';
                        e.target.parentElement.innerHTML = `
                          <div style="display: flex; align-items: center; justify-content: center; height: 100%; font-size: 3rem; opacity: 0.3;">
                            🖼️
                          </div>
                        `;
                      }}
                    />
                    <div className="result-score">
                      {formatScore(result.score)}
                    </div>
                  </div>
                  <div className="result-info">
                    <div className="result-filename">
                      {getFileName(result.payload?.path || result.path)}
                    </div>
                    <div className="result-path" title={result.payload?.path || result.path}>
                      {result.payload?.path || result.path}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {!loading && results.length === 0 && query && (
          <motion.div
            className="empty-state"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="empty-icon">🔍</div>
            <h3 className="empty-title">No Results Found</h3>
            <p className="empty-description">
              Try adjusting your search query or make sure media files are indexed
            </p>
          </motion.div>
        )}

        {!loading && results.length === 0 && !query && (
          <motion.div
            className="empty-state"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="empty-icon">✨</div>
            <h3 className="empty-title">Ready to Search</h3>
            <p className="empty-description">
              Enter a description above to find matching images and videos
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toast Notifications */}
      <AnimatePresence>
        {toast && (
          <motion.div
            className={`toast ${toast.type}`}
            initial={{ x: 400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 400, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          >
            {toast.message}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;

