import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 60000, // 60 seconds for long operations
});

export const searchAPI = {
  /**
   * Search for images/videos using a text query
   * @param {string} query - The search query
   * @returns {Promise} - Search results with file paths and scores
   */
  search: async (query) => {
    try {
      const response = await api.get('/search', {
        params: { query }
      });
      return response.data;
    } catch (error) {
      console.error('Search error:', error);
      throw error;
    }
  }
};

export const controlsAPI = {
  /**
   * Trigger a rescan of the media folders
   * @returns {Promise} - Rescan status and files changed count
   */
  rescan: async () => {
    try {
      const response = await api.get('/controls/rescan');
      return response.data;
    } catch (error) {
      console.error('Rescan error:', error);
      throw error;
    }
  },

  /**
   * Health check endpoint
   * @returns {Promise} - Server status
   */
  healthCheck: async () => {
    try {
      const response = await api.get('/');
      return response.data;
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }
};

export default api;
