import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/search': 'http://localhost:8000',
      '/controls': 'http://localhost:8000'
    }
  }
})
