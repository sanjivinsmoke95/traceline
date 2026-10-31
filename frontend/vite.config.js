import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The API runs on :8000; proxying keeps the browser on one origin (no CORS, Range requests for video work unchanged).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': process.env.API_PROXY || 'http://127.0.0.1:8000' } },
});
