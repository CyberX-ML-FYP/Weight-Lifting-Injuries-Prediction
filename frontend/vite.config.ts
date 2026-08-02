import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Hip & Knee Analytics — frontend build config.
//
// IMPORTANT: this proxy is a *frontend dev-server* configuration only. It
// does not modify the FastAPI backend (src/api/hip_knee_api.py) in any way —
// it simply forwards browser requests from this dev server to the existing,
// unmodified backend running on http://localhost:8000, which avoids needing
// to add CORS middleware to the backend (a backend code change that is
// explicitly out of scope for this task).
//
// Frontend code calls relative paths like `/api/predict` and `/api/health`;
// Vite rewrites `/api/*` -> `http://localhost:8000/*` before the request
// ever reaches the browser's network stack, so no cross-origin request is
// ever made by the browser itself.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  preview: {
    port: 4173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
