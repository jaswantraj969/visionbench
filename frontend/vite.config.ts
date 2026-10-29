/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev: the Vite server proxies /api to the backend, so the browser talks to one origin (no
// CORS). The proxy keeps Host as 127.0.0.1:8000 (changeOrigin), which the API's Host check
// accepts. Production: `visionbench serve` serves frontend/dist itself (same origin).
const backend = process.env.VISIONBENCH_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/api": { target: backend, changeOrigin: true } },
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 900 },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    css: false,
    coverage: {
      provider: "v8",
      reporter: ["text-summary", "text", "json-summary"],
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/**/*.test.{ts,tsx}", "src/test/**", "src/api/schema.d.ts", "src/main.tsx"],
    },
  },
});
