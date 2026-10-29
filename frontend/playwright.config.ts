import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "@playwright/test";

// The e2e test runs against a REAL backend (`scripts/e2e_backend.py`): a fresh temporary data
// root, a generated synthetic dataset (never EuroSAT), the built frontend served by the API.
const repo = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const winPython = join(repo, ".venv", "Scripts", "python.exe");
const posixPython = join(repo, ".venv", "bin", "python");
const python =
  process.env.VISIONBENCH_PYTHON ??
  (existsSync(winPython) ? winPython : existsSync(posixPython) ? posixPython : "python");
const port = Number(process.env.E2E_PORT ?? 8765);
const root = process.env.E2E_ROOT ?? mkdtempSync(join(tmpdir(), "visionbench-e2e-"));
process.env.E2E_ROOT = root;

export default defineConfig({
  testDir: "./e2e",
  timeout: 15 * 60 * 1000,
  expect: { timeout: 30_000 },
  workers: 1,
  fullyParallel: false,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "retain-on-failure",
    viewport: { width: 1280, height: 900 },
  },
  webServer: {
    command: `"${python}" "${join(repo, "scripts", "e2e_backend.py")}" --root "${join(root, "run")}" --port ${port}`,
    url: `http://127.0.0.1:${port}/api/health`,
    reuseExistingServer: false,
    timeout: 180_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
