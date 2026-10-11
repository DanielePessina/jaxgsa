import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end suite for jaxgsa studio.
 *
 * By default it builds the app into `e2e/.build` (served at `/`) and runs it
 * with `vite preview`, so the tests see the production bundle. Set
 * `E2E_BASE_URL` to test an already running server (for example `npm run dev`)
 * instead. Locally the installed Google Chrome is used. In CI, or with
 * `E2E_CHANNEL=chromium`, the Playwright Chromium build is used.
 */
const PORT = 4173;

const external = process.env.E2E_BASE_URL;

const baseURL = external ?? `http://localhost:${PORT}/`;

const channel = process.env.E2E_CHANNEL ?? (process.env.CI ? "chromium" : "chrome");

export default defineConfig({
  testDir: ".",
  testMatch: "*.spec.ts",
  outputDir: "test-results",
  // The engine runs real numerics in WASM, so a full method run takes seconds.
  timeout: 180_000,
  expect: { timeout: 30_000 },
  fullyParallel: true,
  workers: process.env.CI ? 2 : 3,
  retries: process.env.CI ? 1 : 0,
  forbidOnly: Boolean(process.env.CI),
  reporter: process.env.CI ? [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]] : [["list"]],
  use: {
    ...devices["Desktop Chrome"],
    baseURL,
    acceptDownloads: true,
    viewport: { width: 1280, height: 900 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    channel: channel === "chromium" ? undefined : channel,
  },
  webServer: external
    ? undefined
    : {
        command: `npx vite build --outDir e2e/.build --emptyOutDir && npx vite preview --outDir e2e/.build --port ${PORT} --strictPort`,
        cwd: "..",
        env: { VITE_BASE: "/" },
        url: baseURL,
        reuseExistingServer: !process.env.CI,
        timeout: 300_000,
      },
});
