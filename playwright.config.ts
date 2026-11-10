import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "tests/browser",
  timeout: 60000,
  workers: 1,
  reporter: [["list"], ["json", { outputFile: "outputs/browser-results.json" }]],
  use: { baseURL: "http://127.0.0.1:4173", screenshot: "only-on-failure" },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 4173",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: true,
  },
  projects: [
    { name: "chromium-mobile-emulation", use: { ...devices["Pixel 7"], browserName: "chromium" } },
    { name: "webkit-mobile-emulation", use: { ...devices["iPhone 13"], browserName: "webkit" } },
  ],
});
