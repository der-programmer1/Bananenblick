import { test, expect } from "@playwright/test";

test("stopping a pending camera request also stops a stream granted afterwards", async ({ page }) => {
  await page.addInitScript(() => {
    const state = window as any;
    state.testTrack = { readyState: "live", stop() { this.readyState = "ended"; } };
    state.testWorkersStarted = 0;
    Object.defineProperty(navigator, "mediaDevices", { value: {
      getUserMedia: () => new Promise((resolve) => {
        state.grantTestCamera = () => resolve({ getTracks: () => [state.testTrack] });
      }),
    } });
    state.Worker = class { constructor() { state.testWorkersStarted++; } };
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Kamera starten", exact: true }).first().click();
  await expect(page.getByRole("button", { name: "Kamera wird gestartet …" })).toBeDisabled();
  await page.getByRole("button", { name: "Scan beenden", exact: true }).click();
  await page.evaluate(() => (window as any).grantTestCamera());
  await expect.poll(() => page.evaluate(() => (window as any).testTrack.readyState)).toBe("ended");
  expect(await page.evaluate(() => (window as any).testWorkersStarted)).toBe(0);
  expect(await page.locator("video").evaluate((video: HTMLVideoElement) => video.srcObject)).toBeNull();
  await expect(page.locator(".detection-box")).toHaveCount(0);
  await expect(page.locator(".ready-summary")).toHaveText("Aktuell keine Banane sicher als essreif erkannt");
});

test("a model hash mismatch stops inference and explains the retry", async ({ page }) => {
  await page.route("**/models/banana-yolox-nano.onnx?v=*", (route) =>
    route.fulfill({ contentType: "application/octet-stream", body: "incomplete or stale model" }));
  await page.goto("/");
  await page.getByRole("button", { name: "Erst mal ein Beispiel ansehen" }).click();
  await page.locator(".example-tile").filter({ hasText: "Gelbe Banane" }).click();
  await expect(page.locator(".standby-content")).toContainText(
    "Das Bananenmodell passt noch nicht zur aktuellen Version", { timeout: 30000 });
  await expect(page.getByRole("button", { name: "Erneut starten", exact: true })).toBeVisible();
  await expect(page.locator(".detection-box")).toHaveCount(0);
  await expect(page.locator(".ready-summary")).toHaveText("Aktuell keine Banane sicher als essreif erkannt");
});

test("number selection, motion, uncertainty, disappearance, orientation and camera stop", async ({
  page,
  browserName,
}) => {
  const scriptErrors: string[] = [];
  page.on("pageerror", (error) => scriptErrors.push(error.message));
  await page.addInitScript((emulateMedia: boolean) => {
    const state = window as any;
    state.testPhase = "ripe";
    state.testStreams = [];
    state.testForcedWasmStarts = 0;
    // Controlled video clock for the media mock, including WebKit's autoplay policy.
    Object.defineProperties(HTMLVideoElement.prototype, {
      videoWidth: { get: () => 640 },
      videoHeight: { get: () => 480 },
      readyState: { get: () => 4 },
      currentTime: { get: () => Date.now() / 1000 },
    });
    HTMLVideoElement.prototype.play = async () => {};
    HTMLVideoElement.prototype.pause = () => {};
    HTMLVideoElement.prototype.requestVideoFrameCallback = (callback) =>
      window.setTimeout(() => callback(performance.now(), {} as VideoFrameCallbackMetadata), 10);
    HTMLVideoElement.prototype.cancelVideoFrameCallback = (handle) => clearTimeout(handle);
    Object.defineProperty(navigator, "mediaDevices", {
      value: {
        getUserMedia: async () => {
          const canvas = document.createElement("canvas");
          canvas.width = 640;
          canvas.height = 480;
          const ctx = canvas.getContext("2d")!;
          const interval = setInterval(() => {
            ctx.fillStyle = `rgb(${Date.now() % 255},120,80)`;
            ctx.fillRect(0, 0, 640, 480);
          }, 50);
          const stream = emulateMedia ? new MediaStream() : canvas.captureStream(20);
          if (emulateMedia) {
            const track = {
              readyState: "live",
              onended: null,
              onmute: null,
              onunmute: null,
              stop() {
                this.readyState = "ended";
                clearInterval(interval);
              },
              addEventListener() {},
            };
            Object.defineProperties(stream, {
              getTracks: { value: () => [track] },
              getVideoTracks: { value: () => [track] },
            });
            const drawImage = CanvasRenderingContext2D.prototype.drawImage;
            CanvasRenderingContext2D.prototype.drawImage = function (...args: any[]) {
              if (args[0] instanceof HTMLVideoElement) {
                this.fillRect(0, 0, 320, 320);
                return;
              }
              return (drawImage as any).apply(this, args);
            };
          }
          stream.getTracks()[0].addEventListener("ended", () => clearInterval(interval));
          state.testStreams.push(stream);
          return stream;
        },
      },
    });
    state.Worker = class {
      onmessage: ((event: any) => void) | null = null;
      terminated = false;
      forceWasm = false;
      terminate() {
        this.terminated = true;
      }
      postMessage(message: any) {
        if (message.type === "init") {
          this.forceWasm = message.preferGpu === false;
          if (this.forceWasm) state.testForcedWasmStarts++;
          setTimeout(
            () =>
              this.onmessage?.({
                data: {
                  type: "ready",
                  inputSize: 320,
                  provider: "wasm",
                  thresholds: { classification: 0.6 },
                },
              }),
            10,
          );
          return;
        }
        // A hanging GPU must be terminated and retried in a fresh WASM worker.
        if (!this.forceWasm) return;
        const phase = state.testPhase;
        let detections = [0, 1, 2, 3].map((n) => {
          const ripe = n % 2 === 1 && phase !== "uncertain";
          return {
            box: {
              x: 0.06 + n * 0.21 + Math.sin(message.frameId) * 0.005,
              y: 0.25,
              width: 0.25,
              height: 0.45,
            },
            ripeness: ripe ? "reif" : phase === "uncertain" ? "unsicher" : "unreif",
            predictedClass: ripe ? 1 : 0,
            ripeEvidence: ripe,
            confidence: 0.95,
            capturedAt: message.capturedAt,
          };
        });
        if (phase === "missing") detections = [];
        if (message.frameId % 2) detections.reverse();
        setTimeout(() => {
          if (!this.terminated)
            this.onmessage?.({ data: { ...message, type: "result", detections, inferenceMs: 20 } });
        }, 20);
      }
    };
  }, browserName === "webkit");
  await page.goto("/");
  expect(scriptErrors).toEqual([]);
  expect(await page.evaluate(() => Array.isArray((window as any).testStreams))).toBe(true);
  await page.getByRole("button", { name: "Kamera starten", exact: true }).first().click();
  expect(await page.evaluate(() => (window as any).testStreams.length)).toBe(1);
  await expect(page.locator(".ready-summary")).toHaveText("Jetzt essreif: Nr. 2 und 4", { timeout: 15000 });
  await expect(page.locator(".fruit-results li")).toHaveCount(4);
  await page.locator("#fruit-result-2").click();
  await expect(page.locator("#fruit-box-2")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#fruit-box-2")).toHaveClass(/ready.*selected/);
  await expect(page.locator("#fruit-box-1")).not.toContainText("✓");
  await page.setViewportSize({ width: 844, height: 390 });
  await expect(page.locator(".detection-box")).toHaveCount(4);
  await expect
    .poll(async () => (await page.locator(".detection-box").first().boundingBox())?.width ?? 0)
    .toBeGreaterThan(20);
  await page.evaluate(() => {
    (window as any).testPhase = "uncertain";
  });
  await expect(page.locator(".ready-summary")).toHaveText(
    "Aktuell keine Banane sicher als essreif erkannt",
  );
  await expect(page.locator(".detection-box.ready")).toHaveCount(0);
  await page.evaluate(() => {
    (window as any).testPhase = "missing";
  });
  await expect(page.locator(".detection-box")).toHaveCount(0);
  await expect(page.locator(".fruit-results li")).toHaveCount(0);
  await page.getByRole("button", { name: "Scan beenden", exact: true }).click();
  expect(
    await page.evaluate(() =>
      (window as any).testStreams.every((s: MediaStream) =>
        s.getTracks().every((t) => t.readyState === "ended"),
      ),
    ),
  ).toBe(true);
  await page.evaluate(() => {
    (window as any).testPhase = "ripe";
  });
  await page.getByRole("button", { name: "Kamera starten", exact: true }).first().click();
  await expect(page.locator(".ready-summary")).toHaveText("Jetzt essreif: Nr. 2 und 4", { timeout: 15000 });
  expect(await page.evaluate(() => (window as any).testForcedWasmStarts)).toBe(2);
  await page.getByRole("button", { name: "Scan beenden", exact: true }).click();
});

test("real ONNX example runs three passes and stays within both orientations", async ({ page }) => {
  await page.addInitScript(() => {
    const state = window as any;
    state.inferenceResults = [];
    state.inferenceEvents = [];
    const RealWorker = window.Worker;
    state.Worker = class extends RealWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener("message", (event) => {
          state.inferenceEvents.push({ type: event.data.type, message: event.data.message, provider: event.data.provider, at: Date.now() });
          if (event.data.type === "result")
            state.inferenceResults.push({ ...event.data, receivedAt: Date.now() });
        });
      }
    };
  });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const writes: string[] = [];
  page.on("request", (request) => {
    if (["POST", "PUT", "PATCH"].includes(request.method())) writes.push(request.url());
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Erst mal ein Beispiel ansehen" }).click();
  await page.locator(".example-tile").filter({ hasText: "Gelbe Banane" }).click();
  try {
    await expect(page.locator(".detection-box").first()).toBeVisible({ timeout: 45000 });
  } catch (error) {
    console.log("SCANNER_DIAGNOSTICS", await page.evaluate(() => ({
      events: (window as any).inferenceEvents,
      results: (window as any).inferenceResults,
      message: document.querySelector(".standby-content")?.textContent,
      status: document.querySelector(".source-badge")?.textContent,
    })));
    throw error;
  }
  await expect
    .poll(() => page.evaluate(() => (window as any).inferenceResults.length), { timeout: 20000 })
    .toBe(3);
  const results = await page.evaluate(() => (window as any).inferenceResults);
  await test
    .info()
    .attach("local-inference-results", {
      body: JSON.stringify(results),
      contentType: "application/json",
    });
  expect(results.every((r: any) => r.detections.length >= 1)).toBe(true);
  // Browser image resampling can change uncertainty. Only three certain observations recommend.
  if (results.every((r: any) => r.detections.length === 1 && r.detections[0].ripeEvidence))
    await expect(page.locator(".ready-summary")).toHaveText(/Jetzt essreif: Nr\./);
  else await expect(page.locator(".detection-box.ready")).toHaveCount(0);
  for (const viewport of [
    { width: 390, height: 844 },
    { width: 844, height: 390 },
  ]) {
    await page.setViewportSize(viewport);
    const box = await page.locator(".detection-box").first().boundingBox();
    const stage = await page.locator(".camera-stage").boundingBox();
    expect(box!.x).toBeGreaterThanOrEqual(stage!.x - 1);
    expect(box!.x + box!.width).toBeLessThanOrEqual(stage!.x + stage!.width + 1);
  }
  expect(errors).toEqual([]);
  expect(writes).toEqual([]);
  await page.getByRole("button", { name: "Scan beenden", exact: true }).click();
  await expect(page.locator(".detection-box")).toHaveCount(0);
});
