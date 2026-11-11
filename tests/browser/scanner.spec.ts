import { test, expect } from "@playwright/test";

test("a cold WASM session gets its own initialization timeout without a GPU restart", async ({ page }) => {
  await page.addInitScript(() => {
    const state = window as any;
    state.wasmStarts = 0;
    state.wasmAborts = 0;
    state.wasmResults = 0;
    state.Worker = class {
      onmessage: ((event: any) => void) | null = null;
      terminated = false;
      terminate() { this.terminated = true; state.wasmAborts++; }
      postMessage(message: any) {
        if (message.type === "init") {
          state.wasmStarts++;
          setTimeout(() => this.onmessage?.({ data: { type: "status", phase: "session", provider: "wasm", message: "Lokale Erkennung wird vorbereitet …" } }), 10);
          setTimeout(() => { if (!this.terminated) this.onmessage?.({ data: {
            type: "ready", inputSize: 320, provider: "wasm", thresholds: { classification: .8 },
          } }); }, 13000);
          return;
        }
        setTimeout(() => {
          if (!this.terminated) {
            state.wasmResults++;
            this.onmessage?.({ data: { ...message, type: "result", detections: [], inferenceMs: 20 } });
          }
        }, 20);
      }
    };
  });
  await page.goto("./");
  await page.getByRole("button", { name: "Erst mal ein Beispiel ansehen" }).click();
  await page.locator(".example-tile").filter({ hasText: "Gelbe Banane" }).click();
  await expect.poll(() => page.evaluate(() => (window as any).wasmResults), { timeout: 20000 }).toBe(3);
  expect(await page.evaluate(() => (window as any).wasmStarts)).toBe(1);
  expect(await page.evaluate(() => (window as any).wasmAborts)).toBe(0);
  await expect(page.locator(".runtime-indicator")).toContainText("WebAssembly");
  await page.getByRole("button", { name: "Scan beenden", exact: true }).click();
});

for (const phase of ["startup", "inference"] as const) {
  test(`GPU fallback during ${phase} gives WASM its full session budget`, async ({ page }) => {
    await page.addInitScript((fallbackPhase) => {
      const state = window as any;
      state.testNow = 0;
      state.workerStarts = 0;
      state.workerAborts = 0;
      state.workerResults = 0;
      state.inWasmSession = false;
      Object.defineProperty(performance, "now", { value: () => state.testNow });
      state.Worker = class {
        onmessage: ((event: any) => void) | null = null;
        pending: any = null;
        terminated = false;
        constructor() { state.testWorker = this; }
        terminate() { this.terminated = true; state.workerAborts++; }
        send(data: any) { if (!this.terminated) this.onmessage?.({ data }); }
        enterWasm() {
          state.testNow = 11000;
          state.inWasmSession = true;
          this.send({ type: "status", phase: "session", provider: "wasm", message: "Lokale Erkennung wird vorbereitet …" });
        }
        ready(provider: string) {
          this.send({ type: "ready", provider, inputSize: 320, thresholds: { classification: .8 } });
        }
        finishWasm() {
          if (fallbackPhase === "startup") this.ready("wasm");
          else {
            this.send({ type: "provider", provider: "wasm" });
            this.result(this.pending);
          }
        }
        result(message: any) {
          if (this.terminated) return;
          state.workerResults++;
          this.send({ ...message, type: "result", inferenceMs: 20, detections: [] });
        }
        postMessage(message: any) {
          if (message.type === "init") {
            state.workerStarts++;
            this.send({ type: "status", phase: "session", provider: "webgpu", message: "Lokale Erkennung wird vorbereitet …" });
            if (fallbackPhase === "inference") this.ready("webgpu");
            return;
          }
          if (fallbackPhase === "inference" && !state.inWasmSession) {
            this.pending = message;
            this.enterWasm();
            return;
          }
          setTimeout(() => this.result(message), 20);
        }
      };
    }, phase);
    await page.goto("./");
    await page.getByRole("button", { name: "Erst mal ein Beispiel ansehen" }).click();
    await page.locator(".example-tile").filter({ hasText: "Gelbe Banane" }).click();
    await expect.poll(() => page.evaluate(() => (window as any).workerStarts)).toBe(1);
    if (phase === "startup") await page.evaluate(() => (window as any).testWorker.enterWasm());
    await expect.poll(() => page.evaluate(() => (window as any).inWasmSession)).toBe(true);
    // Simulate 24 seconds inside WASM after 11 seconds in WebGPU. The real
    // watchdog ticks normally; no 35-second wall-clock wait is necessary.
    await page.evaluate(() => { (window as any).testNow = 35000; });
    await page.waitForTimeout(300);
    expect(await page.evaluate(() => (window as any).workerStarts)).toBe(1);
    expect(await page.evaluate(() => (window as any).workerAborts)).toBe(0);
    await page.evaluate(() => (window as any).testWorker.finishWasm());
    await expect.poll(() => page.evaluate(() => (window as any).workerResults)).toBeGreaterThanOrEqual(3);
    await expect(page.locator(".runtime-indicator")).toContainText("WebAssembly");
    await page.getByRole("button", { name: "Scan beenden", exact: true }).click();
  });
}

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
  await page.goto("./");
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
  await page.goto("./");
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
                  provider: this.forceWasm ? "wasm" : "webgpu",
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
        const dense = phase === "dense";
        let detections = Array.from({ length: dense ? 20 : 4 }, (_, n) => {
          const ripe = n % 2 === 1 && phase !== "uncertain";
          return {
            box: {
              x: (dense ? .3 + (n % 4) * .015 : 0.06 + n * 0.21) + Math.sin(message.frameId) * 0.005,
              y: dense ? .2 + (n % 3) * .02 : 0.25,
              width: 0.25,
              height: dense ? .6 : 0.45,
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
  await page.goto("./");
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
  await page.evaluate(() => { (window as any).testPhase = "dense"; });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator(".detection-label")).toHaveCount(20);
  const badges = await page.locator(".detection-label").evaluateAll((nodes) => nodes.map((node) => {
    const r = node.getBoundingClientRect();
    return { x: r.x, y: r.y, right: r.right, bottom: r.bottom, text: node.textContent };
  }));
  for (const [index, a] of badges.entries()) for (const b of badges.slice(index + 1))
    expect(a.right <= b.x + .5 || b.right <= a.x + .5 || a.bottom <= b.y + .5 || b.bottom <= a.y + .5,
      `Number badges ${a.text} and ${b.text} must remain readable`).toBe(true);
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

test("slow runtime download and three actual ONNX passes work in both orientations", async ({ page, browser, browserName }) => {
  let runtimeDownloads = 0;
  await page.route("**/ort/*.wasm?*", async (route) => {
    runtimeDownloads++;
    await new Promise((resolve) => setTimeout(resolve, 13000));
    await route.continue();
  });
  await page.addInitScript(() => {
    const state = window as any;
    state.inferenceResults = [];
    state.inferenceEvents = [];
    const RealWorker = window.Worker;
    state.Worker = class extends RealWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener("message", (event) => {
          state.inferenceEvents.push({ type: event.data.type, message: event.data.message, provider: event.data.provider, modelSha256: event.data.modelSha256, at: Date.now() });
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
  await page.goto("./");
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
  const ready = await page.evaluate(() => (window as any).inferenceEvents.find((event: any) => event.type === "ready"));
  const card = await (await page.request.get(new URL("models/model-card.json", page.url()).href)).json();
  expect(ready.modelSha256).toBe(card.sha256);
  await test.info().attach("verified-runtime", {
    body: JSON.stringify({ modelSha256: ready.modelSha256, inputSize: card.inputSize,
      browserName, browserVersion: browser.version(), provider: ready.provider, url: page.url() }),
    contentType: "application/json",
  });
  await test
    .info()
    .attach("local-inference-results", {
      body: JSON.stringify(results),
      contentType: "application/json",
    });
  expect(results.every((r: any) => r.detections.length >= 1)).toBe(true);
  expect(runtimeDownloads).toBe(1);
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
