/// <reference lib="webworker" />
import { decodePredictions, DEFAULT_THRESHOLDS, type Thresholds } from "./detection";
import type * as ORT from "onnxruntime-web";

let ort: typeof ORT;
let session: ORT.InferenceSession | null = null;
let provider = "wasm";
let modelBytes: ArrayBuffer;
let inputSize = 320;
let modelSha256: string | undefined;
let thresholds = DEFAULT_THRESHOLDS;
let running = false;
const scope = self as unknown as DedicatedWorkerGlobalScope;
const send = (message: unknown) => scope.postMessage(message);

async function createSession(preferGpu: boolean) {
  if (preferGpu && "gpu" in navigator) {
    send({ type: "status", phase: "session", provider: "webgpu", message: "Lokale Erkennung wird vorbereitet …" });
    try {
      session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      });
      provider = "webgpu";
      return;
    } catch {
      send({ type: "status", message: "WebGPU nicht verfügbar. WebAssembly wird vorbereitet." });
    }
  }
  send({ type: "status", phase: "session", provider: "wasm", message: "Lokale Erkennung wird vorbereitet …" });
  session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  provider = "wasm";
}
async function initialize(assetBase: string, preferGpu = true) {
  const base = new URL(assetBase);
  if (base.origin !== scope.location.origin) throw new Error("Modell und Laufzeit müssen von derselben Website stammen.");
  send({ type: "status", phase: "download", message: "Lokale Laufzeit wird geladen …" });
  // The all build uses the JSEP binary for both WebGPU and WASM (<25 MiB).
  ort = await import("onnxruntime-web/all");
  const runtimeVersion = ort.env.versions.web ?? ort.env.versions.common;
  const wasmUrl = new URL("ort/ort-wasm-simd-threaded.jsep.wasm", base);
  const mjsUrl = new URL("ort/ort-wasm-simd-threaded.jsep.mjs", base);
  wasmUrl.searchParams.set("v", runtimeVersion);
  mjsUrl.searchParams.set("v", runtimeVersion);
  ort.env.wasm.wasmPaths = { mjs: mjsUrl.href, wasm: wasmUrl.href };
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  // Download the large WASM binary explicitly. Network time must not be mistaken
  // for a stalled GPU session on a slower mobile connection.
  const [card, wasmBytes] = await Promise.all([
    fetch(new URL("models/model-card.json", base), { cache: "no-cache" }).then((response) => {
      if (!response.ok) throw new Error("Modellbeschreibung konnte nicht geladen werden.");
      return response.json() as Promise<{ inputSize?: number; thresholds?: Partial<Thresholds>; sha256?: string }>;
    }),
    fetch(wasmUrl).then((response) => {
      if (!response.ok) throw new Error("Lokale Laufzeit konnte nicht geladen werden.");
      return response.arrayBuffer();
    }),
  ]);
  ort.env.wasm.wasmBinary = wasmBytes;
  inputSize = card.inputSize ?? 320;
  modelSha256 = card.sha256;
  thresholds = { ...DEFAULT_THRESHOLDS, ...card.thresholds };
  const modelUrl = new URL("models/banana-yolox-nano.onnx", base);
  if (card.sha256) modelUrl.searchParams.set("v", card.sha256);
  const response = await fetch(modelUrl, { cache: "no-cache" });
  if (!response.ok)
    throw new Error("Das Bananenmodell konnte nicht geladen werden. Bitte erneut versuchen.");
  modelBytes = await response.arrayBuffer();
  if (card.sha256) {
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", modelBytes)))
      .map((byte) => byte.toString(16).padStart(2, "0")).join("");
    if (digest !== card.sha256) throw new Error("Das Bananenmodell passt noch nicht zur aktuellen Version. Bitte starte den Scan erneut.");
  }
  send({ type: "status", phase: "session", message: "Bananenmodell wird vorbereitet …" });
  await createSession(preferGpu);
  send({ type: "ready", provider, inputSize, thresholds, modelSha256 });
}
scope.onmessage = async (event) => {
  if (event.data.type === "init") {
    try {
      await initialize(event.data.assetBase, event.data.preferGpu !== false);
    } catch (error) {
      send({
        type: "error",
        message:
          error instanceof Error && error.message.startsWith("Das Bananenmodell")
            ? error.message
            : "Das lokale Modell konnte nicht gestartet werden. Prüfe deine Verbindung und versuche es mit einem aktuellen Safari- oder Chrome-Browser erneut.",
      });
    }
    return;
  }
  if (event.data.type !== "frame" || !session || running) return;
  running = true;
  const { rgba, sessionId, frameId, capturedAt, width, height, classification } = event.data;
  const effectiveClassification = Math.max(thresholds.classification, classification ?? thresholds.classification);
  const start = performance.now();
  let tensor: ORT.Tensor | undefined;
  let output: ORT.InferenceSession.OnnxValueMapType | undefined;
  try {
    const pixels = new Uint8ClampedArray(rgba);
    const plane = inputSize * inputSize;
    const input = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      input[i] = pixels[i * 4 + 2];
      input[plane + i] = pixels[i * 4 + 1];
      input[plane * 2 + i] = pixels[i * 4];
    }
    tensor = new ort.Tensor("float32", input, [1, 3, inputSize, inputSize]);
    try {
      output = await session.run({ [session.inputNames[0]]: tensor });
    } catch (error) {
      if (provider !== "webgpu") throw error;
      await session.release();
      await createSession(false);
      send({ type: "provider", provider });
      output = await session!.run({ [session!.inputNames[0]]: tensor });
    }
    const predictions = output[session!.outputNames[0]];
    const detections = decodePredictions(
      predictions.data as Float32Array,
      inputSize,
      width,
      height,
      capturedAt,
      { ...thresholds, classification: effectiveClassification } as Thresholds,
    );
    send({
      type: "result",
      sessionId,
      frameId,
      capturedAt,
      width,
      height,
      classification: effectiveClassification,
      inferenceMs: performance.now() - start,
      detections,
    });
  } catch (error) {
    send({
      type: "error",
      message: "Die lokale Auswertung wurde unterbrochen. Starte den Scan erneut.",
    });
  } finally {
    tensor?.dispose();
    if (output) for (const value of Object.values(output)) if ("dispose" in value) value.dispose();
    running = false;
  }
};
