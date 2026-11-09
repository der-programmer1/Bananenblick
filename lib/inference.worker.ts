/// <reference lib="webworker" />
import { decodePredictions, DEFAULT_THRESHOLDS, type Thresholds } from "./detection";
import type * as ORT from "onnxruntime-web";

let ort: typeof ORT;
let session: ORT.InferenceSession | null = null;
let provider = "wasm";
let modelBytes: ArrayBuffer;
let inputSize = 320;
let thresholds = DEFAULT_THRESHOLDS;
let running = false;
const scope = self as unknown as DedicatedWorkerGlobalScope;
const send = (message: unknown) => scope.postMessage(message);

async function createSession(preferGpu: boolean) {
  if (preferGpu && "gpu" in navigator) {
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
  session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  provider = "wasm";
}
async function initialize(assetBase: string) {
  const base = new URL(assetBase);
  if (base.origin !== scope.location.origin) throw new Error("Modell und Laufzeit müssen von derselben Website stammen.");
  send({ type: "status", message: "Lokale Laufzeit wird geladen …" });
  // The all build uses the JSEP binary for both WebGPU and WASM (<25 MiB).
  ort = await import("onnxruntime-web/all");
  ort.env.wasm.wasmPaths = new URL("ort/", base).href;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  const card = await fetch(new URL("models/model-card.json", base), { cache: "no-cache" }).then((response) => {
    if (!response.ok) throw new Error("Modellbeschreibung konnte nicht geladen werden.");
    return response.json() as Promise<{ inputSize?: number; thresholds?: Partial<Thresholds> }>;
  });
  inputSize = card.inputSize ?? 320;
  thresholds = { ...DEFAULT_THRESHOLDS, ...card.thresholds };
  const response = await fetch(new URL("models/banana-yolox-nano.onnx", base));
  if (!response.ok)
    throw new Error("Das Bananenmodell konnte nicht geladen werden. Bitte erneut versuchen.");
  modelBytes = await response.arrayBuffer();
  send({ type: "status", message: "Bananenmodell wird vorbereitet …" });
  await createSession(true);
  send({ type: "ready", provider, inputSize, thresholds });
}
scope.onmessage = async (event) => {
  if (event.data.type === "init") {
    try {
      await initialize(event.data.assetBase);
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
      { ...thresholds, classification: classification ?? thresholds.classification } as Thresholds,
    );
    send({
      type: "result",
      sessionId,
      frameId,
      capturedAt,
      width,
      height,
      classification: classification ?? thresholds.classification,
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
