/** Actual ORT Web WASM check in Node. This does not simulate a phone camera. */
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";
// Direct typed entry avoids Vinext's optional `sharp` ambient stub.
import sharp from "sharp/lib/index.js";
import * as ort from "onnxruntime-web/all";
import { decodePredictions, DEFAULT_THRESHOLDS, type Thresholds } from "../lib/detection";

const root = process.cwd();
const provisional = Boolean(process.argv[2]);
const modelPath = process.argv[2] ?? "public/models/banana-yolox-nano.onnx";
const card = provisional
  ? { inputSize: 320, thresholds: DEFAULT_THRESHOLDS }
  : JSON.parse(await readFile("public/models/model-card.json", "utf8"));
const examples = JSON.parse(await readFile("public/examples/index.json", "utf8"));
const size = card.inputSize;
const thresholds = card.thresholds as Thresholds;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = {
  mjs: pathToFileURL(path.join(root, "public/ort/ort-wasm-simd-threaded.jsep.mjs")).href,
  wasm: pathToFileURL(path.join(root, "public/ort/ort-wasm-simd-threaded.jsep.wasm")).href,
};
const startLoad = performance.now();
const session = await ort.InferenceSession.create(await readFile(modelPath), {
  executionProviders: ["wasm"],
  graphOptimizationLevel: "all",
});
const loadMs = performance.now() - startLoad;

async function predict(bytes: Uint8Array, width: number, height: number) {
  const ratio = Math.min(size / width, size / height);
  const rw = Math.floor(width * ratio),
    rh = Math.floor(height * ratio);
  const pixels = await sharp(bytes)
    .removeAlpha()
    .resize(rw, rh, { fit: "fill", kernel: "linear" })
    .raw()
    .toBuffer();
  const plane = size * size;
  const input = new Float32Array(plane * 3).fill(114);
  for (let y = 0; y < rh; y++)
    for (let x = 0; x < rw; x++) {
      const p = (y * rw + x) * 3,
        i = y * size + x;
      input[i] = pixels[p + 2];
      input[i + plane] = pixels[p + 1];
      input[i + 2 * plane] = pixels[p];
    }
  const tensor = new ort.Tensor("float32", input, [1, 3, size, size]);
  const began = performance.now();
  const outputs = await session.run({ [session.inputNames[0]]: tensor });
  const inferenceMs = performance.now() - began;
  const output = outputs[session.outputNames[0]];
  assert.deepEqual(output.dims, [1, 2100, 8]);
  const values = Array.from(output.data as Float32Array);
  assert.ok(values.every(Number.isFinite));
  const detections = decodePredictions(values, size, width, height, Date.now(), thresholds);
  tensor.dispose();
  Object.values(outputs).forEach((value) => value.dispose());
  return { inferenceMs, detections, values };
}
const images: any[] = [];
const times: number[] = [];
for (const example of examples) {
  const bytes = await readFile(path.join("public", example.src));
  let latest = await predict(bytes, example.width, example.height);
  for (let i = 0; i < 5; i++) {
    latest = await predict(bytes, example.width, example.height);
    times.push(latest.inferenceMs);
  }
  images.push({
    id: example.id,
    inferenceMs: latest.inferenceMs,
    detections: latest.detections,
    outputSignature: latest.values.reduce((sum, value, i) => sum + value * (1 + (i % 13)), 0),
  });
}
assert.ok(
  new Set(images.map((image) => image.outputSignature)).size > 1,
  "Model output must depend on image pixels.",
);
const controls: any[] = [];
for (const color of ["#727272", "#ffffff", "#1d382b"]) {
  const bytes = await sharp({ create: { width: 640, height: 480, channels: 3, background: color } })
    .jpeg()
    .toBuffer();
  const result = await predict(bytes, 640, 480);
  controls.push({
    syntheticColor: color,
    falseDetections: result.detections.length,
    detections: result.detections,
  });
}
times.sort((a, b) => a - b);
const medianMs = times[Math.floor(times.length / 2)];
const report = {
  generatedAt: new Date().toISOString(),
  status: "passed",
  provisional,
  scope:
    "Actual onnxruntime-web/all 1.24.3 JSEP WASM on Node/Linux, not a browser, camera, Android or iPhone test.",
  environment: {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
    threads: 1,
    modelSha256: card.sha256 ?? card.model?.sha256,
  },
  loadMs,
  outputShape: [1, 2100, 8],
  imageSensitivityPassed: true,
  timing: {
    measurements: times.length,
    medianMs,
    p95Ms: times[Math.floor(times.length * 0.95)],
    inferencePerSecond: 1000 / medianMs,
    includes:
      "WASM model run only; excludes image decode/resize, camera, drawing, worker transfer and scheduling",
  },
  examples: images,
  syntheticBlankControls: controls,
  realDeviceTests: "not performed",
  webgpu: "not tested in this environment",
};
await session.release();
await writeFile(
  provisional
    ? "banana-provisional-web-runtime.json"
    : "public/evaluation/web-runtime.json",
  JSON.stringify(report, null, 2) + "\n",
);
console.log(
  JSON.stringify({
    status: report.status,
    medianMs,
    p95Ms: report.timing.p95Ms,
    examples: images.map(({ id, detections }) => ({ id, detections: detections.length })),
    blankFalseDetections: controls.map((item) => item.falseDetections),
  }),
);
