import { mkdir, readdir, copyFile } from "node:fs/promises";
import path from "node:path";
const source = path.resolve("node_modules/onnxruntime-web/dist");
await mkdir("public/ort", { recursive: true });
for (const file of await readdir(source)) {
  if (/^ort-wasm-simd-threaded\.jsep\.(wasm|mjs)$/.test(file))
    await copyFile(path.join(source, file), path.join("public/ort", file));
}
console.log("Self-hosted ONNX Runtime Web assets prepared.");
