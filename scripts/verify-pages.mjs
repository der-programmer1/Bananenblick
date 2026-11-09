import { readFile, readdir, stat, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import assert from "node:assert/strict";

const root = path.resolve("docs");
const html = await readFile(path.join(root, "index.html"), "utf8");
assert.match(html, /Bananenblick/);
assert.match(html, /Content-Security-Policy/);
assert.match(html, /connect-src 'self'/);
assert.doesNotMatch(html, /(?:src|href)=["']\/(?!\/)/, "Static HTML must not require the domain root.");
for (const match of html.matchAll(/(?:src|href)=["']([^"']+)["']/g)) {
  if (/^(?:https?:|data:|#)/.test(match[1])) continue;
  assert.ok((await stat(path.resolve(root, match[1]))).isFile(), `Missing asset: ${match[1]}`);
  for (const prefix of ["/", "/bananenblick/", "/ein-beliebiger-repository-name/"]) {
    const resolved = new URL(match[1], `https://example.github.io${prefix}`);
    assert.ok(resolved.pathname.startsWith(prefix));
  }
}
for (const name of ["model-card.json", "banana-yolox-nano.onnx"]) assert.ok((await stat(path.join(root, "models", name))).isFile());
const card = JSON.parse(await readFile(path.join(root, "models/model-card.json"), "utf8"));
const model = await readFile(path.join(root, "models/banana-yolox-nano.onnx"));
assert.equal(createHash("sha256").update(model).digest("hex"), card.sha256);
assert.equal(card.sha256, JSON.parse(await readFile(path.join(root, "evaluation/test-report.json"), "utf8")).modelSha256);
const examples = JSON.parse(await readFile(path.join(root, "examples/index.json"), "utf8"));
for (const example of examples) assert.ok((await stat(path.join(root, example.src.replace(/^\/+/, "")))).isFile());
for (const file of ["ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm"]) assert.ok((await stat(path.join(root, "ort", file))).size > 0);
assert.ok((await stat(path.join(root, ".nojekyll"))).isFile());
// GitHub Pages does not process Cloudflare's response-header file.
await rm(path.join(root, "_headers"), { force: true });
const assets = await readdir(path.join(root, "assets"));
const worker = assets.find(file => file.startsWith("inference.worker-") && file.endsWith(".js"));
assert.ok(worker, "Compiled module worker is required.");
const workerCode = await readFile(path.join(root, "assets", worker), "utf8");
assert.doesNotMatch(workerCode, /fetch\(["']\/(?:models|ort)\//);
assert.doesNotMatch(workerCode, /new URL\(["']\/ort\//);
console.log("GitHub Pages build verified: relative HTML assets, arbitrary repository paths, worker, model hash, runtime, examples and .nojekyll.");
