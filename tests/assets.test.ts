import test from "node:test";
import assert from "node:assert/strict";
import { assetPathFor } from "../lib/assets";

test("all runtime assets remain inside a GitHub repository path", () => {
  for (const base of ["./", "/", "/bananenblick/", "/anderes-repository/"]) {
    for (const asset of ["/examples/index.json", "/models/banana-yolox-nano.onnx", "/ort/ort-wasm-simd-threaded.jsep.wasm", "/evaluation/test-report.json", "/"]) {
      const page = new URL(base.startsWith("/anderes") ? "https://example.github.io/anderes-repository/" : "https://example.github.io/bananenblick/");
      const resolved = new URL(assetPathFor(base, asset), page);
      assert.equal(resolved.origin, page.origin);
      if (base === "./") assert.ok(resolved.pathname.startsWith("/bananenblick/"));
      else assert.ok(resolved.pathname.startsWith(base));
    }
  }
});

test("worker model and WASM URLs use the page base, not the worker assets folder", () => {
  const base = new URL(assetPathFor("./", ""), "https://example.github.io/repo/").href;
  assert.equal(new URL("models/banana-yolox-nano.onnx", base).href, "https://example.github.io/repo/models/banana-yolox-nano.onnx");
  assert.equal(new URL("ort/ort-wasm-simd-threaded.jsep.mjs", base).href, "https://example.github.io/repo/ort/ort-wasm-simd-threaded.jsep.mjs");
});
