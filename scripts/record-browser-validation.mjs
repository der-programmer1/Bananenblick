/** Record actual browser evidence, keeping mobile emulation distinct from real phones. */
import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import assert from "node:assert/strict";

const read = async (path) => JSON.parse(await readFile(path, "utf8"));
const report = await read("outputs/browser-results.json");
const card = await read("public/models/model-card.json");
assert.equal(createHash("sha256").update(await readFile("public/models/banana-yolox-nano.onnx")).digest("hex"), card.sha256);
const collect = (suites) => suites.flatMap((suite) => [...(suite.specs ?? []), ...collect(suite.suites ?? [])]);
const cases = [];
const versions = {};
const verifiedModels = [];
for (const spec of collect(report.suites)) for (const test of spec.tests) {
  const result = test.results.at(-1);
  assert.equal(test.status, "expected", `Browser test did not pass: ${test.projectName} ${spec.title}`);
  cases.push({ project: test.projectName, title: spec.title, status: test.status, durationMs: result.duration });
  const attachment = result.attachments.find((item) => item.name === "verified-runtime");
  if (attachment) {
    const runtime = JSON.parse(Buffer.from(attachment.body, "base64").toString("utf8"));
    assert.equal(runtime.modelSha256, card.sha256, "Browser evidence must refer to the current model.");
    assert.ok(new URL(runtime.url).pathname.startsWith("/Bananenblick/"));
    versions[runtime.browserName] = runtime.browserVersion;
    verifiedModels.push(runtime);
  }
}
assert.equal(verifiedModels.length, 2, "Both engines must run the actual current model.");
assert.ok(versions.chromium && versions.webkit);
const result = {
  generatedAt: new Date().toISOString(), modelSha256: card.sha256,
  physicalDeviceTestsPerformed: false, devices: [],
  browserTests: {
    context: "Production GitHub Pages build at /Bananenblick/, Playwright mobile viewport/device emulation on Linux; not physical Android or iPhone",
    versions, passed: cases.length, failed: 0, skipped: 0, cases, verifiedModels,
    media: "Controlled video clock and worker; Chromium canvas MediaStream; WebKit simulated media track. Separate examples run the real hash-verified ONNX worker.",
  },
  acceptance: "pending real Android/Chrome and iPhone/Safari, including camera indicator and >=3 complete camera evaluations per second",
};
await writeFile(process.argv[2] ?? "outputs/browser-validation.json", JSON.stringify(result, null, 2) + "\n");
console.log(`Recorded ${cases.length} passing browser tests and two actual model runs. Physical phone tests: none.`);
