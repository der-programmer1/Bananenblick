/** Deployment acceptance, separate from the ability to build a reviewable prototype. */
import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";
const read = async (path) => JSON.parse(await readFile(path, "utf8"));
const model = await read("public/models/model-card.json");
const report = await read("public/evaluation/individual-report.json");
assert.equal(report.modelSha256, model.sha256);
assert.ok(
  report.positiveScenes >= 30 && report.negativeScenes >= 30,
  "At least 30 positive and 30 negative real scenes required.",
);
assert.ok(
  report.realBunchScenes >= 30,
  "At least 30 verified real bunch scenes required; arranged fruits alone do not meet this criterion.",
);
assert.ok(
  report.precision >= 0.95 && report.recall >= 0.8 && report.passed,
  "Individual-fruit acceptance failed: precision >=95% and recall >=80% required.",
);
const devices = await read("public/evaluation/device-validation.json");
for (const platform of ["android-chrome", "iphone-safari"]) {
  const result = devices.devices.find(
    (d) => d.platform === platform && d.physicalDevice === true && d.modelSha256 === model.sha256,
  );
  assert.ok(
    result &&
      result.cameraStop &&
      result.portrait &&
      result.landscape &&
      result.evaluationsPerSecond >= 3,
    `Real ${platform} device evidence and >=3 complete evaluations/s required.`,
  );
}
console.log("Model and physical-device acceptance passed; GitHub Pages deployment permitted.");
