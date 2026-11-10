import assert from "node:assert/strict";
import test from "node:test";
import { evaluateRecommendations } from "../scripts/evaluate-recommendations";
import type { Detection } from "../lib/detection";
test("recommendation metrics count uncertain ripe fruit as missed and negative-scene recommendations as false", () => {
  const detection: Detection = {
    box: { x: 0.2, y: 0.2, width: 0.3, height: 0.3 },
    ripeness: "reif",
    ripeEvidence: true,
    predictedClass: 1,
    confidence: 0.99,
    capturedAt: 0,
  };
  const report = evaluateRecommendations([
    {
      id: "ripe",
      scene: "multiple-real-fruits",
      boxes: [[1, 0.35, 0.35, 0.3, 0.3]],
      detections: [detection],
    },
    {
      id: "uncertain",
      scene: "multiple-real-fruits",
      boxes: [[1, 0.35, 0.35, 0.3, 0.3]],
      detections: [{ ...detection, ripeness: "unsicher", ripeEvidence: false }],
    },
    { id: "negative", scene: "no-banana", boxes: [], detections: [detection] },
  ]);
  assert.equal(report.truePositive, 1);
  assert.equal(report.falsePositive, 1);
  assert.equal(report.falseNegative, 1);
  assert.equal(report.uncertainOrUnconfirmed, 1);
  assert.equal(report.precision, 0.5);
  assert.equal(report.recall, 0.5);
  assert.equal(report.negativeFalseRecommendations, 1);
  assert.equal(report.passed, false);
});
