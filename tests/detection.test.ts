import assert from "node:assert/strict";
import test from "node:test";
import { containRect, decodePredictions, updateTracks, type Detection } from "../lib/detection";

test("decodes landscape letterbox into source-normalized positions", () => {
  const result = decodePredictions(
    [160, 120, 160, 80, 0.95, 0.98, 0.01, 0.01],
    320,
    640,
    480,
    1234,
  );
  assert.equal(result.length, 1);
  assert.deepEqual(result[0].box, { x: 0.25, y: 1 / 3, width: 0.5, height: 1 / 3 });
  assert.equal(result[0].ripeness, "unreif");
  assert.equal(result[0].capturedAt, 1234);
});
test("boxes entirely in letterbox padding never reach the interface", () => {
  const result = decodePredictions([160, 290, 100, 20, 0.99, 0.99, 0, 0], 320, 640, 480, 1);
  assert.deepEqual(result, []);
});
test("blank and very low objectness frames have no detections", () => {
  assert.deepEqual(decodePredictions(new Float32Array(2100 * 8), 320, 640, 480, 1), []);
  assert.deepEqual(decodePredictions([160, 160, 80, 80, 0.1, 0.99, 0, 0], 320, 320, 320, 1), []);
});
test("NMS suppresses competing ripeness classes for the same fruit", () => {
  const result = decodePredictions(
    [160, 120, 160, 80, 0.95, 0.98, 0.01, 0.01, 162, 121, 160, 80, 0.9, 0.01, 0.99, 0.01],
    320,
    640,
    480,
    1,
  );
  assert.equal(result.length, 1);
  assert.equal(result[0].ripeness, "unreif");
});
test("uncertainty uses joint confidence and class margin", () => {
  const lowJoint = decodePredictions([160, 120, 160, 80, 0.7, 0.7, 0.1, 0.1], 320, 640, 480, 1);
  assert.equal(lowJoint[0].ripeness, "unsicher");
  const ambiguous = decodePredictions([160, 120, 160, 80, 0.99, 0.9, 0.8, 0.1], 320, 640, 480, 1);
  assert.equal(ambiguous[0].ripeness, "unsicher");
});
test("contain geometry matches both phone orientations without stretching", () => {
  assert.deepEqual(containRect(400, 500, 1920, 1080), { x: 0, y: 137.5, width: 400, height: 225 });
  const portrait = containRect(400, 500, 1080, 1920);
  for (const [key, expected] of Object.entries({ x: 59.375, y: 0, width: 281.25, height: 500 }))
    assert.ok(Math.abs(portrait[key as keyof typeof portrait] - expected) < 1e-8);
});
test("smoothing preserves identity briefly and expires absent detections", () => {
  let id = 0;
  const nextId = () => ++id;
  const detection: Detection = {
    box: { x: 0.2, y: 0.2, width: 0.3, height: 0.3 },
    ripeness: "reif",
    confidence: 0.9,
    predictedClass: 1,
    capturedAt: 100,
  };
  const first = updateTracks([], [detection], 0, nextId);
  const moved = updateTracks(
    first,
    [{ ...detection, box: { ...detection.box, x: 0.22 }, capturedAt: 200 }],
    100,
    nextId,
  );
  assert.equal(moved[0].id, first[0].id);
  assert.ok(moved[0].box.x > 0.2 && moved[0].box.x < 0.22);
  assert.equal(moved[0].capturedAt, 200);
  assert.equal(updateTracks(moved, [], 240, nextId).length, 1);
  assert.equal(updateTracks(moved, [], 281, nextId).length, 0);
});
