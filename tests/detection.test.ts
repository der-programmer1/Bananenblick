import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { containRect, decodePredictions, updateTracks, DEFAULT_THRESHOLDS, type Detection } from "../lib/detection";

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
  const policy = { ...DEFAULT_THRESHOLDS, detection: .25, classification: .6, margin: .15 };
  const lowJoint = decodePredictions([160, 120, 160, 80, 0.75, 0.75, 0.1, 0.1], 320, 640, 480, 1, policy);
  assert.equal(lowJoint[0].ripeness, "unsicher");
  const ambiguous = decodePredictions([160, 120, 160, 80, 0.99, 0.9, 0.8, 0.1], 320, 640, 480, 1, policy);
  assert.equal(ambiguous[0].ripeness, "unsicher");
});
test("a lower display threshold cannot turn weak ripe evidence into a recommendation", () => {
  const detections = decodePredictions(
    [160, 120, 160, 80, DEFAULT_THRESHOLDS.classification * .8 / .9, 0.01, 0.9, 0.01],
    320,
    640,
    480,
    100,
    { detection: .1, classification: DEFAULT_THRESHOLDS.classification * .5, margin: 0.15, nmsIou: 0.4 },
  );
  assert.equal(detections[0].ripeEvidence, false);
  let tracks = updateTracks([], detections, 0, () => 1);
  for (let frame = 1; frame <= 3; frame++)
    tracks = updateTracks(
      tracks,
      detections.map((d) => ({ ...d, capturedAt: 100 + frame * 200 })),
      frame * 200,
      () => 2,
    );
  assert.equal(tracks[0].readyToEat, false);
});
test("contain geometry matches both phone orientations without stretching", () => {
  assert.deepEqual(containRect(400, 500, 1920, 1080), { x: 0, y: 137.5, width: 400, height: 225 });
  const portrait = containRect(400, 500, 1080, 1920);
  for (const [key, expected] of Object.entries({ x: 59.375, y: 0, width: 281.25, height: 500 }))
    assert.ok(Math.abs(portrait[key as keyof typeof portrait] - expected) < 1e-8);
});
const fruit = (x = 0.2, capturedAt = 100, ripeness: Detection["ripeness"] = "reif"): Detection => ({
  box: { x, y: 0.2, width: 0.3, height: 0.3 },
  ripeness,
  confidence: 0.9,
  predictedClass: ripeness === "unreif" ? 0 : ripeness === "überreif" ? 2 : 1,
  capturedAt,
  ripeEvidence: ripeness === "reif",
});
test("smoothing preserves identity; absent fruits immediately lose visibility and recommendation", () => {
  let id = 0;
  const nextId = () => ++id;
  const detection = fruit();
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
  const missing = updateTracks(moved, [], 240, nextId);
  assert.equal(missing[0].visible, false);
  assert.equal(missing[0].readyToEat, false);
  assert.equal(missing[0].ripeConfirmations, 0);
  assert.equal(updateTracks(missing, [], 851, nextId).length, 0);
});
test("three consecutive ripe confirmations are required; every other judgement resets them", () => {
  for (const judgement of ["unsicher", "unreif", "überreif"] as const) {
    let id = 0;
    const next = () => ++id;
    let tracks = updateTracks([], [fruit()], 0, next);
    assert.equal(tracks[0].readyToEat, false);
    tracks = updateTracks(tracks, [fruit(0.21, 200)], 100, next);
    assert.equal(tracks[0].readyToEat, false);
    tracks = updateTracks(tracks, [fruit(0.22, 300)], 200, next);
    assert.equal(tracks[0].readyToEat, true);
    tracks = updateTracks(tracks, [fruit(0.22, 400, judgement)], 300, next);
    assert.equal(tracks[0].readyToEat, false);
    assert.equal(tracks[0].ripeConfirmations, 0);
    tracks = updateTracks(tracks, [fruit(0.22, 500)], 400, next);
    assert.equal(tracks[0].ripeConfirmations, 1);
  }
});
test("an interruption and a stale or repeated image cannot accumulate confirmations", () => {
  let id = 0;
  const next = () => ++id;
  let tracks = updateTracks([], [fruit()], 0, next);
  tracks = updateTracks(tracks, [fruit()], 100, next);
  assert.equal(tracks[0].ripeConfirmations, 1);
  tracks = updateTracks(tracks, [fruit(0.2, 200)], 200, next);
  tracks = updateTracks(tracks, [], 300, next);
  tracks = updateTracks(tracks, [fruit(0.2, 400)], 400, next);
  assert.equal(tracks[0].id, 1);
  assert.equal(tracks[0].ripeConfirmations, 1);
  tracks = updateTracks(tracks, [fruit(0.2, 1100)], 1100, next);
  assert.equal(tracks[0].ripeConfirmations, 1);
});
test("global matching keeps overlapping fruit IDs through movement and reversed confidence order", () => {
  let id = 0;
  const next = () => ++id;
  let tracks = updateTracks([], [fruit(0.32), fruit(0.2)], 0, next);
  assert.deepEqual(
    tracks.map((t) => t.id),
    [1, 2],
  );
  for (let frame = 1; frame <= 4; frame++) {
    const shift = frame * 0.01;
    tracks = updateTracks(
      tracks,
      [fruit(0.32 + shift, 100 + frame * 100, "unreif"), fruit(0.2 + shift, 100 + frame * 100)],
      frame * 100,
      next,
    );
    assert.equal(new Set(tracks.map((t) => t.id)).size, 2);
    assert.equal(tracks[0].ripeness, "reif");
    assert.equal(tracks[1].ripeness, "unreif");
  }
  assert.equal(tracks[0].readyToEat, true);
  assert.equal(tracks[1].readyToEat, false);
});
test("global assignment avoids the greedy collision between two close fruit tracks", () => {
  let id = 0;
  const next = () => ++id;
  const first = updateTracks([], [fruit(0.2), fruit(0.37)], 0, next);
  const moved = updateTracks(first, [fruit(0.3, 200), fruit(0.4, 200)], 100, next);
  assert.deepEqual(
    moved.filter((t) => t.visible).map((t) => t.id),
    [1, 2],
  );
  assert.equal(moved[0].measuredBox.x, 0.3);
});
test("near-identical associations cannot transfer a confirmed recommendation", () => {
  let id = 0;
  const next = () => ++id;
  let tracks = updateTracks([], [fruit(0.2), fruit(0.21)], 0, next);
  for (let frame = 1; frame < 5; frame++)
    tracks = updateTracks(
      tracks,
      [fruit(0.2, 100 + frame * 100), fruit(0.21, 100 + frame * 100)],
      frame * 100,
      next,
    );
  assert.ok(tracks.every((t) => !t.readyToEat));
});
test("NMS preserves distinct overlapping fruits while suppressing their duplicate anchors", () => {
  const data = [
    100, 100, 100, 150, 0.99, 0.01, 0.99, 0.01, 130, 100, 100, 150, 0.98, 0.99, 0.01, 0.01, 101,
    100, 100, 150, 0.97, 0.01, 0.99, 0.01,
  ];
  const detections = decodePredictions(data, 320, 320, 320, 1, {
    detection: 0.5,
    classification: 0.6,
    margin: 0.15,
    nmsIou: 0.65,
  });
  assert.equal(detections.length, 2);
});
test("Python evaluation and browser decoder agree on clipping, ties, overlap and uncertainty", () => {
  let seed = 42;
  const random = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
  const requests = [0.45, 0.6, 0.75].flatMap((nmsIou) =>
    [
      { width: 1920, height: 1080 },
      { width: 1080, height: 1920 },
    ].map((dimensions) => ({
      ...dimensions,
      size: 320,
      capturedAt: 123,
      thresholds: { detection: 0.25, classification: 0.6, margin: 0.15, nmsIou },
      data: Array.from({ length: 100 }, () => [
        random() * 360,
        random() * 360,
        random() * 160,
        random() * 160,
        random(),
        random(),
        random(),
        random(),
      ]).flat(),
    })),
  );
  const python = spawnSync("python3", ["ml/postprocess.py"], {
    input: JSON.stringify(requests),
    encoding: "utf8",
  });
  assert.equal(python.status, 0, python.stderr);
  const expected = JSON.parse(python.stdout);
  requests.forEach((r, index) => {
    assert.deepEqual(
      decodePredictions(r.data, r.size, r.width, r.height, r.capturedAt, r.thresholds),
      expected[index],
    );
  });
});
