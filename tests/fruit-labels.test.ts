import assert from "node:assert/strict";
import test from "node:test";
import { layoutFruitLabels } from "../lib/fruit-labels";

const fruit = (id: number) => ({ id, box: { x: .3 + (id % 4) * .015, y: .2 + (id % 3) * .02, width: .25, height: .6 }, visible: true, readyToEat: id % 2 === 0 });
test("all numbers remain separate and inside the image for a dense phone-sized bunch", () => {
  const tracks = Array.from({ length: 20 }, (_, index) => fruit(index + 1));
  const original = structuredClone(tracks);
  const labels = layoutFruitLabels(tracks, { x: 10, y: 70, width: 320, height: 260 });
  assert.deepEqual(tracks, original, "Label placement must not change the fruit boxes.");
  assert.deepEqual(labels.map((l) => l.id), tracks.map((t) => t.id));
  for (const [index, a] of labels.entries()) {
    assert.ok(a.x >= 10 && a.y >= 70 && a.x + a.width <= 330 && a.y + a.height <= 330);
    for (const b of labels.slice(index + 1))
      assert.ok(a.x + a.width <= b.x || b.x + b.width <= a.x || a.y + a.height <= b.y || b.y + b.height <= a.y,
        `Numbers ${a.id} and ${b.id} overlap.`);
  }
});
test("label identity is stable under reordered detections and hidden fruit has no badge", () => {
  const tracks = [fruit(1), fruit(2), { ...fruit(3), visible: false }];
  const rect = { x: 0, y: 0, width: 320, height: 260 };
  assert.deepEqual(layoutFruitLabels(tracks, rect), layoutFruitLabels([...tracks].reverse(), rect));
  assert.deepEqual(layoutFruitLabels(tracks, rect).map((l) => l.id), [1, 2]);
  assert.deepEqual(layoutFruitLabels(tracks, { ...rect, width: 0 }), []);
});
