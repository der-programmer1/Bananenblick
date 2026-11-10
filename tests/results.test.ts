import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { FruitBoxes, FruitList } from "../app/fruit-results";
import { fruitStatus, recommendation } from "../lib/result-view";
import { updateTracks, type Detection } from "../lib/detection";

function confirmed() {
  let id = 0;
  const detections: Detection[] = [0, 1, 2, 3].map((n) => ({
    box: { x: n * 0.22, y: 0.2, width: 0.18, height: 0.5 },
    ripeness: n % 2 ? "reif" : "unreif",
    predictedClass: n % 2,
    confidence: 0.99,
    capturedAt: 0,
    ripeEvidence: !!(n % 2),
  }));
  let tracks = updateTracks([], detections, 0, () => ++id);
  for (let frame = 1; frame < 3; frame++)
    tracks = updateTracks(
      tracks,
      detections.map((d) => ({ ...d, capturedAt: frame * 200 })),
      frame * 200,
      () => ++id,
    );
  return tracks;
}
test("overview names only confirmed ripe fruits with the same numbers as boxes and list", () => {
  const tracks = confirmed();
  assert.equal(recommendation(tracks), "Jetzt essreif: Nr. 2 und 4");
  const props = { tracks, selectedId: 2, onSelect: () => {} };
  const boxes = renderToStaticMarkup(
    createElement(FruitBoxes, { ...props, rect: { x: 0, y: 0, width: 400, height: 300 } }),
  );
  const list = renderToStaticMarkup(createElement(FruitList, props));
  for (const t of tracks) {
    assert.ok(boxes.includes(`id="fruit-box-${t.id}"`));
    assert.ok(boxes.includes(`aria-controls="fruit-result-${t.id}"`));
    assert.ok(list.includes(`id="fruit-result-${t.id}"`));
    assert.ok(list.includes(`aria-controls="fruit-box-${t.id}"`));
    assert.ok(list.includes(`Nr. ${t.id}`));
  }
  assert.equal((boxes.match(/aria-pressed="true"/g) ?? []).length, 1);
  assert.equal((list.match(/aria-pressed="true"/g) ?? []).length, 1);
  assert.equal((list.match(/✓ Jetzt essreif/g) ?? []).length, 2);
});
test("other stages have no check; an empty frame has no recommendation or visible result", () => {
  const tracks = confirmed();
  assert.equal(fruitStatus({ ...tracks[0], ripeness: "überreif" }), "Überreif");
  assert.equal(fruitStatus({ ...tracks[0], ripeness: "unsicher" }), "Nicht sicher beurteilbar");
  const missing = updateTracks(tracks, [], 500, () => 5);
  assert.equal(recommendation(missing), "Aktuell keine Banane sicher als essreif erkannt");
  assert.equal(
    renderToStaticMarkup(
      createElement(FruitList, { tracks: missing, selectedId: null, onSelect: () => {} }),
    ).includes("<li>"),
    false,
  );
});
