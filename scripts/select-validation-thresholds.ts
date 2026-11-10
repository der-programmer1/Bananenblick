/** Tune on cached validation predictions using exactly the browser recommendation state. */
import { readFile, writeFile } from "node:fs/promises";
import assert from "node:assert/strict";
import { evaluateRecommendations, type BenchmarkCase } from "./evaluate-recommendations";
import { CLASSES, type Detection, type Thresholds } from "../lib/detection";

type CachedDetection = Detection & { _classMargin: number };
const input = JSON.parse(await readFile(process.argv[2], "utf8")) as {
  split: string;
  byNms: Record<string, (Omit<BenchmarkCase, "detections"> & { detections: CachedDetection[] })[]>;
};
assert.equal(input.split, "validation", "Threshold tuning is only permitted on validation data.");
const candidates = [];
for (const [nms, cases] of Object.entries(input.byNms)) {
  for (const detection of [.2, .3, .4, .5]) for (const classification of [.4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9]) for (const margin of [.1, .15, .2, .3, .4]) {
    const thresholds: Thresholds = { detection, classification, margin, nmsIou: Number(nms) };
    const filtered: BenchmarkCase[] = cases.map((scene) => ({ ...scene,
      detections: scene.detections.filter((d) => d.confidence >= detection).map((d) => {
        const certain = d.confidence >= classification && d._classMargin >= margin;
        return { ...d, ripeness: certain ? CLASSES[d.predictedClass] : "unsicher",
          ripeEvidence: certain && d.predictedClass === 1 };
      }),
    }));
    const { scenes: _, ...result } = evaluateRecommendations(filtered);
    const f1 = 2 * result.precision * result.recall / Math.max(1e-12, result.precision + result.recall);
    candidates.push({ thresholds, ...result, f1 });
  }
}
const eligible = candidates.filter((c) => c.precision >= .95);
const ranked = (eligible.length ? eligible : candidates).sort((a, b) =>
  (eligible.length ? b.recall - a.recall : b.f1 - a.f1) || b.precision - a.precision ||
  a.thresholds.nmsIou - b.thresholds.nmsIou || b.thresholds.detection - a.thresholds.detection);
const selection = { selected: ranked[0], candidates,
  rule: "Validation only: require >=95% confirmed-ripe precision then maximize recall. If none qualifies, maximize validation F1. Evaluate three confirmations with the actual browser tracker. Never tune on the v2 test." };
await writeFile(process.argv[3], JSON.stringify(selection, null, 2) + "\n");
console.log(JSON.stringify(selection.selected));
