/** Report figures from saved predictions; never invokes or tunes a model. */
import { readFile, writeFile } from "node:fs/promises";
import { updateTracks, type Track } from "../lib/detection";
import { layoutFruitLabels } from "../lib/fruit-labels";
import { fruitStatus, recommendation } from "../lib/result-view";
import type { BenchmarkCase } from "./evaluate-recommendations";

const predictions = JSON.parse(await readFile("public/evaluation/experiment-v4/individual-predictions.json", "utf8")) as BenchmarkCase[];
const records = JSON.parse(await readFile("ml/work/individual-v4/acceptance.json", "utf8"));
// Fixed before the holdout is inferred: household bunch, mixed ripeness, green bunch.
const selected = [2, 13, 29];
const figures = selected.map((index) => {
  const record = records[index];
  const scene = predictions.find((item) => item.id === record.id)!;
  let tracks: Track[] = [], next = 0;
  for (let frame = 0; frame < 3; frame++)
    tracks = updateTracks(tracks, scene.detections.map((d) => ({ ...d, capturedAt: 1000 + frame * 200 })), frame * 200, () => ++next);
  const rect = { x: 0, y: 0, width: 760, height: Math.round(760 * record.height / record.width) };
  return {
    record, rect, tracks: tracks.filter((track) => track.visible).map((track) => ({ ...track, status: fruitStatus(track) })),
    labels: layoutFruitLabels(tracks, rect), summary: recommendation(tracks),
  };
});
await writeFile("ml/work/training-v4/example-frames.json", JSON.stringify(figures, null, 2) + "\n");
console.log("Prepared three prespecified report figures from cached predictions, using the app's tracker and numbering.");
