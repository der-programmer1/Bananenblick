/** Static-photo benchmark using the actual browser tracker and three confirmations. */
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { iou, updateTracks, type Box, type Detection, type Track } from "../lib/detection";

export type BenchmarkCase = {
  id: string;
  scene: string;
  boxes: number[][];
  detections: Detection[];
};
export function evaluateRecommendations(cases: BenchmarkCase[]) {
  let truePositive = 0,
    falsePositive = 0,
    falseNegative = 0,
    uncertainOrUnconfirmed = 0;
  let negativeFalseRecommendations = 0,
    negativeFalseDetections = 0;
  const scenes = cases.map((scene) => {
    let next = 0,
      tracks: Track[] = [];
    for (let frame = 0; frame < 3; frame++)
      tracks = updateTracks(
        tracks,
        scene.detections.map((d) => ({ ...d, capturedAt: 1000 + frame * 200 })),
        frame * 200,
        () => ++next,
      );
    const truth = scene.boxes.map(([cls, cx, cy, width, height]) => ({
      cls,
      box: { x: cx - width / 2, y: cy - height / 2, width, height } as Box,
    }));
    const confirmed = tracks
      .filter((t) => t.visible && t.readyToEat)
      .sort((a, b) => b.confidence - a.confidence);
    const used = new Set<number>();
    let tp = 0,
      fp = 0;
    for (const track of confirmed) {
      const matches = truth
        .map((g, index) => ({ g, index, overlap: iou(g.box, track.box) }))
        .filter((m) => !used.has(m.index) && m.overlap >= 0.5)
        .sort((a, b) => b.overlap - a.overlap);
      const best = matches[0];
      if (best) used.add(best.index);
      if (best?.g.cls === 1) tp++;
      else fp++;
    }
    const support = truth.filter((g) => g.cls === 1).length;
    const uncertain = truth.filter(
      (g, index) =>
        g.cls === 1 &&
        !used.has(index) &&
        scene.detections.some(
          (d) =>
            iou(g.box, d.box) >= 0.5 &&
            (!d.ripeEvidence || !confirmed.some((t) => iou(t.box, g.box) >= 0.5)),
        ),
    ).length;
    truePositive += tp;
    falsePositive += fp;
    falseNegative += support - tp;
    uncertainOrUnconfirmed += uncertain;
    if (scene.scene === "no-banana") {
      negativeFalseRecommendations += fp;
      negativeFalseDetections += scene.detections.length;
    }
    return {
      id: scene.id,
      scene: scene.scene,
      support,
      truePositive: tp,
      falsePositive: fp,
      falseNegative: support - tp,
      uncertainOrUnconfirmed: uncertain,
      confirmed: confirmed.map((t) => ({ id: t.id, box: t.box, confidence: t.confidence })),
    };
  });
  const precision =
    truePositive + falsePositive ? truePositive / (truePositive + falsePositive) : 0;
  const recall = truePositive + falseNegative ? truePositive / (truePositive + falseNegative) : 0;
  return {
    protocol:
      "Real static photos, exact browser tracking on three identical inference results at 200 ms intervals; no live-camera evidence",
    iou: 0.5,
    requiredConfirmations: 3,
    positiveScenes: cases.filter((c) => c.scene !== "no-banana").length,
    negativeScenes: cases.filter((c) => c.scene === "no-banana").length,
    truePositive,
    falsePositive,
    falseNegative,
    precision,
    recall,
    uncertainOrUnconfirmed,
    negativeFalseRecommendations,
    negativeFalseDetections,
    targets: { precision: 0.95, recall: 0.8 },
    passed: precision >= 0.95 && recall >= 0.8,
    scenes,
  };
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const cases = JSON.parse(await readFile(process.argv[2], "utf8")) as BenchmarkCase[];
  const report = evaluateRecommendations(cases);
  await writeFile(process.argv[3], JSON.stringify(report, null, 2) + "\n");
  console.log(
    JSON.stringify({ precision: report.precision, recall: report.recall, passed: report.passed }),
  );
}
