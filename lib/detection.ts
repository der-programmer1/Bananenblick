import postprocessConfig from "./postprocess-config.json";
export type Ripeness = "unreif" | "reif" | "überreif" | "unsicher";
export type Box = { x: number; y: number; width: number; height: number };
export type Detection = {
  box: Box;
  ripeness: Ripeness;
  predictedClass: number;
  confidence: number;
  capturedAt: number;
  /** A single, sufficiently certain ripe observation; never a recommendation alone. */
  ripeEvidence: boolean;
};
export type Thresholds = {
  detection: number;
  classification: number;
  margin: number;
  nmsIou: number;
};
export const DEFAULT_THRESHOLDS: Thresholds = postprocessConfig;
export const CLASSES = ["unreif", "reif", "überreif"] as const;
export type FrameResult = {
  type: "result";
  sessionId: number;
  frameId: number;
  capturedAt: number;
  width: number;
  height: number;
  inferenceMs: number;
  classification: number;
  detections: Detection[];
};
export function iou(a: Box, b: Box): number {
  const w = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const h = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  const intersection = w * h;
  return intersection / Math.max(1e-10, a.width * a.height + b.width * b.height - intersection);
}
/** YOLOX output has already decoded cx/cy/w/h in model-input pixels. */
export function decodePredictions(
  data: ArrayLike<number>,
  size: number,
  width: number,
  height: number,
  capturedAt: number,
  thresholds = DEFAULT_THRESHOLDS,
): Detection[] {
  if (![size, width, height].every((n) => Number.isFinite(n) && n > 0)) return [];
  const ratio = Math.min(size / width, size / height);
  const candidates: Detection[] = [];
  for (let p = 0; p + 7 < data.length; p += 8) {
    if (!Array.from({ length: 8 }, (_, i) => data[p + i]).every(Number.isFinite)) continue;
    if (data[p + 2] <= 0 || data[p + 3] <= 0) continue;
    const probabilities = [data[p + 5], data[p + 6], data[p + 7]];
    const ranked = probabilities
      .map((probability, cls) => ({ probability, cls }))
      .sort((a, b) => b.probability - a.probability);
    const confidence = data[p + 4] * ranked[0].probability;
    if (!Number.isFinite(confidence) || confidence < thresholds.detection) continue;
    const left = (data[p] - data[p + 2] / 2) / ratio;
    const top = (data[p + 1] - data[p + 3] / 2) / ratio;
    const x = Math.max(0, Math.min(width, left));
    const y = Math.max(0, Math.min(height, top));
    const right = Math.max(0, Math.min(width, left + data[p + 2] / ratio));
    const bottom = Math.max(0, Math.min(height, top + data[p + 3] / ratio));
    if (right - x < 3 || bottom - y < 3) continue;
    const certain =
      confidence >= thresholds.classification &&
      ranked[0].probability - ranked[1].probability >= thresholds.margin;
    candidates.push({
      box: {
        x: x / width,
        y: y / height,
        width: (right - x) / width,
        height: (bottom - y) / height,
      },
      predictedClass: ranked[0].cls,
      ripeness: certain ? CLASSES[ranked[0].cls] : "unsicher",
      confidence,
      capturedAt,
      ripeEvidence:
        certain &&
        ranked[0].cls === 1 &&
        confidence >= Math.max(DEFAULT_THRESHOLDS.classification, thresholds.classification),
    });
  }
  const kept: Detection[] = [];
  for (const candidate of candidates.sort((a, b) => b.confidence - a.confidence)) {
    if (!kept.some((other) => iou(candidate.box, other.box) > thresholds.nmsIou))
      kept.push(candidate);
    if (kept.length === 40) break;
  }
  return kept;
}
export function containRect(
  containerWidth: number,
  containerHeight: number,
  sourceWidth: number,
  sourceHeight: number,
) {
  if (!sourceWidth || !sourceHeight)
    return { x: 0, y: 0, width: containerWidth, height: containerHeight };
  const scale = Math.min(containerWidth / sourceWidth, containerHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  return { x: (containerWidth - width) / 2, y: (containerHeight - height) / 2, width, height };
}
export type Track = Detection & {
  id: number;
  lastSeen: number;
  measuredBox: Box;
  visible: boolean;
  ripeConfirmations: number;
  readyToEat: boolean;
};
export const REQUIRED_RIPE_CONFIRMATIONS = 3;
const TRACK_MEMORY_MS = 750;
const CONFIRMATION_GAP_MS = 600;

/** Rectangular Hungarian assignment, including one unmatched column per detection. */
function assign(cost: number[][]): number[] {
  if (!cost.length) return [];
  const n = cost.length,
    m = cost[0].length;
  const u = new Array(n + 1).fill(0),
    v = new Array(m + 1).fill(0);
  const p = new Array(m + 1).fill(0),
    way = new Array(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const min = new Array(m + 1).fill(Infinity),
      used = new Array(m + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity,
        j1 = 0;
      for (let j = 1; j <= m; j++)
        if (!used[j]) {
          const current = cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (current < min[j]) {
            min[j] = current;
            way[j] = j0;
          }
          if (min[j] < delta) {
            delta = min[j];
            j1 = j;
          }
        }
      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else min[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  const result = new Array(n).fill(-1);
  for (let j = 1; j <= m; j++) if (p[j]) result[p[j] - 1] = j - 1;
  return result;
}
function matchCost(a: Box, b: Box): number {
  const overlap = iou(a, b);
  const distance = Math.hypot(
    (a.x + a.width / 2 - b.x - b.width / 2) / Math.max(a.width, b.width),
    (a.y + a.height / 2 - b.y - b.height / 2) / Math.max(a.height, b.height),
  );
  const scale = Math.max(
    a.width / b.width,
    b.width / a.width,
    a.height / b.height,
    b.height / a.height,
  );
  if (overlap < 0.1 || distance > 0.75 || scale > 2) return 10;
  return 0.7 * (1 - overlap) + 0.3 * distance;
}
/** Missing tracks retain only identity memory. They immediately lose their recommendation. */
export function updateTracks(
  previous: Track[],
  detections: Detection[],
  now: number,
  nextId: () => number,
): Track[] {
  const active = previous
    .filter((track) => now - track.lastSeen < TRACK_MEMORY_MS)
    .sort((a, b) => a.id - b.id);
  // Initial numbers follow the picture, independent of model confidence order.
  const ordered = [...detections].sort((a, b) => a.box.x - b.box.x || a.box.y - b.box.y);
  const costs = ordered.map((d) => [
    ...active.map((t) => matchCost(t.measuredBox, d.box)),
    ...ordered.map(() => 0.9),
  ]);
  const assignment = assign(costs);
  const used = new Set<number>();
  const result: Track[] = ordered.map((detection, index) => {
    const column = assignment[index];
    const match = column < active.length && costs[index][column] < 0.9 ? active[column] : undefined;
    // An ambiguous association must not transfer a ripe confirmation to another fruit.
    const ambiguous =
      match &&
      (active.some(
        (_, j) => j !== column && Math.abs(costs[index][j] - costs[index][column]) < 0.06,
      ) ||
        ordered.some(
          (_, i) => i !== index && Math.abs(costs[i][column] - costs[index][column]) < 0.06,
        ));
    const consecutive =
      match &&
      match.visible &&
      !ambiguous &&
      now - match.lastSeen <= CONFIRMATION_GAP_MS &&
      detection.capturedAt > match.capturedAt;
    const ripeConfirmations =
      detection.ripeEvidence && detection.ripeness === "reif"
        ? Math.min(REQUIRED_RIPE_CONFIRMATIONS, (consecutive ? match.ripeConfirmations : 0) + 1)
        : 0;
    const state = {
      measuredBox: detection.box,
      visible: true,
      ripeConfirmations,
      readyToEat: ripeConfirmations >= REQUIRED_RIPE_CONFIRMATIONS,
    };
    if (match) {
      used.add(match.id);
      const alpha = 0.85;
      const box = Object.fromEntries(
        Object.keys(detection.box).map((key) => {
          const k = key as keyof Box;
          return [key, alpha * detection.box[k] + (1 - alpha) * match.box[k]];
        }),
      ) as Box;
      return { ...detection, ...state, box, id: match.id, lastSeen: now };
    }
    return { ...detection, ...state, id: nextId(), lastSeen: now };
  });
  for (const track of active) {
    if (!used.has(track.id))
      result.push({
        ...track,
        visible: false,
        ripeEvidence: false,
        ripeConfirmations: 0,
        readyToEat: false,
        ripeness: "unsicher",
      });
  }
  return result.sort((a, b) => a.id - b.id);
}
