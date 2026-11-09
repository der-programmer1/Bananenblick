export type Ripeness = "unreif" | "reif" | "überreif" | "unsicher";
export type Box = { x: number; y: number; width: number; height: number };
export type Detection = {
  box: Box;
  ripeness: Ripeness;
  predictedClass: number;
  confidence: number;
  capturedAt: number;
  id?: number;
};
export type Thresholds = {
  detection: number;
  classification: number;
  margin: number;
  nmsIou: number;
};
export const DEFAULT_THRESHOLDS: Thresholds = {
  detection: 0.25,
  classification: 0.6,
  margin: 0.15,
  nmsIou: 0.45,
};
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
  const ratio = Math.min(size / width, size / height);
  const candidates: Detection[] = [];
  for (let p = 0; p + 7 < data.length; p += 8) {
    if (![data[p], data[p + 1], data[p + 2], data[p + 3]].every(Number.isFinite)) continue;
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
export type Track = Detection & { id: number; lastSeen: number };
/** Short EMA smoothing; empty frames retain boxes for at most 180ms. */
export function updateTracks(
  previous: Track[],
  detections: Detection[],
  now: number,
  nextId: () => number,
): Track[] {
  const used = new Set<number>();
  const result: Track[] = detections.map((detection) => {
    const match = previous
      .filter((track) => !used.has(track.id) && now - track.lastSeen < 500)
      .map((track) => ({ track, overlap: iou(track.box, detection.box) }))
      .sort((a, b) => b.overlap - a.overlap)[0];
    if (match && match.overlap > 0.2) {
      used.add(match.track.id);
      const alpha = 0.78;
      const box = Object.fromEntries(
        Object.keys(detection.box).map((key) => {
          const k = key as keyof Box;
          return [key, alpha * detection.box[k] + (1 - alpha) * match.track.box[k]];
        }),
      ) as Box;
      return { ...detection, box, id: match.track.id, lastSeen: now };
    }
    return { ...detection, id: nextId(), lastSeen: now };
  });
  for (const track of previous) {
    if (
      !used.has(track.id) &&
      now - track.lastSeen < 180 &&
      !result.some((item) => iou(item.box, track.box) > 0.2)
    )
      result.push(track);
  }
  return result;
}
