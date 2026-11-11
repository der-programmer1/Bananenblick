import type { Track } from "./detection";

type Rect = { x: number; y: number; width: number; height: number };
type Fruit = Pick<Track, "id" | "box" | "visible" | "readyToEat">;
export type FruitLabel = Rect & { id: number; anchorX: number; anchorY: number };
const overlapArea = (a: Rect, b: Rect, gap = 2) =>
  Math.max(0, Math.min(a.x + a.width + gap, b.x + b.width) - Math.max(a.x - gap, b.x)) *
  Math.max(0, Math.min(a.y + a.height + gap, b.y + b.height) - Math.max(a.y - gap, b.y));

/** Keep compact number badges readable without changing the detected fruit boxes. */
export function layoutFruitLabels(tracks: Fruit[], rect: Rect): FruitLabel[] {
  if (rect.width <= 0 || rect.height <= 0) return [];
  const placed: FruitLabel[] = [];
  for (const track of tracks.filter((t) => t.visible).toSorted((a, b) => a.id - b.id)) {
    const frame = { x: rect.x + track.box.x * rect.width, y: rect.y + track.box.y * rect.height,
      width: track.box.width * rect.width, height: track.box.height * rect.height };
    const width = Math.min(rect.width, 14 + String(track.id).length * 7 + (track.readyToEat ? 11 : 0));
    const height = Math.min(rect.height, 22);
    const at = (x: number, y: number): Rect => ({
      x: Math.max(rect.x, Math.min(x, rect.x + rect.width - width)),
      y: Math.max(rect.y, Math.min(y, rect.y + rect.height - height)), width, height,
    });
    const candidates = [
      at(frame.x, frame.y), at(frame.x + frame.width - width, frame.y),
      at(frame.x, frame.y + frame.height - height), at(frame.x + frame.width - width, frame.y + frame.height - height),
      at(frame.x, frame.y + (frame.height - height) / 2),
      at(frame.x + frame.width - width, frame.y + (frame.height - height) / 2),
    ];
    const overlaps = (candidate: Rect) => placed.reduce((sum, other) => sum + overlapArea(candidate, other), 0);
    let best = candidates.find((candidate) => overlaps(candidate) === 0);
    if (!best) {
      // Dense or almost coincident boxes may have no free corner. Search nearby
      // image space and draw a connector so the number still identifies its box.
      for (let y = rect.y; y <= rect.y + rect.height - height; y += height + 4)
        for (let x = rect.x; x <= rect.x + rect.width - width; x += width + 4) candidates.push(at(x, y));
      best = candidates.reduce((a, b) => {
        const score = (p: Rect) => overlaps(p) * 1e6 + (p.x - frame.x) ** 2 + (p.y - frame.y) ** 2;
        return score(b) < score(a) ? b : a;
      });
    }
    const centerX = best.x + width / 2, centerY = best.y + height / 2;
    placed.push({ ...best, id: track.id,
      anchorX: Math.max(frame.x, Math.min(centerX, frame.x + frame.width)),
      anchorY: Math.max(frame.y, Math.min(centerY, frame.y + frame.height)),
    });
  }
  return placed;
}
