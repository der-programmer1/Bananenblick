import type { Track } from "./detection";

export function fruitStatus(track: Track): string {
  if (track.visible && track.readyToEat) return "✓ Jetzt essreif";
  if (track.ripeness === "unreif") return "Unreif";
  if (track.ripeness === "überreif") return "Überreif";
  return "Nicht sicher beurteilbar";
}
export function recommendation(tracks: Track[]): string {
  const ids = tracks
    .filter((t) => t.visible && t.readyToEat)
    .map((t) => t.id)
    .sort((a, b) => a - b);
  if (!ids.length) return "Aktuell keine Banane sicher als essreif erkannt";
  const last = ids.pop();
  return `Jetzt essreif: Nr. ${ids.length ? `${ids.join(", ")} und ` : ""}${last}`;
}
