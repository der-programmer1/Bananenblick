import type { Track } from "../lib/detection";
import { fruitStatus } from "../lib/result-view";

type Selection = { selectedId: number | null; onSelect: (id: number) => void };
export function FruitBoxes({
  tracks,
  rect,
  selectedId,
  onSelect,
}: Selection & {
  tracks: Track[];
  rect: { x: number; y: number; width: number; height: number };
}) {
  return (
    <div className="detection-layer" aria-label="Nummerierte Bananen im Bild">
      {tracks
        .filter((t) => t.visible)
        .map((track) => (
          <button
            type="button"
            key={track.id}
            data-track-id={track.id}
            id={`fruit-box-${track.id}`}
            aria-controls={`fruit-result-${track.id}`}
            aria-label={`Nr. ${track.id}: ${fruitStatus(track)}`}
            aria-pressed={selectedId === track.id}
            onClick={() => onSelect(track.id)}
            className={`detection-box ${track.readyToEat ? "ready" : track.ripeness === "reif" ? "unsicher" : track.ripeness} ${selectedId === track.id ? "selected" : ""}`}
            style={{
              left: rect.x + track.box.x * rect.width,
              top: rect.y + track.box.y * rect.height,
              width: track.box.width * rect.width,
              height: track.box.height * rect.height,
            }}
          >
            <span className="detection-label">
              <b>Nr. {track.id}</b>
              {track.readyToEat && <span>✓ Jetzt essreif</span>}
            </span>
          </button>
        ))}
    </div>
  );
}
export function FruitList({ tracks, selectedId, onSelect }: Selection & { tracks: Track[] }) {
  return (
    <ol className="fruit-results" aria-label="Ergebnis pro Banane">
      {tracks
        .filter((t) => t.visible)
        .map((track) => (
          <li key={track.id}>
            <button
              type="button"
              id={`fruit-result-${track.id}`}
              data-track-id={track.id}
              aria-controls={`fruit-box-${track.id}`}
              aria-pressed={selectedId === track.id}
              className={`${track.readyToEat ? "ready" : ""} ${selectedId === track.id ? "selected" : ""}`}
              onClick={() => onSelect(track.id)}
            >
              <span className="fruit-number">Nr. {track.id}</span>
              <span>
                <strong>{fruitStatus(track)}</strong>
                {track.ripeness === "reif" && !track.readyToEat && (
                  <small>Reife wird bestätigt ({track.ripeConfirmations}/3)</small>
                )}
              </span>
            </button>
          </li>
        ))}
    </ol>
  );
}
