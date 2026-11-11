import type { Track } from "../lib/detection";
import { fruitStatus } from "../lib/result-view";
import { layoutFruitLabels } from "../lib/fruit-labels";

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
  const labels = layoutFruitLabels(tracks, rect);
  const byId = new Map(labels.map((label) => [label.id, label]));
  return (
    <div className="detection-layer" aria-label="Nummerierte Bananen im Bild">
      <svg className="fruit-label-connectors" aria-hidden="true">
        {labels.map((label) => <line key={label.id} x1={label.x + label.width / 2} y1={label.y + label.height / 2}
          x2={label.anchorX} y2={label.anchorY} />)}
      </svg>
      {tracks
        .filter((t) => t.visible)
        .map((track) => {
          const label = byId.get(track.id);
          return (
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
            <span className="detection-label" aria-hidden="true" style={label ? {
              left: label.x - rect.x - track.box.x * rect.width,
              top: label.y - rect.y - track.box.y * rect.height,
              width: label.width, height: label.height,
            } : undefined}>
              <b>{track.id}</b>
              {track.readyToEat && <span>✓</span>}
            </span>
          </button>
          );
        })}
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
