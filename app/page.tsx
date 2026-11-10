"use client";
import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  Banana,
  Camera,
  CameraOff,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Cpu,
  Download,
  Expand,
  Eye,
  Focus,
  Leaf,
  LoaderCircle,
  LockKeyhole,
  RotateCcw,
  ScanLine,
  Settings2,
  ShieldCheck,
  Sparkles,
  Square,
  Sun,
  X,
  Zap,
} from "lucide-react";
import { containRect, DEFAULT_THRESHOLDS, type Ripeness } from "../lib/detection";
import { recommendation } from "../lib/result-view";
import { FruitBoxes, FruitList } from "./fruit-results";
import { useScanner, type Example } from "../lib/use-scanner";
import type { ModelCard, EvaluationReport } from "../lib/model-types";
import { assetPath } from "../lib/assets";

const STAGES: {
  id: Ripeness;
  title: string;
  subtitle: string;
  description: string;
  color: string;
  number: string;
}[] = [
  {
    id: "unreif",
    title: "Noch ein bisschen Geduld.",
    subtitle: "Unreif",
    description: "Grüne Schale, feste Frucht. Darf zu Hause noch nachreifen.",
    color: "green",
    number: "01",
  },
  {
    id: "reif",
    title: "Genau der richtige Moment.",
    subtitle: "Reif",
    description: "Sattgelbe Schale. Bereit zum Genießen oder für unterwegs.",
    color: "yellow",
    number: "02",
  },
  {
    id: "überreif",
    title: "Voller Süße. Voller Möglichkeiten.",
    subtitle: "Überreif",
    description: "Viele braune Flecken. Oft ideal für Smoothies und Bananenbrot.",
    color: "brown",
    number: "03",
  },
];
function BananaArt({ stage }: { stage: string }) {
  const color = stage === "unreif" ? "#a6bd58" : stage === "reif" ? "#f3ce48" : "#c9a263";
  const shade = stage === "unreif" ? "#77933e" : stage === "reif" ? "#d8ab28" : "#947347";
  return (
    <svg viewBox="0 0 150 100" className="banana-art" aria-hidden="true">
      <defs>
        <linearGradient id={`banana-${stage}`} x1="0" y1="0" x2="1" y2="1">
          <stop stopColor={color} />
          <stop offset="1" stopColor={shade} />
        </linearGradient>
      </defs>
      <path
        d="M30 19c4 23 17 38 39 41 24 4 48-10 62-29-5 30-31 54-63 49C37 75 23 48 26 24Z"
        fill={`url(#banana-${stage})`}
      />
      <path
        d="M30 21c6 27 20 45 47 49 21 2 38-10 49-25"
        fill="none"
        stroke={shade}
        strokeWidth="2"
        opacity=".6"
      />
      <path d="m27 26-5-12 6-3 7 12" fill="#676947" />
      <path d="m126 37 6-9 3 3-3 10" fill="#5b583c" />
      {stage === "überreif" && (
        <g fill="#755338" opacity=".85">
          <ellipse cx="45" cy="54" rx="3" ry="2" transform="rotate(35 45 54)" />
          <circle cx="61" cy="67" r="2" />
          <ellipse cx="86" cy="68" rx="4" ry="2" />
          <circle cx="99" cy="58" r="2.4" />
          <circle cx="48" cy="63" r="1.5" />
          <circle cx="74" cy="72" r="1.4" />
          <ellipse cx="113" cy="49" rx="2" ry="3" />
        </g>
      )}
      <path
        d="M36 36c4 13 11 22 23 26"
        fill="none"
        stroke="#fff"
        strokeWidth="3"
        opacity=".22"
        strokeLinecap="round"
      />
    </svg>
  );
}
function Modal({
  title,
  eyebrow,
  children,
  onClose,
  wide = false,
}: {
  title: string;
  eyebrow: string;
  children: ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    dialog?.showModal();
    const close = () => onClose();
    dialog?.addEventListener("cancel", close);
    return () => dialog?.removeEventListener("cancel", close);
  }, [onClose]);
  return (
    <dialog
      ref={ref}
      className={`modal ${wide ? "wide" : ""}`}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="modal-header">
        <div>
          <span className="eyebrow">{eyebrow}</span>
          <h2>{title}</h2>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="Dialog schließen">
          <X size={21} />
        </button>
      </div>
      <div className="modal-body">{children}</div>
    </dialog>
  );
}
function downloadJson(value: unknown, filename: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export default function Home() {
  const scanner = useScanner();
  const [dialog, setDialog] = useState<
    "guide" | "quality" | "privacy" | "settings" | "examples" | null
  >(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [exampleError, setExampleError] = useState(false);
  const [modelCard, setModelCard] = useState<ModelCard | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationReport | null>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });
  const [fullscreen, setFullscreen] = useState(false);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const scannerRef = useRef<HTMLElement>(null);
  const live = scanner.mode === "live";
  const isExample = scanner.mode === "example";
  const hasSource = live || isExample;
  const loading = scanner.mode === "starting" || (hasSource && !scanner.modelReady);
  useEffect(() => {
    if (!scanner.tracks.some((track) => track.id === selectedId)) setSelectedId(null);
  }, [scanner.tracks, selectedId]);
  const counts = Object.fromEntries(
    ["unreif", "reif", "überreif", "unsicher"].map((stage) => [
      stage,
      scanner.tracks.filter((track) => stage === "reif" ? track.readyToEat :
        stage === "unsicher" ? track.ripeness === "unsicher" || (track.ripeness === "reif" && !track.readyToEat) : track.ripeness === stage).length,
    ]),
  );
  const average = scanner.tracks.length
    ? scanner.tracks.reduce((sum, track) => sum + track.confidence, 0) / scanner.tracks.length
    : 0;
  const sourceRect = containRect(
    stageSize.width,
    stageSize.height,
    scanner.dimensions.width,
    scanner.dimensions.height,
  );
  useEffect(() => {
    fetch(assetPath("/examples/index.json"))
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.json() as Promise<Example[]>;
      })
      .then(setExamples)
      .catch(() => setExampleError(true));
    fetch(assetPath("/models/model-card.json"))
      .then((r) => (r.ok ? (r.json() as Promise<ModelCard>) : null))
      .then(setModelCard)
      .catch(() => {});
    fetch(assetPath("/evaluation/test-report.json"))
      .then((r) => (r.ok ? (r.json() as Promise<EvaluationReport>) : null))
      .then(setEvaluation)
      .catch(() => {});
  }, []);
  useEffect(() => {
    const node = stageRef.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) =>
      setStageSize({ width: entry.contentRect.width, height: entry.contentRect.height }),
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const update = () => setFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", update);
    return () => document.removeEventListener("fullscreenchange", update);
  }, []);
  const toggleFullscreen = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (scannerRef.current?.requestFullscreen) await scannerRef.current.requestFullscreen();
      else setFullscreen((value) => !value);
    } catch {
      setFullscreen((value) => !value);
    }
  };
  const currentExampleIndex = examples.findIndex((example) => example.id === scanner.example?.id);
  const chooseExample = (example: Example) => {
    setDialog(null);
    scanner.showExample(example);
  };
  const runtime =
    scanner.provider === "webgpu"
      ? "WebGPU"
      : scanner.provider === "wasm"
        ? "WebAssembly"
        : "Lokale Auswertung";
  const qualityMetrics = evaluation?.test?.perClass ?? modelCard?.metrics?.perClass;
  return (
    <>
      <header className="site-header">
        <a href={assetPath("/")} className="brand" aria-label="Bananenblick Startseite">
          <span className="brand-icon">
            <Banana size={27} strokeWidth={1.7} />
          </span>
          <span>
            bananen<span className="brand-light">blick</span>
            <span className="brand-dot">.</span>
          </span>
        </a>
        <nav aria-label="Hauptnavigation">
          <a className="active" href="#scanner">
            Live-Scanner
          </a>
          <button onClick={() => setDialog("guide")}>So funktioniert’s</button>
          <button onClick={() => setDialog("quality")}>
            Modell & Qualität <ArrowUpRight size={13} />
          </button>
        </nav>
        <button className="privacy-pill" onClick={() => setDialog("privacy")}>
          <ShieldCheck size={16} />
          <span>Bleibt auf deinem Handy</span>
          <span className="small-status-dot" />
        </button>
      </header>
      <main>
        <section className="intro">
          <div className="intro-copy">
            <div className="eyebrow intro-eyebrow">
              <span className="little-line" /> WENIGER RÄTSELN. BEWUSSTER GENIESSEN.
            </div>
            <h1>
              Die richtige Reife.
              <br />
              <span>Auf einen Blick.</span>
              <span className="heading-spark">
                <Sparkles />
              </span>
            </h1>
            <p>
              Richte deine Kamera auf die Bananen. Bananenblick erkennt
              <br className="desktop-break" /> ihre sichtbare Reife – direkt im Bild, direkt auf
              deinem Handy.
            </p>
          </div>
          <div className="intro-note">
            <div className="note-icon">
              <ScanLine size={24} />
            </div>
            <div>
              <strong>Deine Kamera. Ein neuer Blick.</strong>
              <p>
                Ohne Foto-Upload.
                <br />
                Ohne langes Überlegen.
              </p>
            </div>
            <span className="hand-arrow" aria-hidden="true">
              ↙
            </span>
          </div>
        </section>

        <div className="workspace">
          <section
            className={`scanner-card ${fullscreen ? "fullscreen-fallback" : ""}`}
            ref={scannerRef}
            id="scanner"
            aria-label="Bananen Live-Scanner"
          >
            <div className="scanner-toolbar">
              <div className="scanner-title">
                <ScanLine size={18} />
                <strong>{isExample ? "Beispiel-Scanner" : "Live-Scanner"}</strong>
                <span className={`source-badge ${live && scanner.modelReady ? "is-live" : ""}`}>
                  <i />
                  {live
                    ? scanner.modelReady
                      ? "LIVE"
                      : "LÄDT"
                    : isExample
                      ? "BEISPIEL"
                      : "BEREIT"}
                </span>
              </div>
              <div className="toolbar-actions">
                <button
                  className="icon-button"
                  onClick={() => setDialog("settings")}
                  aria-label="Erkennung einstellen"
                  title="Erkennung einstellen"
                >
                  <Settings2 size={18} />
                </button>
                <span className="toolbar-divider" />
                <button
                  className="icon-button"
                  onClick={toggleFullscreen}
                  aria-label={fullscreen ? "Vollbild beenden" : "Ansicht vergrößern"}
                  title="Ansicht vergrößern"
                >
                  {fullscreen ? <X size={18} /> : <Expand size={18} />}
                </button>
              </div>
            </div>
            <div className={`camera-stage ${hasSource ? "has-source" : ""}`} ref={stageRef}>
              <video
                ref={scanner.videoRef}
                className={`camera-video ${live ? "visible" : ""}`}
                autoPlay
                playsInline
                muted
                aria-label="Livebild deiner Rückkamera"
              />
              {isExample && scanner.example && (
                <img
                  ref={scanner.imageRef}
                  className="example-image"
                  src={assetPath(scanner.example.src)}
                  alt={scanner.example.title}
                  onLoad={scanner.capture}
                  onError={() =>
                    scanner.fail(
                      "Das Beispielbild konnte nicht geladen werden. Bitte lade die Seite erneut.",
                    )
                  }
                />
              )}
              {!hasSource && <div className="standby-photo" />}
              {!hasSource && <div className="standby-shade" />}
              {!hasSource && (
                <>
                  <div className="viewfinder-corner top-left" />
                  <div className="viewfinder-corner top-right" />
                  <div className="viewfinder-corner bottom-left" />
                  <div className="viewfinder-corner bottom-right" />
                </>
              )}
              <div className="camera-topline">
                <span className={`camera-caption ${isExample ? "example-caption" : ""}`}>
                  <span className="caption-dot" />
                  {isExample
                    ? "Beispielbild · keine Liveaufnahme"
                    : live
                      ? "Rückkamera · nur auf deinem Gerät"
                      : "DEIN HANDY WIRD ZUM REIFE-SCANNER"}
                </span>
                {live && (
                  <span className="camera-resolution">
                    {scanner.dimensions.width} × {scanner.dimensions.height}
                  </span>
                )}
              </div>
              {hasSource && <FruitBoxes tracks={scanner.tracks} rect={sourceRect}
                selectedId={selectedId} onSelect={setSelectedId} />}
              {!hasSource && (
                <div className="standby-content">
                  <div
                    className={`camera-emblem ${scanner.mode === "error" ? "error-emblem" : ""}`}
                  >
                    {scanner.mode === "error" ? (
                      <CameraOff size={31} strokeWidth={1.4} />
                    ) : (
                      <Camera size={32} strokeWidth={1.5} />
                    )}
                  </div>
                  <h2>
                    {scanner.mode === "error"
                      ? "Der Scan ist unterbrochen."
                      : "Bereit für einen frischen Blick?"}
                  </h2>
                  <p>{scanner.error || "Finde heraus, welche Banane heute zu dir passt."}</p>
                  <button
                    className="button button-yellow start-button"
                    onClick={scanner.start}
                    disabled={loading}
                  >
                    {loading ? <LoaderCircle className="spin" size={19} /> : <Camera size={19} />}{" "}
                    {loading
                      ? "Kamera wird gestartet …"
                      : scanner.mode === "error"
                        ? "Erneut starten"
                        : "Kamera starten"}
                    {!loading && <ArrowRight size={18} />}
                  </button>
                  <span className="camera-permission">
                    <LockKeyhole size={12} /> Du entscheidest, wann deine Kamera an ist.
                  </span>
                </div>
              )}
              {loading && hasSource && (
                <div className="loading-overlay">
                  <LoaderCircle size={30} className="spin" />
                  <strong>{scanner.modelStatus}</strong>
                  <span>Ein kurzer Moment. Die Auswertung bleibt auf deinem Gerät.</span>
                </div>
              )}
              {live && scanner.modelReady && (
                <div className="scan-instruction">
                  <Focus size={17} />
                  <span>
                    {scanner.tracks.length
                      ? "Erkannte Bananen werden direkt markiert"
                      : "Halte Bananen gut sichtbar ins Bild"}
                  </span>
                </div>
              )}
              {scanner.notice && hasSource && (
                <div className="camera-notice" role="status">
                  {scanner.notice}
                </div>
              )}
              {!hasSource && (
                <span className="standby-footnote">
                  Hintergrund: Beispielbild · keine aktive Kamera
                </span>
              )}
            </div>
            <div className="scanner-bottom">
              <div className="runtime-indicator">
                <span className={`status-dot ${scanner.modelReady ? "ready" : ""}`} />
                <span>
                  {scanner.modelReady ? (
                    <>
                      <strong>Modell aktiv</strong>
                      <span className="separator">·</span>
                      {runtime}
                    </>
                  ) : loading ? (
                    scanner.modelStatus
                  ) : (
                    "Alles bereit für deinen ersten Scan"
                  )}
                </span>
              </div>
              <div className="scanner-measures">
                <span>
                  <Zap size={13} />
                  {live && scanner.stats.fps
                    ? `${scanner.stats.fps.toFixed(1).replace(".", ",")} Bilder/s`
                    : "– Bilder/s"}
                </span>
                <span className="measure-divider" />
                <span>
                  {hasSource && scanner.stats.inferenceMs
                    ? `${Math.round(scanner.stats.inferenceMs)} ms`
                    : "Lokal & privat"}
                </span>
              </div>
            </div>
            <div className="scanner-controls">
              <button
                className={`button ${hasSource || loading ? "button-stop" : "button-forest"}`}
                onClick={hasSource || loading ? scanner.stop : scanner.start}
              >
                {hasSource || loading ? (
                  <Square size={15} fill="currentColor" />
                ) : (
                  <Camera size={17} />
                )}{" "}
                {hasSource || loading ? "Scan beenden" : "Kamera starten"}
              </button>
              <button className="example-button" onClick={() => setDialog("examples")}>
                {isExample ? <RotateCcw size={15} /> : <Eye size={16} />}{" "}
                {isExample ? "Anderes Beispiel" : "Erst mal ein Beispiel ansehen"}
                <ArrowUpRight size={14} />
              </button>
            </div>
            {isExample && scanner.example && (
              <div className="example-navigation">
                <button
                  className="icon-button"
                  onClick={() =>
                    chooseExample(
                      examples[(currentExampleIndex - 1 + examples.length) % examples.length],
                    )
                  }
                  disabled={examples.length < 2}
                  aria-label="Vorheriges Beispiel"
                >
                  <ChevronLeft size={18} />
                </button>
                <div>
                  <strong>{scanner.example.title}</strong>
                  <span>{scanner.example.description}</span>
                </div>
                <button
                  className="icon-button"
                  onClick={() =>
                    chooseExample(examples[(currentExampleIndex + 1) % examples.length])
                  }
                  disabled={examples.length < 2}
                  aria-label="Nächstes Beispiel"
                >
                  <ChevronRight size={18} />
                </button>
              </div>
            )}
          </section>

          <aside className="scan-panel">
            <div className="panel-heading">
              <h2>Dein Überblick</h2>
              <span className="panel-tag">
                {isExample ? "Beispiel" : live ? "Aktueller Scan" : "Live-Auswertung"}
              </span>
            </div>
            <p className={`ready-summary ${scanner.tracks.some((track) => track.readyToEat) ? "has-ready" : ""}`}
              role="status" aria-live="polite">{recommendation(scanner.tracks)}</p>
            {hasSource && scanner.modelReady && (!scanner.tracks.length || scanner.tracks.some((track) => !track.readyToEat && track.ripeness !== "unreif" && track.ripeness !== "überreif")) &&
              <p className="recognition-hint">Nicht alle Früchte erkannt? Nimm den Büschel näher oder von der Seite auf, sodass einzelne Bananen sichtbar sind.</p>}
            <FruitList tracks={scanner.tracks} selectedId={selectedId} onSelect={setSelectedId} />
            <div className="total-count">
              <span className="total-number">
                {scanner.tracks.length.toString().padStart(2, "0")}
              </span>
              <div>
                <strong>
                  {scanner.tracks.length === 1 ? "Banane erkannt" : "Bananen erkannt"}
                </strong>
                <span>
                  {hasSource ? "im aktuellen Bild" : "Dein nächster Einkauf kann kommen."}
                </span>
              </div>
              <div className="count-icon">
                <Banana size={24} strokeWidth={1.4} />
              </div>
            </div>
            <div className="ripeness-counts">
              {STAGES.map((stage) => (
                <div className="ripeness-row" key={stage.id}>
                  <span className={`ripeness-symbol ${stage.color}`}>
                    {stage.id === "unreif" ? (
                      <Leaf size={17} />
                    ) : stage.id === "reif" ? (
                      <Sun size={18} />
                    ) : (
                      <Eye size={18} />
                    )}
                  </span>
                  <div>
                    <strong>{stage.subtitle}</strong>
                    <span>
                      {stage.id === "unreif"
                        ? "Braucht noch Zeit"
                        : stage.id === "reif"
                          ? "Bereit zum Genießen"
                          : "Extra süß & weich"}
                    </span>
                  </div>
                  <span className="ripeness-count">{counts[stage.id]}</span>
                </div>
              ))}
            </div>
            <div className="uncertain-row">
              <span>
                <CircleHelp size={15} />
                Unsicher{" "}
                <button
                  className="hint-button"
                  onClick={() => setDialog("quality")}
                  aria-label="Mehr zu unsicheren Ergebnissen"
                >
                  <CircleHelp size={12} />
                </button>
              </span>
              <strong>{counts.unsicher}</strong>
            </div>
            <div className="confidence-summary">
              <span>Ø Modellkonfidenz</span>
              <strong>{average ? `${Math.round(average * 100)} %` : "–"}</strong>
              <div className="confidence-track">
                <span style={{ width: `${average * 100}%` }} />
              </div>
            </div>
            <div className="panel-tip">
              <span className="tip-icon">
                <Sun size={18} />
              </span>
              <div>
                <strong>Ein bisschen Licht hilft.</strong>
                <p>Halte dein Handy ruhig und achte auf natürliches, gleichmäßiges Licht.</p>
              </div>
            </div>
            <button className="panel-help" onClick={() => setDialog("guide")}>
              So gelingt dein Scan <ArrowUpRight size={15} />
            </button>
            <span className="panel-disclaimer">
              <Eye size={13} /> Bewertet wird die sichtbare Schalenreife.
            </span>
          </aside>
        </div>

        {scanner.notice && !hasSource && (
          <div className="inline-notice" role="status">
            <ShieldCheck size={16} />
            {scanner.notice}
          </div>
        )}
        <section className="ripeness-guide" id="reifestufen">
          <div className="section-heading">
            <div>
              <span className="eyebrow">VON GRÜN BIS GOLDGELB</span>
              <h2>Jede Reife hat ihren Moment.</h2>
            </div>
            <button className="text-link" onClick={() => setDialog("guide")}>
              Die Reifestufen kennenlernen <ArrowRight size={16} />
            </button>
          </div>
          <div className="stage-cards">
            {STAGES.map((stage) => (
              <button
                className={`stage-card ${stage.color}`}
                key={stage.id}
                onClick={() => setDialog("guide")}
              >
                <div className="stage-card-top">
                  <span className={`stage-chip ${stage.color}`}>
                    <span />
                    {stage.subtitle}
                  </span>
                  <span className="stage-number">{stage.number}</span>
                </div>
                <BananaArt stage={stage.id} />
                <h3>{stage.title}</h3>
                <p>{stage.description}</p>
                <span className="stage-link" aria-label={`Mehr über ${stage.subtitle}`}>
                  <ArrowUpRight size={17} />
                </span>
              </button>
            ))}
          </div>
        </section>
        <section className="privacy-strip">
          <div className="privacy-strip-icon">
            <ShieldCheck size={25} strokeWidth={1.5} />
          </div>
          <div>
            <h3>Ein guter Blick braucht keine Cloud.</h3>
            <p>
              Deine Kamerabilder bleiben im Arbeitsspeicher deines Handys. Kein Upload. Keine
              Speicherung.
            </p>
          </div>
          <button onClick={() => setDialog("privacy")}>
            So schützen wir deine Bilder <ArrowUpRight size={15} />
          </button>
        </section>
      </main>
      <footer>
        <a href={assetPath("/")} className="footer-brand">
          <Banana size={17} /> bananenblick.
        </a>
        <span>Ein kleiner Blick. Ein bewussterer Einkauf.</span>
        <div>
          <span className="prototype-dot" /> Forschungsprototyp{" "}
          <button onClick={() => setDialog("quality")}>
            Modell & Teststand <ArrowUpRight size={12} />
          </button>
        </div>
      </footer>

      {dialog === "guide" && (
        <Modal
          title="So wird aus deinem Blick ein Scan."
          eyebrow="EINFACH LOSLEGEN"
          onClose={() => setDialog(null)}
          wide
        >
          <div className="guide-steps">
            <div>
              <span>01</span>
              <Camera />
              <h3>Kamera freigeben</h3>
              <p>
                Tippe auf „Kamera starten“ und erlaube den Zugriff. Auf dem Handy wird bevorzugt die
                Rückkamera genutzt.
              </p>
            </div>
            <div>
              <span>02</span>
              <Focus />
              <h3>Bananen ins Bild</h3>
              <p>
                Halte etwas Abstand. Einzelne Früchte sollten gut sichtbar sein. Bei Büscheln und
                Überlappungen kann die Erkennung Früchte übersehen.
              </p>
            </div>
            <div>
              <span>03</span>
              <ScanLine />
              <h3>Reife ablesen</h3>
              <p>
                Farbe und Beschriftung zeigen die Einschätzung. „Unsicher“ bedeutet, dass das Modell
                keine klare Zuordnung trifft.
              </p>
            </div>
          </div>
          <div className="guide-stages">
            {STAGES.map((stage) => (
              <div key={stage.id}>
                <BananaArt stage={stage.id} />
                <div>
                  <strong>{stage.subtitle}</strong>
                  <p>{stage.description}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="info-callout">
            <CircleHelp size={19} />
            <p>
              Schalenreife ist eine visuelle Einschätzung. Das Modell erkennt weder Verderb noch
              innere Qualität. Konfidenzwerte sind Modellwerte und keine garantierte
              Trefferwahrscheinlichkeit.
            </p>
          </div>
          <button
            className="button button-forest modal-primary"
            onClick={() => {
              setDialog(null);
              scanner.start();
            }}
          >
            <Camera size={17} /> Kamera starten <ArrowRight size={16} />
          </button>
        </Modal>
      )}
      {dialog === "privacy" && (
        <Modal
          title="Deine Bilder bleiben deine Bilder."
          eyebrow="LOKAL. FLÜCHTIG. PRIVAT."
          onClose={() => setDialog(null)}
        >
          <div className="privacy-hero">
            <ShieldCheck size={38} />
            <p>Die gesamte Bildauswertung läuft im Browser auf deinem Gerät.</p>
          </div>
          <div className="detail-list">
            <div>
              <Check />
              <p>
                <strong>Kein Bild-Upload</strong>Videoframes gehen direkt aus dem Kamerastream an
                die lokale Modelllaufzeit.
              </p>
            </div>
            <div>
              <Check />
              <p>
                <strong>Keine Aufnahmen</strong>Es gibt weder Fotoarchiv noch Videoaufzeichnung.
                Bilddaten bleiben ausschließlich im Arbeitsspeicher.
              </p>
            </div>
            <div>
              <Check />
              <p>
                <strong>Du behältst die Kontrolle</strong>„Scan beenden“ gibt die Kamera frei und
                beendet die lokale Erkennung. Auch beim Wechsel in eine andere App stoppt die
                Kamera.
              </p>
            </div>
            <div>
              <Download />
              <p>
                <strong>Einmal laden, lokal auswerten</strong>App, ONNX-Modell und Laufzeit werden
                über HTTPS geladen. Diese GitHub-Pages-Ausgabe enthält keine Anmeldung oder Zugangsbeschränkung.
              </p>
            </div>
          </div>
          <p className="fine-print">
            Der technische Prüfbericht dokumentiert die Netzwerkregeln und den noch offenen Nachweis
            auf echten Geräten.
          </p>
        </Modal>
      )}
      {dialog === "settings" && (
        <Modal
          title="Ein klarer Blick nach deinem Maß."
          eyebrow="ERKENNUNG EINSTELLEN"
          onClose={() => setDialog(null)}
        >
          <div className="setting-title">
            <span>Grenze für eine klare Reifestufe</span>
            <strong>{Math.round(scanner.classification * 100)} %</strong>
          </div>
          <input
            type="range"
            min={Math.round((modelCard?.thresholds.classification ?? DEFAULT_THRESHOLDS.classification) * 100)}
            max="90"
            step="5"
            value={Math.round(scanner.classification * 100)}
            onChange={(event) => scanner.setConfidence(Number(event.target.value) / 100)}
            aria-label="Konfidenzgrenze für klare Einstufungen"
          />
          <div className="range-labels">
            <span>Mehr Zuordnungen</span>
            <span>Zurückhaltender</span>
          </div>
          <p className="setting-description">
            Unter dieser Modellkonfidenz erscheint „unsicher“. Zusätzlich muss sich die beste Klasse
            deutlich von der zweitbesten unterscheiden. Die Anzahl erkannter Früchte wird davon
            nicht beeinflusst.
          </p>
          <div className="runtime-card">
            <Cpu size={22} />
            <div>
              <strong>{scanner.modelReady ? runtime : "Automatische Gerätewahl"}</strong>
              <p>WebGPU, wenn verfügbar. Sonst WebAssembly.</p>
            </div>
          </div>
          <button
            className="button button-light"
            onClick={() => scanner.setConfidence(modelCard?.thresholds?.classification ?? DEFAULT_THRESHOLDS.classification)}
          >
            <RotateCcw size={15} /> Standard wiederherstellen
          </button>
        </Modal>
      )}
      {dialog === "examples" && (
        <Modal
          title="Erst schauen. Dann selbst scannen."
          eyebrow="ECHTE BILDER. ECHTE MODELLAUSWERTUNG."
          onClose={() => setDialog(null)}
          wide
        >
          <p className="modal-lead">
            Diese Testbilder werden mit demselben lokalen Modell ausgewertet. Sie sind ausdrücklich
            keine Live-Kamerabilder.
          </p>
          <div className="example-grid">
            {examples.map((example) => (
              <button
                key={example.id}
                className="example-tile"
                onClick={() => chooseExample(example)}
              >
                <img src={assetPath(example.src)} alt={example.title} />
                <div>
                  <strong>{example.title}</strong>
                  <ArrowUpRight size={17} />
                </div>
                <p>{example.description}</p>
              </button>
            ))}
          </div>
          {!examples.length && (
            <p role="status">
              {exampleError
                ? "Die Beispielbilder konnten nicht geladen werden. Bitte lade die Seite erneut."
                : "Beispielbilder werden geladen …"}
            </p>
          )}
          <p className="fine-print">
            Quelle:{" "}
            <a
              href="https://data.mendeley.com/datasets/56c75hvm7r/1"
              target="_blank"
              rel="noreferrer"
            >
              Annotated Banana Ripeness Stage Dataset
            </a>
            , Rohit et al., CC BY 4.0. Die Bilder stammen aus dem getrennten Testbestand.
            {" "}Büschelbeispiel: <a href="https://www.flickr.com/photos/scaredykat/4430551771/" target="_blank" rel="noreferrer">goodiesfirst · putahracsa fruit plate</a>,
            {" "}<a href="https://creativecommons.org/licenses/by/2.0/" target="_blank" rel="noreferrer">CC BY 2.0</a> (verkleinert, via COCO 2017).
          </p>
        </Modal>
      )}
      {dialog === "quality" && (
        <Modal
          title="Ein ehrlicher Blick aufs Modell."
          eyebrow="MODELL & QUALITÄT"
          onClose={() => setDialog(null)}
          wide
        >
          <div className="quality-summary">
            <div className="quality-icon">
              <Cpu size={28} />
            </div>
            <div>
              <h3>YOLOX-Nano · 3 Reifestufen</h3>
              <p>
                Auf öffentlichen Bananenbildern trainiert. ONNX Runtime Web führt das Modell lokal
                aus.
              </p>
            </div>
            <span className="research-tag">Forschungsprototyp</span>
          </div>
          <div className="quality-grid">
            <div>
              <span>Verarbeitung</span>
              <strong>{scanner.provider ? runtime : "WebGPU / WebAssembly"}</strong>
              <p>Automatischer Rückfall auf WebAssembly</p>
            </div>
            <div>
              <span>Aktueller Live-Durchsatz</span>
              <strong>
                {live && scanner.stats.fps
                  ? `${scanner.stats.fps.toFixed(1).replace(".", ",")} Bilder/s`
                  : "Noch nicht gemessen"}
              </strong>
              <p>Ziel: mindestens 3 Auswertungen/s</p>
            </div>
          </div>
          <h3 className="modal-section-title">Einzelbananen in realen Mehrfruchtszenen</h3>
          {modelCard?.individualBenchmark && <div className="info-callout">
            <CircleHelp size={19} />
            <p><strong>{modelCard.individualBenchmark.passed ? "Zielwerte im Fototest erreicht." : "Zielwerte im Fototest nicht erreicht."}</strong>{" "}
              {modelCard.individualBenchmark.positiveScenes} Mehrfruchtszenen und {modelCard.individualBenchmark.negativeScenes} Negativszenen:{" "}
              {(modelCard.individualBenchmark.precision * 100).toFixed(1).replace(".", ",")} % Präzision,{" "}
              {(modelCard.individualBenchmark.recall * 100).toFixed(1).replace(".", ",")} % Erkennungsquote für bestätigte essreife Einzelbananen.
              Ziel: mindestens 95 % und 80 %. Unsicherheit zählt als ausgelassene Empfehlung.</p>
          </div>}
          <h3 className="modal-section-title">Bisheriger Testbestand mit überwiegend Einzelaufnahmen</h3>
          <p className="setting-description">
            Die Erkennungsquote misst, wie viele markierte Früchte korrekt gefunden und eingestuft
            werden (IoU ≥ 0,5). Die Rohwerte zählen die beste Modellklasse auch bei „unsicher“.
            „Sicher beschriftet“ zählt nur korrekte, sicher angezeigte Reifestufen. Alle Werte
            gelten für die Standardschwellwerte.
          </p>
          {qualityMetrics ? (
            <div className="metrics-table">
              <table>
                <thead>
                  <tr>
                    <th>Reifestufe</th>
                    <th>Präzision</th>
                    <th>Erkennungsquote</th>
                    <th>Sicher beschriftet</th>
                  </tr>
                </thead>
                <tbody>
                  {qualityMetrics.map((metric, index) => (
                    <tr key={index}>
                      <td>{metric.class}</td>
                      <td>
                        {typeof metric.precision === "number"
                          ? `${(metric.precision * 100).toFixed(1).replace(".", ",")} %`
                          : "–"}
                      </td>
                      <td>
                        {typeof metric.recall === "number"
                          ? `${(metric.recall * 100).toFixed(1).replace(".", ",")} %`
                          : "–"}
                      </td>
                      <td>
                        {typeof metric.visibleRecall === "number"
                          ? `${(metric.visibleRecall * 100).toFixed(1).replace(".", ",")} %`
                          : "–"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="info-callout">
              Der Messbericht wird geladen. Unbestätigte Genauigkeitswerte werden hier nicht
              angezeigt.
            </p>
          )}
          <div className="info-callout">
            <CircleHelp size={19} />
            <p>
              <strong>Die Geräteabnahme ist noch offen.</strong> Das Nachtraining und der neue Fototest
              belegen noch keine zuverlässige Erkennung jeder einzelnen Banane in dichten Büscheln. Kamerazugriff, Drehung,
              Bewegung, Rahmenpositionen, ≥ 3 Bilder/s und Netzwerkverkehr müssen auf echten
              Android- und iPhone-Geräten geprüft werden. Beispielbilder ersetzen diesen Nachweis
              nicht.
            </p>
          </div>
          <div className="quality-actions">
            <a className="button button-forest" href={assetPath("/evaluation/individual-report.json")} download>
              <Download size={16} /> Modellbericht
            </a>
            <button
              className="button button-light"
              onClick={() => downloadJson(scanner.report(), "bananenblick-browser-messung.json")}
            >
              <Download size={16} /> Browser-Messung
            </button>
            <a
              className="text-link"
              href={assetPath("/evaluation/acceptance.md")}
              target="_blank"
              rel="noreferrer"
            >
              Prüfprotokoll <ArrowUpRight size={14} />
            </a>
          </div>
          <p className="fine-print">
            Trainingsquelle:{" "}
            <a
              href="https://data.mendeley.com/datasets/56c75hvm7r/1"
              target="_blank"
              rel="noreferrer"
            >
              Rohit et al. · Annotated Banana Ripeness Stage Dataset
            </a>{" "}
            (CC BY 4.0). Modellarchitektur:{" "}
            <a
              href="https://github.com/Megvii-BaseDetection/YOLOX"
              target="_blank"
              rel="noreferrer"
            >
              Megvii YOLOX
            </a>{" "}
            (Apache 2.0). Keine Zusage für Lebensmittelqualität.
          </p>
        </Modal>
      )}
    </>
  );
}
