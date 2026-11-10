"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { updateTracks, type FrameResult, type Track } from "./detection";
import { assetPath } from "./assets";

export type ScannerMode = "idle" | "starting" | "live" | "example" | "error";
export type Example = {
  id: string;
  src: string;
  title: string;
  description: string;
  width: number;
  height: number;
  sourceUrl?: string;
};
const cameraMessage = (error: unknown) => {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "PermissionDeniedError")
    return "Die Kamera ist nicht freigegeben. Erlaube den Kamerazugriff in den Website-Einstellungen deines Browsers und starte erneut.";
  if (name === "NotFoundError" || name === "DevicesNotFoundError")
    return "Keine Kamera gefunden. Öffne Bananenblick auf einem Handy mit Kamera oder teste ein Beispielbild.";
  if (name === "NotReadableError" || name === "TrackStartError")
    return "Die Kamera ist gerade belegt. Schließe andere Kamera-Apps und versuche es erneut.";
  return "Die Kamera konnte nicht gestartet werden. Prüfe die Kamerafreigabe und versuche es erneut.";
};
export function useScanner() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sessionRef = useRef(0);
  const modeRef = useRef<ScannerMode>("idle");
  const tracksRef = useRef<Track[]>([]);
  const nextTrackRef = useRef(1);
  const readyRef = useRef(false);
  const busyRef = useRef(false);
  const sizeRef = useRef(320);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const vfcRef = useRef<number | null>(null);
  const frameRef = useRef(0);
  const clearAfterFrameRef = useRef(0);
  const examplePassesRef = useRef(0);
  const thresholdRef = useRef(0.6);
  const userThresholdRef = useRef(false);
  const captureRef = useRef<() => void>(() => {});
  const loopStartRef = useRef(0);
  const lastResultRef = useRef(0);
  const lastVideoTimeRef = useRef({ videoTime: 0, changedAt: 0 });
  const lastCapturedVideoTimeRef = useRef(-1);
  const sourceDimensionsRef = useRef({ width: 0, height: 0 });
  const samplesRef = useRef<{ at: number; inference: number; age: number }[]>([]);
  const [mode, setMode] = useState<ScannerMode>("idle");
  const [tracks, setTracks] = useState<Track[]>([]);
  const [modelStatus, setModelStatus] = useState("Modell startet mit deinem Scan");
  const [modelReady, setModelReady] = useState(false);
  const [provider, setProvider] = useState("");
  const [error, setError] = useState("");
  const [example, setExample] = useState<Example | null>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [stats, setStats] = useState({
    fps: 0,
    inferenceMs: 0,
    ageMs: 0,
    frames: 0,
    capturedAt: 0,
  });
  const [classification, setClassification] = useState(0.6);
  const [notice, setNotice] = useState("");
  const changeMode = useCallback((value: ScannerMode) => {
    modeRef.current = value;
    setMode(value);
  }, []);
  const clearTracks = useCallback(() => {
    clearAfterFrameRef.current = frameRef.current;
    tracksRef.current = [];
    setTracks([]);
  }, []);
  const cancelLoop = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (vfcRef.current !== null) videoRef.current?.cancelVideoFrameCallback?.(vfcRef.current);
    timerRef.current = null;
    vfcRef.current = null;
  }, []);
  const stop = useCallback(() => {
    sessionRef.current++;
    cancelLoop();
    streamRef.current?.getTracks().forEach((track) => {
      track.onended = null;
      track.onmute = null;
      track.onunmute = null;
      track.stop();
    });
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
    workerRef.current?.terminate();
    workerRef.current = null;
    readyRef.current = false;
    busyRef.current = false;
    setModelReady(false);
    setProvider("");
    setModelStatus("Modell startet mit deinem Scan");
    if (canvasRef.current) {
      canvasRef.current.width = 0;
      canvasRef.current.height = 0;
    }
    canvasRef.current = null;
    clearTracks();
    setExample(null);
    changeMode("idle");
    setError("");
    setNotice("");
    setStats({ fps: 0, inferenceMs: 0, ageMs: 0, frames: 0, capturedAt: 0 });
    samplesRef.current = [];
    frameRef.current = 0;
    clearAfterFrameRef.current = 0;
    examplePassesRef.current = 0;
    nextTrackRef.current = 1;
    lastCapturedVideoTimeRef.current = -1;
    sourceDimensionsRef.current = { width: 0, height: 0 };
  }, [cancelLoop, changeMode, clearTracks]);
  const fail = useCallback(
    (message: string) => {
      stop();
      changeMode("error");
      setError(message);
    },
    [changeMode, stop],
  );
  const schedule = useCallback(() => {
    cancelLoop();
    if (modeRef.current !== "live" || !readyRef.current) return;
    const delay = Math.max(0, 200 - (performance.now() - loopStartRef.current));
    timerRef.current = setTimeout(() => {
      const video = videoRef.current;
      if (video?.requestVideoFrameCallback)
        vfcRef.current = video.requestVideoFrameCallback(() => {
          vfcRef.current = null;
          captureRef.current();
        });
      else captureRef.current();
    }, delay);
  }, [cancelLoop]);
  const capture = useCallback(() => {
    if (!readyRef.current || busyRef.current || !workerRef.current) return;
    const video = videoRef.current;
    const image = imageRef.current;
    const live = modeRef.current === "live";
    if (modeRef.current !== "live" && modeRef.current !== "example") return;
    const source = live ? video : image;
    const width = live ? video?.videoWidth : image?.naturalWidth;
    const height = live ? video?.videoHeight : image?.naturalHeight;
    if (!source || !width || !height || (live && (!video || video.readyState < 2))) {
      if (live) schedule();
      return;
    }
    if (live && video) {
      if (video.currentTime === lastCapturedVideoTimeRef.current) {
        schedule();
        return;
      }
      lastCapturedVideoTimeRef.current = video.currentTime;
    }
    const size = sizeRef.current;
    const canvas = canvasRef.current ?? document.createElement("canvas");
    canvasRef.current = canvas;
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) {
      fail("Dein Browser unterstützt die lokale Bildverarbeitung nicht.");
      return;
    }
    const ratio = Math.min(size / width, size / height);
    loopStartRef.current = performance.now();
    const capturedAt = Date.now();
    context.fillStyle = "rgb(114,114,114)";
    context.fillRect(0, 0, size, size);
    context.drawImage(source, 0, 0, Math.floor(width * ratio), Math.floor(height * ratio));
    const pixels = context.getImageData(0, 0, size, size);
    busyRef.current = true;
    workerRef.current.postMessage(
      {
        type: "frame",
        sessionId: sessionRef.current,
        frameId: ++frameRef.current,
        capturedAt,
        width,
        height,
        rgba: pixels.data.buffer,
        classification: thresholdRef.current,
      },
      [pixels.data.buffer],
    );
  }, [fail, schedule]);
  useEffect(() => {
    captureRef.current = capture;
  }, [capture]);
  const prepareWorker = useCallback(() => {
    let worker: Worker;
    try {
      worker = new Worker(new URL("./inference.worker.ts", import.meta.url), { type: "module" });
    } catch {
      fail(
        "Die lokale Modelllaufzeit konnte nicht gestartet werden. Bitte öffne die Website direkt in einem aktuellen Safari- oder Chrome-Browser.",
      );
      return;
    }
    workerRef.current = worker;
    worker.onerror = () =>
      fail(
        "Die lokale Erkennung konnte nicht gestartet werden. Lade die Seite neu oder versuche einen aktuellen Safari- oder Chrome-Browser.",
      );
    worker.onmessage = (event) => {
      if (workerRef.current !== worker) return;
      const data = event.data;
      if (data.type === "status") setModelStatus(data.message);
      if (data.type === "provider") setProvider(data.provider);
      if (data.type === "error") {
        fail(data.message);
        return;
      }
      if (data.type === "ready") {
        readyRef.current = true;
        sizeRef.current = data.inputSize;
        setModelReady(true);
        setProvider(data.provider);
        if (!userThresholdRef.current) {
          thresholdRef.current = data.thresholds.classification;
          setClassification(data.thresholds.classification);
        }
        setModelStatus("YOLOX-Nano bereit");
        captureRef.current();
      }
      if (data.type !== "result") return;
      const result = data as FrameResult;
      busyRef.current = false;
      if (result.sessionId !== sessionRef.current) return;
      const live = modeRef.current === "live";
      if (!live && modeRef.current !== "example") return;
      if (result.frameId <= clearAfterFrameRef.current) {
        if (live) schedule();
        else captureRef.current();
        return;
      }
      if (Math.abs(result.classification - thresholdRef.current) > 0.001) {
        if (live) schedule();
        else captureRef.current();
        return;
      }
      const age = Date.now() - result.capturedAt;
      if (
        live &&
        (videoRef.current?.videoWidth !== result.width ||
          videoRef.current?.videoHeight !== result.height)
      ) {
        clearTracks();
        schedule();
        return;
      }
      const now = performance.now();
      if (
        sourceDimensionsRef.current.width &&
        (sourceDimensionsRef.current.width !== result.width ||
          sourceDimensionsRef.current.height !== result.height)
      ) {
        tracksRef.current = [];
      }
      sourceDimensionsRef.current = { width: result.width, height: result.height };
      const next = updateTracks(
        tracksRef.current,
        !live || age <= 900 ? result.detections : [],
        live ? now : result.frameId * 200,
        () => nextTrackRef.current++,
      );
      tracksRef.current = next;
      setTracks(next.filter((track) => track.visible));
      setDimensions({ width: result.width, height: result.height });
      lastResultRef.current = now;
      samplesRef.current.push({ at: now, inference: result.inferenceMs, age });
      if (samplesRef.current.length > 3600)
        samplesRef.current.splice(0, samplesRef.current.length - 3600);
      const recent = samplesRef.current.filter((sample) => now - sample.at < 5000);
      const fps =
        recent.length > 1 ? ((recent.length - 1) * 1000) / (recent.at(-1)!.at - recent[0].at) : 0;
      setStats({
        fps: live ? fps : 0,
        inferenceMs: result.inferenceMs,
        ageMs: age,
        frames: frameRef.current,
        capturedAt: result.capturedAt,
      });
      if (live) schedule();
      else if (++examplePassesRef.current < 3)
        timerRef.current = setTimeout(() => captureRef.current(), 200);
    };
    worker.postMessage({ type: "init", assetBase: new URL(assetPath(""), document.baseURI).href });
  }, [clearTracks, fail, schedule]);
  const start = useCallback(async () => {
    stop();
    changeMode("starting");
    setModelStatus("Kamera wird vorbereitet …");
    const token = sessionRef.current;
    if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
      fail(
        "Der Kamerazugriff benötigt HTTPS und einen aktuellen Browser. Öffne die sichere Adresse direkt in Safari oder Chrome.",
      );
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      if (sessionRef.current !== token) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      const track = stream.getVideoTracks()[0];
      track.onended = () =>
        fail(
          "Die Kamera wurde unterbrochen. Tippe auf „Erneut starten“, um sie wieder zu verbinden.",
        );
      track.onmute = () => {
        clearTracks();
        setNotice("Kamerabild kurz unterbrochen …");
        track.onunmute = () => {
          setNotice("");
          captureRef.current();
        };
      };
      const video = videoRef.current;
      if (!video) {
        stream.getTracks().forEach((t) => t.stop());
        fail("Die Kameraansicht ist noch nicht bereit. Bitte erneut starten.");
        return;
      }
      video.srcObject = stream;
      await video.play();
      if (sessionRef.current !== token) return;
      setDimensions({ width: video.videoWidth, height: video.videoHeight });
      changeMode("live");
      setModelStatus("Lokales Modell wird geladen …");
      prepareWorker();
      lastVideoTimeRef.current = { videoTime: video.currentTime, changedAt: performance.now() };
    } catch (error) {
      if (sessionRef.current === token) fail(cameraMessage(error));
    }
  }, [changeMode, clearTracks, fail, prepareWorker, stop]);
  const showExample = useCallback(
    (value: Example) => {
      stop();
      setExample(value);
      changeMode("example");
      setModelStatus("Beispiel wird vorbereitet …");
      prepareWorker();
    },
    [changeMode, prepareWorker, stop],
  );
  const setConfidence = useCallback(
    (value: number) => {
      userThresholdRef.current = true;
      const threshold = Math.max(0.6, Math.min(0.9, value));
      thresholdRef.current = threshold;
      setClassification(threshold);
      examplePassesRef.current = 0;
      clearTracks();
      if (modeRef.current === "example") {
        cancelLoop();
        captureRef.current();
      }
    },
    [cancelLoop, clearTracks],
  );
  useEffect(() => {
    const interval = setInterval(() => {
      if (modeRef.current !== "live") return;
      const video = videoRef.current;
      const now = performance.now();
      if (video && video.currentTime !== lastVideoTimeRef.current.videoTime) {
        lastVideoTimeRef.current = { videoTime: video.currentTime, changedAt: now };
        setNotice("");
      } else if (
        lastVideoTimeRef.current.changedAt &&
        now - lastVideoTimeRef.current.changedAt > 2200
      ) {
        setNotice("Das Kamerabild ist unterbrochen. Beende den Scan und starte die Kamera erneut.");
      }
      if (tracksRef.current.some((track) => Date.now() - track.capturedAt > 1200)) clearTracks();
      if (performance.now() - lastResultRef.current > 900) {
        if (tracksRef.current.length) clearTracks();
        setStats((old) => (old.fps ? { ...old, fps: 0 } : old));
      }
    }, 100);
    const hidden = () => {
      if (document.hidden && (modeRef.current === "live" || modeRef.current === "starting")) {
        stop();
        setNotice("Kamera beim Verlassen der App gestoppt.");
      }
    };
    const unload = () => stop();
    const geometry = () => {
      // A static example keeps its source coordinates when only the viewport rotates.
      if (modeRef.current !== "live") return;
      clearTracks();
      const v = videoRef.current;
      if (v?.videoWidth) setDimensions({ width: v.videoWidth, height: v.videoHeight });
    };
    document.addEventListener("visibilitychange", hidden);
    window.addEventListener("pagehide", unload);
    window.addEventListener("orientationchange", geometry);
    return () => {
      clearInterval(interval);
      document.removeEventListener("visibilitychange", hidden);
      window.removeEventListener("pagehide", unload);
      window.removeEventListener("orientationchange", geometry);
      stop();
    };
  }, [clearTracks, stop]);
  const report = useCallback(
    () => ({
      generatedAt: new Date().toISOString(),
      device: navigator.userAgent,
      provider,
      source: modeRef.current,
      thresholds: { classification: thresholdRef.current },
      measurements: samplesRef.current,
      capturedFrames: frameRef.current,
      storedImages: 0,
      note: "Messungen dieses Browsers; keine Bestätigung echter Android-/iPhone-Gerätetests. Keine Kamerabilder enthalten.",
    }),
    [provider],
  );
  return {
    videoRef,
    imageRef,
    mode,
    tracks,
    modelStatus,
    modelReady,
    provider,
    error,
    example,
    dimensions,
    stats,
    classification,
    notice,
    start,
    stop,
    showExample,
    capture,
    setConfidence,
    clearTracks,
    report,
    fail,
  };
}
