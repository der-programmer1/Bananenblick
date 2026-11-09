# Prüf- und Abnahmeprotokoll – Bananenblick

Der Prototyp verwendet ein tatsächlich trainiertes YOLOX-Nano-Modell. Die endgültige Geräteabnahme ist noch offen. Dieses Dokument unterscheidet ausgeführte Prüfungen von geplanten Prüfungen; es enthält keine erfundenen Geräte- oder Kameraergebnisse.

## Tatsächlich ausgeführte Modell- und Laufzeitprüfungen (9. November 2026)

- 16 CPU-Trainingsepochen; Checkpoint aus Epoche 15 ausschließlich auf Validierung gewählt. Finaler Detektionsschwellwert 0,50.
- 471 Trainings-, 101 Validierungs-, 101 Testbilder. 192 erkannte ähnliche Bildpaare in 503 unteilbare Gruppen gefasst; keine erkannten gruppenübergreifenden Duplikate. Unbekannte Frucht-/Aufnahmeserien bleiben eine Grenze des Audits.
- Einmaliger ONNX-Test: mAP50 93,95 %, Rohpräzision und Roherkennungsquote je 95,05 %, sicher sichtbare richtige Reifestufen 92,08 %, drei unsichere Treffer. Pro Klasse stehen auch Fehlklassen, Falschalarme und übersehene Früchte im [Modellbericht](./model-report.md).
- Zehn begrenzte Stresskontrollen wurden ausgeführt; die helle Variante einer reifen Banane wurde nicht ausreichend genau lokalisiert (Box-IoU unter 0,50 bei korrekter Klasse). Die vier Negativkontrollen erzeugten keine Falschalarme. Positive Montagen und Helligkeitsvarianten sind synthetisch, ein Negativfoto ist real. Dies belegt keine allgemeine Robustheit.
- ONNX Runtime Web 1.24.3, JSEP-WASM, Node/Linux, ein Thread: drei unterschiedliche Testbilder erzeugen endliche `[1,2100,8]`-Ausgaben und je eine Detektion; drei zusätzliche homogene Kontrollen erzeugen keine Rahmen. Gemessener Inferenzmedian 54,3 ms, p95 374,6 ms bei 15 Messungen. Ausschließlich Modellaufruf; kein Handy-/Browser-/Kamera-Benchmark.
- Sieben TypeScript-Tests für Decoder, Padding, NMS, Unsicherheit, beide Orientierungen und Stabilisierung bestanden. Sechs Python-Pipelinechecks bestanden. TypeScript-Prüfung bestanden.
- Tatsächlicher Modell-Hash: `75ced29a13b890afaf328751289567fe48ea8471865c637a91244506d6ec6419`.

Vollständige Nachweise: [Modell-Test](./test-report.json), [Web-Laufzeit](./web-runtime.json), [Stresskontrollen](./stress-report.json), [Split-Audit](./split-audit.json). Keine der Zahlen belegt die Herkunft aus einer echten laufenden Handykamera. Dieser Nachweis folgt erst mit den unten beschriebenen Gerätetests.

## In dieser Entwicklungsumgebung überprüfbar

| Prüfpunkt | Nachweis / Status |
|---|---|
| Getrennte Trainings-/Validierungs-/Testdaten | CPU-Pipeline und Audit im Modellbericht; SHA-/Pixel-/pHash-Duplikatgruppen, Grenzen des Ähnlichkeitsverfahrens dokumentiert |
| Trainiertes YOLOX-Nano als ONNX | Modellkarte, Prüfsumme und tatsächlicher Export unter `/models/` |
| Erkennungsquote und Fehlklassen je Reifestufe | `/evaluation/test-report.json`; Werte beziehen sich auf zurückgehaltene Bilder, nicht auf Kamerastreams |
| Dekodierung und Rahmengeometrie | Automatisierte Tests für Padding, Bildgrenzen, NMS, Contain, Hoch-/Querformat |
| Unsicherheit | Tests für gemeinsame Konfidenz und geringe Klassenmarge |
| Kurzzeitige Stabilisierung | Tests für Identitätszuordnung und Ablauf fehlender Boxen |
| Nur jüngster Frame / keine Warteschlange | Implementierung: ein Inferenzauftrag gleichzeitig; nächste Aufnahme erfolgt erst nach Antwort |
| Keine Bildübertragung / Speicherung | Quellcode und CSP geprüft; Frames nur im Canvas und Worker; keine Bild-API, kein Recorder, kein Bildspeicher |
| ONNX Runtime Web mit tatsächlichen Bilddaten | Separater Bericht `/evaluation/web-runtime.json`, sobald erfolgreich ausgeführt; Node-/Linux-Kontext ausdrücklich ausgewiesen |
| Produktionsbuild und TypeScript | Ausgeführte Prüfungen im Lieferbericht dokumentiert |

Die Browser-Steuerung `cua_repl.js` ist in der angehängten Cloud-Umgebung nicht verfügbar, obwohl die Desktop-Fähigkeit als aktiviert gemeldet wird. Der Sites-Skill untersagt in diesem Fall einen Ersatzbrowser. Daher wurden keine sichtbare Browser-, WebGPU-, Kamera- oder Geräteabnahme behauptet. Umgebung: `ccarenv_b64_Y2NhcmVudl84MjdmZWYxY2UyNGM4MTkxODM1NDIyMGZkMDNjZTcyYw`.

## Noch auszuführen: reale Geräte

Je ein echtes Android-Handy mit aktuellem Chrome und ein echtes iPhone mit aktuellem Safari. Gerät, Betriebssystem, Browserversion, Modell-Hash, Datum und Testperson festhalten. Die private HTTPS-Adresse direkt als Seite öffnen und anmelden.

1. **Freigabe und Stopp:** Zugriff zunächst verweigern. Verständliche Meldung prüfen. Freigabe in Website-Einstellungen erlauben, erneut starten. Rückkamera und laufendes Bild prüfen. „Scan beenden“ antippen; System-Kameraindikator muss erlöschen. Während einer Freigabeanfrage abbrechen und anschließend freigeben: Kein Kameratrack darf weiterlaufen.
2. **Live-Herkunft:** Eine Banane ins Bild halten, dann herausnehmen und durch einen anderen Gegenstand ersetzen. Rahmen müssen mit der Frucht erscheinen, sich mit ihr bewegen und nach ihrem Entfernen verschwinden. Gerät vor dunklen und hellen Hintergründen bewegen. Vorgang extern filmen; keine Videoaufzeichnung in die App einbauen.
3. **Mehrere Früchte:** Mindestens 30 getrennte Szenen mit gemischten grünen und gelben Bananen, überreifen Früchten, Büscheln, starker Überlappung sowie mindestens 30 Szenen ohne Bananen. Referenzboxen für einzelne Früchte manuell erfassen. Für jede Klasse TP, FP, FN und Fehlklassifikationen bei IoU ≥0,5 ermitteln. Büschel als ein Objekt genügen nicht dem Einzelbananen-Nachweis.
4. **Licht und Bewegung:** Tageslicht, künstliches Licht, Schatten, leichte Unschärfe und Kamerabewegung prüfen. Beleuchtungsbedingungen getrennt protokollieren. „Unsicher“ und ausbleibende Erkennungen mitzählen.
5. **Ausrichtung / Rahmen:** Im Livebetrieb mehrfach zwischen Hoch- und Querformat wechseln. Frucht an alle vier Bildränder bewegen, Vollbild öffnen/schließen. Keine überdehnten, versetzten oder gespiegelt wandernden Rahmen akzeptieren. Rahmenalter muss bei ausbleibenden Ergebnissen begrenzt sein.
6. **Unterbrechung:** Zu einer anderen App wechseln, Bildschirm sperren, Telefonat/andere Kamera-App auslösen. Kamera muss frei werden oder die Unterbrechung verständlich anzeigen; erneuter Start muss funktionieren.
7. **Geschwindigkeit:** Nach 10 Sekunden Aufwärmen mindestens 60 Sekunden scannen. Gemessenen Gesamtdurchsatz, Inferenz-Median/p95, Framealter und Backend dokumentieren; Ziel ≥3 Auswertungen/s. Browser-Messbericht herunterladen. WebGPU und erzwungenen WASM-Pfad getrennt prüfen. Linux-Referenzzahlen nicht als Handy-Leistung verwenden.
8. **Netzwerk:** Android per Chrome Remote Debugging, iPhone per Safari Web Inspector prüfen. Nach abgeschlossenen Downloads während eines 60-Sekunden-Scans darf kein Request Kamerapixel enthalten. Worker-/Fetch-/XHR-/WebSocket-Anfragen einschließen. HAR nur ohne Auth-Cookies/Token weitergeben. Lokale Speicher prüfen: keine Kameraaufnahmen in CacheStorage, IndexedDB, LocalStorage oder Dateisystem. Statische Modell-/Beispieldateien dürfen im Browsercache liegen.

## Freigabekriterium

Abnahme erst, wenn die Markierungen nachweisbar aus einem laufenden Kamerastream stammen, beide Gerätetests vorliegen, die Netzwerkprüfung abgeschlossen ist und tatsächliche Genauigkeit sowie Geschwindigkeit dokumentiert sind. Schlechte Testwerte offen ausweisen; fehlende Daten oder Tests nicht durch Beispielbilder ersetzen.

## Ergänzung: GitHub-Pages-ZIP

Der statische Export verwendet dieselben Modellgewichte und dieselbe React-Oberfläche mit einer eigenen Vite-Einstiegsdatei. Alle Bild-/Modell-/Berichtspfade werden relativ zur Website-Basis aufgelöst; der Worker erhält diese Basis explizit. Zwei ergänzende URL-Tests und der statische Build prüfen Root- und Repository-Unterpfade. Insgesamt neun TypeScript-Tests und die TypeScript-Prüfung bestehen. Das GitHub-Workflow-YAML ist mitgeliefert; ein tatsächlicher GitHub-Actions-Lauf und eine sichtbare Browser-/Handyabnahme wurden hier nicht ausgeführt. Diese Ausgabe ist für öffentliche, anonyme Verwendung eingerichtet.
