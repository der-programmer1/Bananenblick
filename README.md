# Bananenblick · nummerierte Einzelbananen

**Stand 0.3.0 – Veröffentlichungsabnahme noch offen.** Die bestehende öffentliche App steht unter [der-programmer1.github.io/Bananenblick](https://der-programmer1.github.io/Bananenblick/). Der neue Stand liegt im Änderungsbranch. Die Zielwerte für essreife Einzelbananen sind nicht erreicht. Tatsächliche Android-/iPhone-Prüfungen fehlen.

Jeder aktuelle Treffer erhält eine eindeutige Nummer. Rahmen und Ergebnisliste verwenden dieselbe Nummer; Antippen wählt die zugehörige Frucht aus. Die globale eindeutige Zuordnung hält Nummern bei leichter Bewegung stabil. Beim Neustart beginnt die Nummerierung wieder bei 1.

Ausschließlich drei aufeinanderfolgende sichere Bewertungen als „reif“ ergeben einen grünen Rahmen und „✓ Jetzt essreif“. Unsicherheit, ein verschwundener Treffer oder eine zweifelhafte Zuordnung unterbrechen die Bestätigung. Andere Ergebnisse heißen „Unreif“, „Überreif“ oder „Nicht sicher beurteilbar“. Der Überblick nennt die empfohlenen Nummern oder zeigt „Aktuell keine Banane sicher als essreif erkannt“. Bewertet wird ausschließlich die sichtbare Schalenreife.

## Tatsächlicher Modellstand

Die bestehende YOLOX-Nano-Pipeline wurde mit engen Einzelboxen vollständig nachtrainiert. 13 abgeschlossene Pilotepochen und 9 Fortsetzungsepochen, 448 × 448, zwei CPU-Threads. Datensätze der Fortsetzung: **575 Training / 117 Validierung / 101 bisheriger Test / 60 neue Prüffotos**. Überlappung, wechselndes Licht, gemischte Schalenreife und echte Negativfotos sind ergänzt; Trainingsmontagen verwenden ausschließlich ursprüngliche Trainingsfrüchte.

Der neue Bestand wurde am 2026-11-10T18:48:59.307322+00:00 in Commit `59efe9e` vor dem vollständigen Nachtraining festgelegt: **30 reale Büschelfotos von Wikimedia Commons und 30 reale Fotos ohne Bananen aus COCO 2017**. Die 60 Fotos und ihre Annotationen blieben auch bei der Trainingsfortsetzung in Commit `18308d4` unverändert. Ergebnis nach drei Bestätigungen: **6 richtige Empfehlungen, 1 falsche Empfehlungen, 90 ausgelassene reife Einzelbananen; Präzision 85,7 %, Recall 6,3 %**. Ziel: ≥95 % Präzision, ≥80 % Recall. Unsicherheit zählt als ausgelassene Empfehlung.

Die Annotation wurde vom Coding-Assistenten visuell geprüft; eine unabhängige menschliche Gegenprüfung fehlt. COCO-Trainingsfotos waren im generischen Vortraining enthalten. Das frühere fehlgeschlagene Experiment bleibt archiviert; seine bereits eingesehenen Szenen sind ausdrücklich Trainings-/Validierungsdaten und kein neuer unabhängiger Test.

- [Modellbericht](documentation/model-validation.md)
- [Neue Empfehlungsmetriken](public/evaluation/individual-report.json)
- [Eingefrorene Szenen, Boxen, Hashes und Quellen](ml/individual-benchmark-v2.json)
- [Tatsächliche Browserprüfungen und offene Geräteabnahme](public/evaluation/acceptance.md)
- [Trainingspipeline und Reproduktion](ml/README.md)

## Weiterer Trainingsversuch

Der weitere, getrennt dokumentierte Trainingsversuch 0.4 prüfte einen neuen, in Commit `b1de06f` vor Training festgelegten Bestand mit 30 echten Büschelfotos (dokumentierte Ausschnitte) und 30 neuen Negativfotos. Ergebnis: **2 TP, 0 FP, 109 FN; Präzision 100,0 %, Recall 1,8 %**. Er hat die Zielwerte ebenfalls nicht erreicht und ersetzt die Gewichte 0.3 im Änderungsstand nicht. Die öffentliche Pages-Version bleibt 0.1.0. [Bericht mit Grenzen](public/evaluation/experiment-v4/report.md), [Ergebnisse](public/evaluation/individual-v4-report.json), [Validierungswahl](public/evaluation/experiment-v4/validation-selection.json), [Trainingsverlauf](public/evaluation/experiment-v4/training-history.json), [Beispiele aus gespeicherten Ausgaben](public/evaluation/experiment-v4/examples.md).

## Lokal prüfen

Node.js 24, Python 3 für den direkten Vergleich zwischen Python- und Browser-Nachverarbeitung:

```bash
npm ci
npm run dev
npm run check
npm test
npx playwright install --with-deps chromium webkit
npm run test:browser
npm run check:runtime
```

`npm run test:browser` erstellt zuerst den statischen GitHub-Pages-Build und prüft ihn unter `/Bananenblick/` in Chromium und WebKit. Die Kamerafälle verwenden kontrollierte Video-/Worker-Doubles; der Beispieltest führt das echte ONNX-Modell aus. Die Artefakte enthalten den vom Worker geprüften Modellhash und die Browserversionen. Erfolgreiche Emulation wird nicht als Android-/iPhone-Geräteabnahme bezeichnet.

`npm run build` erstellt separat den statischen Export unter `docs/` und prüft relative Pfade, Assets, Modellhashes und die gemeinsame Python-/TypeScript-Nachverarbeitung. `npm run check:runtime` führt das echte ONNX-Modell mit WASM unter Node/Linux aus; das ist kein Smartphone-Benchmark.

## Veröffentlichung

Ziel bleibt ausschließlich **GitHub Pages** über `.github/workflows/pages.yml`. Pull Requests prüfen TypeScript, Tests, Chromium/WebKit und den statischen Build. Nur `main` kann veröffentlichen. Vorher kontrolliert `scripts/check-release.mjs` die Modellziele (≥95 % Präzision, ≥80 % Recall), mindestens 30 reale Büschelszenen und tatsächliche Android-/iPhone-Nachweise einschließlich mindestens drei vollständiger Auswertungen/s. Die offenen Nachweise werden nicht als bestanden ausgegeben. Ein lokaler Build bleibt für die Prüfung des Änderungsstands möglich.

Nach testgetriebenem Nachtraining ist ein neuer, vorab festgelegter unberührter Prüfbestand erforderlich. Bereits eingesehene Testfotos dürfen nicht für weiteres Tuning als unabhängiger Nachweis ausgegeben werden. Geräteprüfschritte stehen im [Prüfprotokoll](public/evaluation/acceptance.md).

Die Auswertung bleibt lokal im Browser. Kamerabilder werden weder hochgeladen noch gespeichert. Es gibt keine neue öffentliche API. Die private Sites-Veröffentlichung wurde nicht geändert oder ausgeliefert.

Der [AppShelf-Eintrag](https://der-programmer1.github.io/AppShelf/apps/bananenblick/) verlinkt die bestehende öffentliche Version 0.1.0 als Prototyp. Das noch nicht abgenommene Büschel-Update ist damit nicht als veröffentlicht gekennzeichnet.

Quellen: Rohit et al. (2026), [Mendeley Banana Ripeness Stage Dataset](https://data.mendeley.com/datasets/56c75hvm7r/1), CC BY 4.0; [COCO 2017](https://cocodataset.org/) und Wikimedia Commons mit Quellen und individuellen Lizenzen im Manifest. Tellerbeispiel: [goodiesfirst, „putahracsa fruit plate“](https://www.flickr.com/photos/scaredykat/4430551771/), CC BY 2.0, verkleinert. Gemischter Büschel: [Esther Musasike, „Bunch of banana.jpg“](https://commons.wikimedia.org/wiki/File:Bunch_of_banana.jpg), CC BY-SA 4.0, Ausschnitt und verkleinert. YOLOX: Megvii, Apache 2.0. Ursprüngliche Herkunft: offizieller Sites-/Vinext-Starter; das Standard-Buildziel ist der statische Pages-Export.
