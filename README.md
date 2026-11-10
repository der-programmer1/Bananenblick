# Bananenblick · nummerierte Einzelbananen

**Änderungsentwurf 0.2 – noch nicht veröffentlicht.** Die bisherige öffentliche App steht unter [der-programmer1.github.io/Bananenblick](https://der-programmer1.github.io/Bananenblick/). Dieser Branch ersetzt sie nicht: Der neue Mehrfruchttest verfehlt die Modellziele, und echte Android-/iPhone-Prüfungen fehlen.

Jeder aktuelle Treffer erhält eine eindeutige Nummer. Rahmen und Ergebnisliste verwenden dieselbe Nummer; Antippen wählt die zugehörige Frucht aus. Die globale Zuordnung hält Nummern bei leichter Bewegung stabil. Beim Neustart beginnt die Nummerierung wieder bei 1.

Ausschließlich drei aufeinanderfolgende sichere Bewertungen als „reif“ ergeben einen grünen Rahmen und „✓ Jetzt essreif“. Unsicherheit, ein verschwundener Treffer oder eine zweifelhafte Zuordnung unterbrechen die Bestätigung. Andere Ergebnisse heißen „Unreif“, „Überreif“ oder „Nicht sicher beurteilbar“. Der Überblick nennt die empfohlenen Nummern oder zeigt „Aktuell keine Banane sicher als essreif erkannt“. Bewertet wird ausschließlich die sichtbare Schalenreife.

## Tatsächlicher Modellstand

Die bestehende YOLOX-Nano-Pipeline wurde mit korrigierten Einzelboxen um zwölf reale positive COCO-Fotos und zwanzig Negativfotos ergänzt und acht CPU-Epochen nachtrainiert. Der auf Validierung gewählte Checkpoint stammt aus Epoche 1. Datensätze: **503 Training / 110 Validierung / 101 bisheriger Test / 60 neue Prüffotos**. Einzelannotation und Mitgliedschaft wurden vor Trainingsbeginn in Commit `28b35eb` festgelegt.

**Neuer Test: 0 richtige Empfehlungen, 1 falsche Empfehlung, 169 ausgelassene reife Einzelbananen; Präzision 0 %, Recall 0 %.** Der Bestand enthält 30 reale Mehrfruchtszenen und 30 reale Negativszenen. Er enthält auch arrangierte Früchte und ersetzt noch keine unabhängige Abnahme von 30 Büschelszenen. Die zusätzliche Annotation wurde vom Coding-Assistenten visuell geprüft; eine unabhängige Gegenprüfung fehlt. COCO-Trainingsbilder waren außerdem im generischen Vortraining enthalten.

Das Modell gibt bei Büscheln weiterhin überwiegend keinen oder einen zu großen Gesamtrahmen aus. Die gewünschte zuverlässige Erkennung einzelner Früchte ist damit **nicht erreicht**. Der neue Bericht nennt diese Grenze ausdrücklich; bisherige Einzelbildwerte werden separat ausgewiesen. Das neue Büschelbeispiel zeigt einen fehlgeschlagenen Fall.

- [Modellbericht](documentation/model-validation.md)
- [Neue Empfehlungsmetriken](public/evaluation/individual-report.json)
- [Eingefrorene Szenen, Boxen, Hashes und Quellen](ml/individual-benchmark.json)
- [Tatsächliche Browserprüfungen und offene Geräteabnahme](public/evaluation/acceptance.md)
- [Trainingspipeline und Reproduktion](ml/README.md)

## Lokal prüfen

Node.js 24, Python 3 für den direkten Vergleich zwischen Python- und Browser-Nachverarbeitung:

```bash
npm ci
npm run dev
npm run check
npm test
npx playwright install --with-deps chromium webkit
npm run test:browser
npm run build
```

`npm run build` erstellt den statischen GitHub-Pages-Export unter `docs/` und prüft relative Pfade, Assets, Modellhashes und die gemeinsame Nachverarbeitungskonfiguration. `npm run check:runtime` führt zusätzlich das echte ONNX-Modell mit WASM unter Node/Linux aus. Dies ist kein Smartphone-Benchmark.

Die Browserfälle nutzen Chromium und WebKit mit mobilen Ansichtsgrößen unter Linux. Der Kamerafall verwendet kontrollierte Video-/Worker-Doubles; der Beispieltest verwendet das echte ONNX-Modell. Erfolgreiche Emulation wird nicht als Android-/iPhone-Geräteabnahme bezeichnet.

## Veröffentlichung

Ziel bleibt ausschließlich **GitHub Pages** über `.github/workflows/pages.yml`. Pull Requests prüfen TypeScript, Tests, Chromium/WebKit und den statischen Build. Nur `main` kann veröffentlichen. Vor der Veröffentlichung kontrolliert `scripts/check-release.mjs` die Modellziele (≥95 % Präzision, ≥80 % Recall) und tatsächliche Android-/iPhone-Nachweise einschließlich mindestens drei vollständiger Auswertungen/s. Diese Abnahme schlägt aktuell erwartungsgemäß fehl. Ein lokaler Build bleibt für die Prüfung des Entwurfs möglich.

Nach weiterem Training mit überprüften Einzelbananen-Annotationen ist ein **neuer unberührter Prüfbestand** erforderlich; der jetzige Test wurde eingesehen und darf nicht für weiteres Tuning als unabhängiger Nachweis ausgegeben werden. Anschließend sind reale Geräteprüfungen nach dem [Prüfprotokoll](public/evaluation/acceptance.md) nötig.

Die Auswertung bleibt lokal im Browser. Kamerabilder werden weder hochgeladen noch gespeichert. Es gibt keine neue öffentliche API. Die private Sites-Veröffentlichung wurde nicht geändert oder ausgeliefert.

Quellen: Rohit et al. (2026), [Mendeley Banana Ripeness Stage Dataset](https://data.mendeley.com/datasets/56c75hvm7r/1), CC BY 4.0; [COCO 2017](https://cocodataset.org/) mit den individuellen CC-BY-/CC-BY-SA-2.0-Bildlizenzen und Flickr-Verweisen im Manifest. Büschelbeispiel: [goodiesfirst, „putahracsa fruit plate“](https://www.flickr.com/photos/scaredykat/4430551771/), CC BY 2.0, verkleinert. YOLOX: Megvii, Apache 2.0. Ursprüngliche Herkunft: offizieller Sites-/Vinext-Starter; das Standard-Buildziel ist der statische Pages-Export.
