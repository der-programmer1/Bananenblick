#!/usr/bin/env python3
"""Generate the current report from actual measured results and frozen provenance."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent;SITE=ROOT.parent
load=lambda p:json.loads((SITE/p).read_text())
card=load('public/models/model-card.json');report=load('public/evaluation/individual-report.json')
legacy=load('public/evaluation/test-report.json')['test'];training=card['training'];thresholds=card['thresholds']
plan=load('ml/individual-training-v3.json');selection=load('public/evaluation/individual-validation-selection.json')
pct=lambda x:f'{x*100:.1f} %'.replace('.',',')
status='Die Zielwerte im statischen Fototest sind erreicht.' if report['passed'] else 'Die Zielwerte für essreife Einzelbananen sind nicht erreicht.'
lines=[
f"# Bananenblick {card['version']}: Modellprüfung",'',
f'**{status} Echte Android-/iPhone-Prüfungen und die Veröffentlichungsabnahme stehen aus.**','',
'## Unberührter Prüfbestand und Ergebnis','',
f"Der neue Bestand wurde am {report['frozenAt']} in Commit `59efe9e` vor dem vollständigen Nachtraining festgelegt: **30 reale Büschelfotos von Wikimedia Commons und 30 reale Fotos ohne Bananen aus COCO 2017**. Die 60 Fotos und ihre Annotationen blieben auch bei der Trainingsfortsetzung in Commit `18308d4` unverändert. Keine Montage zählt als Prüfszene. Modell und Schwellwerte wurden ausschließlich mit Trainings- und Validierungsdaten gewählt, danach wurde der neue Bestand einmal ausgewertet.",'',
'| Empfehlung nach drei Bestätigungen | Ergebnis | Ziel |','|---|---:|---:|',
f"| Präzision | {pct(report['precision'])} | ≥ 95 % |",
f"| Recall sichtbarer reifer Einzelbananen | {pct(report['recall'])} | ≥ 80 % |",
f"| Richtige / falsche Empfehlungen | {report['truePositive']} / {report['falsePositive']} | |",
f"| Ausgelassene reife Einzelbananen | {report['falseNegative']} | |",
f"| Falsche Empfehlungen auf Negativfotos | {report['negativeFalseRecommendations']} | |",'',
'IoU ≥ 0,50 definiert eine passende Einzelbox. Unsichere, nicht dreimal bestätigte oder nicht lokalisierte reife Früchte zählen als ausgelassene Empfehlungen. Die exakte TypeScript-Objektverfolgung erhält je Foto drei identische Inferenzergebnisse in 200-ms-Abständen. Dies prüft die Empfehlung auf statischen Fotos; es belegt keine zeitliche Stabilität einer echten Kamera. Zweifelhafte Zuordnungen dürfen auch bei statischen überlappenden Boxen keine Bestätigungen ansammeln.','',
'Die sichtbaren Einzelboxen und Reifestufen wurden vom Coding-Assistenten visuell annotiert und geprüft. Eine unabhängige menschliche Gegenprüfung fehlt. Ungewöhnliche oder nicht sicher erkennbare Schalen sind als nicht beurteilbar markiert. Verdeckte Früchte zählen nach sichtbarem Umfang, vollständig verdeckte Früchte nicht als sichtbare Instanzen. Quellen, Einzelboxen, Ausschnitte, Dateihashes und Lizenzen sind im eingefrorenen Manifest dokumentiert.','',
'## Training und Auswahl','',
f"YOLOX-Nano, {card['inputSize']} × {card['inputSize']} BGR/Float32, vollständiges Backbone ab der ersten Epoche. Der Pilot wurde nach {training['pilot']['completedEpochs']} vollständigen Epochen beendet; Validierungs-F1 wählte Pilotepoche {training['pilot']['selectedEpoch']}. Darauf folgten {training['epochs']} Fortsetzungsepochen mit Seed {training['seed']}; höchstes bestätigtes Reife-F1 auf der Validierung wählte Epoche {training['selectedEpoch']}. CPU/PyTorch, zwei Threads, keine Bild- oder Inferenz-API.",'',
f"Die endgültigen Splits enthalten **{training['data']['train']} Training / {training['data']['val']} Validierung / {training['data']['test']} bisheriger Test / {training['data']['acceptance']} neuer Test**. 471 Mendeley-Trainingsfotos werden durch echte COCO- und Commons-Fotos ergänzt. Aus dem zuvor fehlgeschlagenen und bereits eingesehenen v1-Test wurden 29 positive Fotos ausdrücklich Trainingsdaten und 30 Negativfotos ebenfalls Trainingsdaten; das visuell unbestimmbare positive Foto blieb in der Validierung. Sechs zusätzliche COCO-Ausschnitte erhielten vor der Fortsetzung neue Einzelboxen ausschließlich für die Validierung. Es gibt keine Überlappung der Bildgruppen zwischen Training, Validierung und dem neuen Test.",'',
'600 zusätzliche Trainingsmontagen pro Epoche verwenden ausschließlich aus ursprünglichen Trainingsfotos gewonnene Fruchtmasken. Überdeckungen verkleinern die sichtbaren Zielboxen; fast vollständig verdeckte Früchte werden ausgelassen. Echte Trainingshintergründe und zu 30 % einfarbige Hintergründe, Drehungen und Beleuchtungsänderungen ergänzen die realen Fotos. Positive reale Zusatzfotos werden sechsmal je Epoche gezogen. Diese Montagen sind Trainingsaugmentation und kein Nachweis von realer Erkennungsleistung.','',
'Das erste Experiment 0.2 mit einem früh ausgewählten Head-Checkpoint verfehlte die Ziele (0 richtige Empfehlungen, 1 falsche, 169 ausgelassene reife Früchte). Seine Originalberichte bleiben unter `experiment-v1/` erhalten. Dieser bereits gesehene Bestand wird nicht als unabhängiger neuer Test ausgegeben. COCO-Trainingsfotos waren im generischen Vortraining enthalten; die 30 Commons-Prüffotos sind davon getrennt. Physische Frucht- und Aufnahmeserien-IDs sind bei den öffentlichen Quellen nicht verfügbar.','',
f"{len(selection['candidates'])} Kombinationen wurden ausschließlich auf der Validierung geprüft: NMS-IoU 0,45/0,55/0,65/0,75/0,85/0,90, Detektion 0,20–0,50, sichere Klassifikation 0,40–0,90 und Klassenmarge 0,10–0,40. Die Auswahl verlangt ≥95 % Präzision und maximiert dann Recall; falls kein Punkt genügt, wählt sie höchstes F1 und dokumentiert das Verfehlen. Gewählt: Detektion **{thresholds['detection']}**, sichere Klassifikation **{thresholds['classification']}**, Klassenmarge **{thresholds['margin']}**, klassenübergreifende NMS-IoU **{thresholds['nmsIou']}**. Nach Einsicht des neuen Tests wurde nichts nachjustiert.",'',
'Python und Browser lesen dieselbe versionierte Konfiguration. Ein direkter automatisierter Vergleich prüft beide Decoder auf identischen Daten einschließlich Clipping, Gleichständen, Überlappung und Unsicherheit. Die globale eindeutige Zuordnung nutzt IoU und Bewegung; zweifelhafte Zuordnungen übertragen keine bestätigte Reife. Fehlende oder unsichere Treffer verlieren die Empfehlung sofort. Ein Neustart setzt die Nummern auf 1 zurück. Die App prüft den SHA-256 der geladenen Gewichte gegen die Modellbeschreibung.','',
'## Bisheriger Einzelbildtest mit neuem Modell','',
f"Die 101 bisherigen Testbilder wurden separat ausgewertet: rohe Präzision {pct(legacy['precision'])}, roher Recall {pct(legacy['recall'])}, sicher richtig beschriftet {pct(legacy['visibleRecall'])}. Diese Werte sind keine Aussage über bestätigte Einzelbananen in Büscheln. Die alte Einzelaufnahme-Metrik wird nicht mit dem neuen Büscheltest vermischt.",'',
'## Geschwindigkeit und Geräte','',
f"ONNX Runtime CPU unter Linux, zwei Threads: Median {report['inferenceMs']['median']:.1f} ms, p95 {report['inferenceMs']['p95']:.1f} ms für reine Modellaufrufe. Kamera, Vorverarbeitung, Tracking und Darstellung sind nicht enthalten. Der separate WASM-Bericht misst Node/Linux. Beide Zahlen ersetzen keine Messung von mindestens drei vollständigen Kameraauswertungen/s auf Android/Chrome oder iPhone/Safari.",'',
'Chromium- und WebKit-Prüfungen verwenden mobile Ansichtsgrößen in Linux. Kameratests verwenden kontrollierte Video-/Worker-Doubles; der Beispieltest führt das echte lokale ONNX-Modell aus. Tatsächliche physische Gerätetests: keine. Einzelheiten, Browserergebnisse und reproduzierbare Geräteprüfschritte stehen in `acceptance.md` und `device-validation.json`. Die Versions- und Geräteabnahme wird vor einer GitHub-Pages-Veröffentlichung geprüft.','',
'## Artefakte','',
'- [Neue Empfehlungen und Ergebnisse pro Szene](./individual-report.json)',
'- [Eingefrorene Mitgliedschaft, Einzelboxen, Bildhashes, Quellen und Lizenzen](./individual-benchmark.json)',
'- [Unveränderte Vorhersagen des neuen Tests](./individual-predictions.json)',
'- [Validierungswahl einschließlich aller NMS-Kandidaten](./individual-validation-selection.json)',
'- [Tatsächlicher Trainingsverlauf der Fortsetzung](./individual-training-history.json)',
'- [Trainingsplan und unveränderte Testhashes](./individual-training-plan.json)',
'- [Bildgruppen- und Duplikatprüfung](./individual-split-audit.json)',
'- [Separater bisheriger Einzelbildtest](./test-report.json)',
'- [WASM-Referenzlauf](./web-runtime.json)',
'- [Geräte- und Browserprüfprotokoll](./acceptance.md)',
'- [Ergebnis des fehlgeschlagenen ersten Experiments](./experiment-v1/individual-report.json)','',
f"Modell-SHA-256: `{card['sha256']}`. Benchmark-SHA-256: `{report['benchmarkSha256']}`. Trainingsplan-SHA-256: `{report['trainingPlanSha256']}`.",'',
'## Quellen','',
'Mendeley: Rohit et al. (2026), [Annotated Banana Ripeness Stage Dataset](https://data.mendeley.com/datasets/56c75hvm7r/1), CC BY 4.0. COCO: [Common Objects in Context 2017](https://cocodataset.org/), Bildautoren und individuelle CC-BY-/CC-BY-SA-2.0-Lizenzen mit Flickr-Verweisen im eingefrorenen Manifest. Wikimedia Commons: individuelle CC-BY-/CC-BY-SA-/gemeinfreie Quellen mit Autor und Lizenz je Foto im Manifest. Quellfotos sind lokal heruntergeladen; nur ausdrücklich gekennzeichnete Beispiele werden mit Lizenzhinweisen mitgeliefert. YOLOX: Megvii, Apache 2.0.','',
'„Jetzt essreif“ beschreibt ausschließlich die sichtbare Schalenreife; Verderb und innere Qualität werden nicht beurteilt.',''
]
text='\n'.join(lines)
(SITE/'public/evaluation/model-report.md').write_text(text)
(SITE/'documentation/model-validation.md').write_text(text.replace('](./', '](../public/evaluation/'))
print('Current model report generated from actual results.')

readme=f'''# Bananenblick · nummerierte Einzelbananen

**Stand {card['version']} – Veröffentlichungsabnahme noch offen.** Die bestehende öffentliche App steht unter [der-programmer1.github.io/Bananenblick](https://der-programmer1.github.io/Bananenblick/). Der neue Stand liegt im Änderungsbranch. {status} Tatsächliche Android-/iPhone-Prüfungen fehlen.

Jeder aktuelle Treffer erhält eine eindeutige Nummer. Rahmen und Ergebnisliste verwenden dieselbe Nummer; Antippen wählt die zugehörige Frucht aus. Die globale eindeutige Zuordnung hält Nummern bei leichter Bewegung stabil. Beim Neustart beginnt die Nummerierung wieder bei 1.

Ausschließlich drei aufeinanderfolgende sichere Bewertungen als „reif“ ergeben einen grünen Rahmen und „✓ Jetzt essreif“. Unsicherheit, ein verschwundener Treffer oder eine zweifelhafte Zuordnung unterbrechen die Bestätigung. Andere Ergebnisse heißen „Unreif“, „Überreif“ oder „Nicht sicher beurteilbar“. Der Überblick nennt die empfohlenen Nummern oder zeigt „Aktuell keine Banane sicher als essreif erkannt“. Bewertet wird ausschließlich die sichtbare Schalenreife.

## Tatsächlicher Modellstand

Die bestehende YOLOX-Nano-Pipeline wurde mit engen Einzelboxen vollständig nachtrainiert. {training['pilot']['completedEpochs']} abgeschlossene Pilotepochen und {training['epochs']} Fortsetzungsepochen, {card['inputSize']} × {card['inputSize']}, zwei CPU-Threads. Datensätze der Fortsetzung: **{training['data']['train']} Training / {training['data']['val']} Validierung / 101 bisheriger Test / 60 neue Prüffotos**. Überlappung, wechselndes Licht, gemischte Schalenreife und echte Negativfotos sind ergänzt; Trainingsmontagen verwenden ausschließlich ursprüngliche Trainingsfrüchte.

Der neue Test enthält **30 reale Commons-Büschelszenen und 30 reale COCO-Negativszenen**, vor Training in Commit `59efe9e` festgelegt und bei der Fortsetzung in Commit `18308d4` unverändert. Ergebnis nach drei Bestätigungen: **{report['truePositive']} richtige Empfehlungen, {report['falsePositive']} falsche Empfehlungen, {report['falseNegative']} ausgelassene reife Einzelbananen; Präzision {pct(report['precision'])}, Recall {pct(report['recall'])}**. Ziel: ≥95 % Präzision, ≥80 % Recall. Unsicherheit zählt als ausgelassene Empfehlung.

Die Annotation wurde vom Coding-Assistenten visuell geprüft; eine unabhängige menschliche Gegenprüfung fehlt. COCO-Trainingsfotos waren im generischen Vortraining enthalten. Das frühere fehlgeschlagene Experiment bleibt archiviert; seine bereits eingesehenen Szenen sind ausdrücklich Trainings-/Validierungsdaten und kein neuer unabhängiger Test.

- [Modellbericht](documentation/model-validation.md)
- [Neue Empfehlungsmetriken](public/evaluation/individual-report.json)
- [Eingefrorene Szenen, Boxen, Hashes und Quellen](ml/individual-benchmark-v2.json)
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
npm run check:runtime
```

`npm run test:browser` erstellt zuerst den statischen GitHub-Pages-Build und prüft ihn unter `/Bananenblick/` in Chromium und WebKit. Die Kamerafälle verwenden kontrollierte Video-/Worker-Doubles; der Beispieltest führt das echte ONNX-Modell aus. Die Artefakte enthalten den vom Worker geprüften Modellhash und die Browserversionen. Erfolgreiche Emulation wird nicht als Android-/iPhone-Geräteabnahme bezeichnet.

`npm run build` erstellt separat den statischen Export unter `docs/` und prüft relative Pfade, Assets, Modellhashes und die gemeinsame Python-/TypeScript-Nachverarbeitung. `npm run check:runtime` führt das echte ONNX-Modell mit WASM unter Node/Linux aus; das ist kein Smartphone-Benchmark.

## Veröffentlichung

Ziel bleibt ausschließlich **GitHub Pages** über `.github/workflows/pages.yml`. Pull Requests prüfen TypeScript, Tests, Chromium/WebKit und den statischen Build. Nur `main` kann veröffentlichen. Vorher kontrolliert `scripts/check-release.mjs` die Modellziele (≥95 % Präzision, ≥80 % Recall), mindestens 30 reale Büschelszenen und tatsächliche Android-/iPhone-Nachweise einschließlich mindestens drei vollständiger Auswertungen/s. Die offenen Nachweise werden nicht als bestanden ausgegeben. Ein lokaler Build bleibt für die Prüfung des Änderungsstands möglich.

Nach testgetriebenem Nachtraining ist ein neuer, vorab festgelegter unberührter Prüfbestand erforderlich. Bereits eingesehene Testfotos dürfen nicht für weiteres Tuning als unabhängiger Nachweis ausgegeben werden. Geräteprüfschritte stehen im [Prüfprotokoll](public/evaluation/acceptance.md).

Die Auswertung bleibt lokal im Browser. Kamerabilder werden weder hochgeladen noch gespeichert. Es gibt keine neue öffentliche API. Die private Sites-Veröffentlichung wurde nicht geändert oder ausgeliefert.

Quellen: Rohit et al. (2026), [Mendeley Banana Ripeness Stage Dataset](https://data.mendeley.com/datasets/56c75hvm7r/1), CC BY 4.0; [COCO 2017](https://cocodataset.org/) und Wikimedia Commons mit Quellen und individuellen Lizenzen im Manifest. Tellerbeispiel: [goodiesfirst, „putahracsa fruit plate“](https://www.flickr.com/photos/scaredykat/4430551771/), CC BY 2.0, verkleinert. Gemischter Büschel: [Esther Musasike, „Bunch of banana.jpg“](https://commons.wikimedia.org/wiki/File:Bunch_of_banana.jpg), CC BY-SA 4.0, Ausschnitt und verkleinert. YOLOX: Megvii, Apache 2.0. Ursprüngliche Herkunft: offizieller Sites-/Vinext-Starter; das Standard-Buildziel ist der statische Pages-Export.
'''
(SITE/'README.md').write_text(readme)
(SITE/'START-HIER.md').write_text(f'''# Bananenblick {card['version']} prüfen

Die Nummerierung, Rahmen-/Listenauswahl und Reifebestätigung sind umgesetzt. {status} Echte Android-/iPhone-Prüfungen und die Veröffentlichungsabnahme sind noch offen. Dieser Stand wurde nicht auf GitHub Pages veröffentlicht.

Mit Node.js 24 lokal starten:

```bash
npm ci
npm run dev
```

`npm run check`, `npm test` und `npm run test:browser` prüfen den Stand. Der Browserbefehl baut zuerst den statischen Pages-Export. Browserinstallation und Ergebnisse stehen in [README.md](README.md).

Bestehende öffentliche Version: https://der-programmer1.github.io/Bananenblick/. Die private Sites-Veröffentlichung ist nicht Teil dieser Änderung.
''')
