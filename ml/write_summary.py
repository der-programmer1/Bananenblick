#!/usr/bin/env python3
"""Generate the current model report without promoting failed acceptance to success."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent; SITE=ROOT.parent
load=lambda p:json.loads((SITE/p).read_text())
card=load('public/models/model-card.json'); report=load('public/evaluation/individual-report.json')
legacy=load('public/evaluation/test-report.json')['test']; training=card['training']; thresholds=card['thresholds']
pct=lambda x:f'{x*100:.1f} %'.replace('.',',')
lines=[
'# Bananenblick 0.2: Modellprüfung', '',
'**Änderungsentwurf, nicht zur Veröffentlichung abgenommen. Die Zielwerte für essreife Einzelbananen werden verfehlt.**', '',
'## Neuer Prüfbestand und Ergebnis', '',
f"Der Prüfbestand wurde am {report['frozenAt']} in Commit `28b35eb` vor dem Nachtraining festgelegt. Er enthält 30 reale Mehrfruchtszenen (Büschel sowie arrangierte Früchte) und 30 reale Fotos ohne sichtbare Bananen. Die enge Forderung nach 30 unabhängig abgenommenen Büschelszenen ist damit noch nicht vollständig belegt. Keine Montage zählt als reale Szene.", '',
'| Empfehlung nach drei Bestätigungen | Ergebnis | Ziel |', '|---|---:|---:|',
f"| Präzision | {pct(report['precision'])} | ≥ 95 % |",
f"| Recall sichtbarer reifer Einzelbananen | {pct(report['recall'])} | ≥ 80 % |",
f"| Richtige / falsche Empfehlungen | {report['truePositive']} / {report['falsePositive']} | |",
f"| Ausgelassene reife Einzelbananen | {report['falseNegative']} | |",
f"| Falsche Empfehlungen auf Negativfotos | {report['negativeFalseRecommendations']} | |", '',
'IoU ≥ 0,50 definiert eine passende Einzelbox. Unsichere, nicht bestätigte oder nicht lokalisierte reife Früchte zählen als ausgelassene Empfehlungen. Die exakte TypeScript-Objektverfolgung erhält je Foto drei identische Inferenzergebnisse in 200-ms-Abständen. Dies prüft die Empfehlung auf statischen Fotos; es belegt keine zeitliche Stabilität einer echten Kamera.', '',
'Der Detektor liefert in diesem Bestand überwiegend keinen Treffer oder einen zu großen Gesamtrahmen für mehrere Früchte. Zehn der 30 positiven Fotos haben überhaupt einen Treffer oberhalb der Detektionsschwelle. Die hohe Quote auf bisherigen Einzelaufnahmen lässt sich nicht auf Büschel übertragen. Ein weiteres Nachtraining oder Tuning anhand dieses nun eingesehenen Prüfbestands benötigt einen neuen unberührten Holdout.', '',
'## Training und Nachverarbeitung', '',
f"YOLOX-Nano, 320 × 320 BGR/Float32, acht zusätzliche CPU-Epochen mit Seed {training['seed']}. Höchstes Validierungs-mAP50 wählt Epoche {training['selectedEpoch']}. Der gewählte Checkpoint ist somit ein früher Trainingsstand; es wird kein Erfolg des vollständigen Nachtrainings behauptet. 503 Trainingsbilder, 110 Validierungsbilder, 101 bisherige Testbilder und 60 neue Prüffotos.", '',
'471 ursprüngliche Trainingsbilder wurden um 12 reale positive COCO-Fotos und 20 reale Negativfotos ergänzt. Die Validierung erhielt vier positive und fünf negative COCO-Fotos. Die positiven Ergänzungen enthalten Überlappungen, unterschiedliche Hintergründe und Beleuchtungen sowie gemischte Schalenreife. Visuell geprüfte Einzelboxen ersetzen die ausgewählten groben COCO-Büschelboxen. Eine doppelte identische Mendeley-Annotation wurde entfernt. Die neue Menge genügt noch nicht für belastbare Büschelerkennung.', '',
'Die Einzelboxen und Reifestufen wurden vom Coding-Assistenten visuell geprüft. Eine unabhängige menschliche Gegenprüfung fehlt. Die Klassen für ungewöhnlich rote Schalen sind im Holdout als visuell nicht bestimmbar markiert. Die COCO-Trainingsfotos waren außerdem bereits im ursprünglichen COCO-Vortraining des Detektors enthalten. Dies ist ein Holdout vom Bananen-Nachtraining, kein vollständig unabhängiger Test vom generischen Vortraining.', '',
f"Ausschließlich Validierung wählte Detektion {thresholds['detection']}, sichere Klassifikation {thresholds['classification']}, Klassenmarge {thresholds['margin']} und klassenübergreifende NMS-IoU {thresholds['nmsIou']}. Geprüfte NMS-Werte: 0,40–0,75. Trotz höherer Kandidaten bevorzugte die Validierung 0,40; deren kleine Ergänzung deckt dichte Überlappungen nicht ausreichend ab. Es wurde nach Sichtung des neuen Tests kein Schwellwert angepasst.", '',
'Python und Browser lesen dieselbe versionierte Konfiguration. Ein direkter automatisierter Vergleich prüft beide Decoder auf identischen Daten einschließlich Clipping, Gleichständen, Überlappung und Unsicherheit. Die globale eindeutige Zuordnung nutzt IoU und Bewegung; zweifelhafte Zuordnungen übertragen keine bestätigte Reife. Fehlende oder unsichere Treffer verlieren die Empfehlung sofort. Ein Neustart setzt die Nummern auf 1 zurück.', '',
'## Bisheriger Einzelbildtest mit neuem Modell', '',
f"Die 101 bisherigen Testbilder wurden separat mit dem neuen Artefakt ausgewertet: rohe Präzision {pct(legacy['precision'])}, roher Recall {pct(legacy['recall'])}, sicher richtig beschriftet {pct(legacy['visibleRecall'])}. Diese Werte sind keine Aussage über bestätigte Einzelbananen in Büscheln.", '',
'## Geschwindigkeit und Geräte', '',
f"ONNX Runtime CPU unter Linux, drei Threads: Median {report['inferenceMs']['median']:.1f} ms, p95 {report['inferenceMs']['p95']:.1f} ms für reine Modellaufrufe. Kamera, Vorverarbeitung, Tracking und Darstellung sind nicht enthalten. Der separate WASM-Bericht misst Node/Linux. Beide Zahlen ersetzen keine Messung von mindestens drei vollständigen Kameraauswertungen/s auf Android/Chrome oder iPhone/Safari.", '',
'Chromium- und WebKit-Prüfungen verwenden mobile Ansichtsgrößen in Linux. Der Kameratest verwendet kontrollierte Video-/Worker-Doubles; der Beispieltest führt das echte lokale ONNX-Modell aus. Tatsächliche Gerätetests: keine. Einzelheiten und reproduzierbare Schritte stehen in `acceptance.md` und `device-validation.json`.', '',
'## Artefakte', '',
'- [Neue Empfehlungen und Ergebnisse pro Szene](./individual-report.json)',
'- [Eingefrorene Mitgliedschaft, Einzelboxen, Bildhashes, Quellen und Lizenzen](./individual-benchmark.json)',
'- [Unveränderte Vorhersagen des neuen Tests](./individual-predictions.json)',
'- [Validierungswahl einschließlich aller NMS-Kandidaten](./individual-validation-selection.json)',
'- [Acht zusätzliche Trainingsepochen](./individual-training-history.json)',
'- [Separater bisheriger Einzelbildtest](./test-report.json)',
'- [WASM-Referenzlauf](./web-runtime.json)',
'- [Geräte- und Browserprüfprotokoll](./acceptance.md)', '',
f"Modell-SHA-256: `{card['sha256']}`. Benchmark-SHA-256: `{report['benchmarkSha256']}`.", '',
'## Quellen', '',
'Mendeley: Rohit et al. (2026), [Annotated Banana Ripeness Stage Dataset](https://data.mendeley.com/datasets/56c75hvm7r/1), CC BY 4.0. COCO: [Common Objects in Context 2017](https://cocodataset.org/), Bildautoren und individuelle CC-BY-/CC-BY-SA-2.0-Lizenzen mit Flickr-Verweisen im eingefrorenen Manifest. Quellfotos sind lokal heruntergeladen; nur ausdrücklich gekennzeichnete Beispiele werden mit Lizenzhinweisen mitgeliefert. YOLOX: Megvii, Apache 2.0.', '',
'„Jetzt essreif“ beschreibt ausschließlich die sichtbare Schalenreife; Verderb und innere Qualität werden nicht beurteilt.', ''
]
text='\n'.join(lines)
(SITE/'public/evaluation/model-report.md').write_text(text)
# Repository copy uses paths appropriate to its own location.
repo=text.replace('](./', '](../public/evaluation/')
(SITE/'documentation/model-validation.md').write_text(repo)
print('Current model report generated; acceptance remains failed.')
