# Bananenblick 0.3.0: Modellprüfung

**Die Zielwerte für essreife Einzelbananen sind nicht erreicht. Echte Android-/iPhone-Prüfungen und die Veröffentlichungsabnahme stehen aus.**

## Unberührter Prüfbestand und Ergebnis

Der neue Bestand wurde am 2026-11-10T18:48:59.307322+00:00 in Commit `59efe9e` vor dem vollständigen Nachtraining festgelegt: **30 reale Büschelfotos von Wikimedia Commons und 30 reale Fotos ohne Bananen aus COCO 2017**. Die 60 Fotos und ihre Annotationen blieben auch bei der Trainingsfortsetzung in Commit `18308d4` unverändert. Keine Montage zählt als Prüfszene. Modell und Schwellwerte wurden ausschließlich mit Trainings- und Validierungsdaten gewählt, danach wurde der neue Bestand einmal ausgewertet.

| Empfehlung nach drei Bestätigungen | Ergebnis | Ziel |
|---|---:|---:|
| Präzision | 85,7 % | ≥ 95 % |
| Recall sichtbarer reifer Einzelbananen | 6,3 % | ≥ 80 % |
| Richtige / falsche Empfehlungen | 6 / 1 | |
| Ausgelassene reife Einzelbananen | 90 | |
| Falsche Empfehlungen auf Negativfotos | 0 | |

IoU ≥ 0,50 definiert eine passende Einzelbox. Unsichere, nicht dreimal bestätigte oder nicht lokalisierte reife Früchte zählen als ausgelassene Empfehlungen. Die exakte TypeScript-Objektverfolgung erhält je Foto drei identische Inferenzergebnisse in 200-ms-Abständen. Dies prüft die Empfehlung auf statischen Fotos; es belegt keine zeitliche Stabilität einer echten Kamera. Zweifelhafte Zuordnungen dürfen auch bei statischen überlappenden Boxen keine Bestätigungen ansammeln.

Die sichtbaren Einzelboxen und Reifestufen wurden vom Coding-Assistenten visuell annotiert und geprüft. Eine unabhängige menschliche Gegenprüfung fehlt. Ungewöhnliche oder nicht sicher erkennbare Schalen sind als nicht beurteilbar markiert. Verdeckte Früchte zählen nach sichtbarem Umfang, vollständig verdeckte Früchte nicht als sichtbare Instanzen. Quellen, Einzelboxen, Ausschnitte, Dateihashes und Lizenzen sind im eingefrorenen Manifest dokumentiert.

## Training und Auswahl

YOLOX-Nano, 448 × 448 BGR/Float32, vollständiges Backbone ab der ersten Epoche. Der Pilot wurde nach 13 vollständigen Epochen beendet; Validierungs-F1 wählte Pilotepoche 11. Darauf folgten 9 Fortsetzungsepochen mit Seed 20261113; höchstes bestätigtes Reife-F1 auf der Validierung wählte Epoche 4. CPU/PyTorch, zwei Threads, keine Bild- oder Inferenz-API.

Die endgültigen Splits enthalten **575 Training / 117 Validierung / 101 bisheriger Test / 60 neuer Test**. 471 Mendeley-Trainingsfotos werden durch echte COCO- und Commons-Fotos ergänzt. Aus dem zuvor fehlgeschlagenen und bereits eingesehenen v1-Test wurden 29 positive Fotos ausdrücklich Trainingsdaten und 30 Negativfotos ebenfalls Trainingsdaten; das visuell unbestimmbare positive Foto blieb in der Validierung. Sechs zusätzliche COCO-Ausschnitte erhielten vor der Fortsetzung neue Einzelboxen ausschließlich für die Validierung. Es gibt keine Überlappung der Bildgruppen zwischen Training, Validierung und dem neuen Test.

600 zusätzliche Trainingsmontagen pro Epoche verwenden ausschließlich aus ursprünglichen Trainingsfotos gewonnene Fruchtmasken. Überdeckungen verkleinern die sichtbaren Zielboxen; fast vollständig verdeckte Früchte werden ausgelassen. Echte Trainingshintergründe und zu 30 % einfarbige Hintergründe, Drehungen und Beleuchtungsänderungen ergänzen die realen Fotos. Positive reale Zusatzfotos werden sechsmal je Epoche gezogen. Diese Montagen sind Trainingsaugmentation und kein Nachweis von realer Erkennungsleistung.

Das erste Experiment 0.2 mit einem früh ausgewählten Head-Checkpoint verfehlte die Ziele (0 richtige Empfehlungen, 1 falsche, 169 ausgelassene reife Früchte). Seine Originalberichte bleiben unter `experiment-v1/` erhalten. Dieser bereits gesehene Bestand wird nicht als unabhängiger neuer Test ausgegeben. COCO-Trainingsfotos waren im generischen Vortraining enthalten; die 30 Commons-Prüffotos sind davon getrennt. Physische Frucht- und Aufnahmeserien-IDs sind bei den öffentlichen Quellen nicht verfügbar.

1320 Kombinationen wurden ausschließlich auf der Validierung geprüft: NMS-IoU 0,45/0,55/0,65/0,75/0,85/0,90, Detektion 0,20–0,50, sichere Klassifikation 0,40–0,90 und Klassenmarge 0,10–0,40. Die Auswahl verlangt ≥95 % Präzision und maximiert dann Recall; falls kein Punkt genügt, wählt sie höchstes F1 und dokumentiert das Verfehlen. Gewählt: Detektion **0.5**, sichere Klassifikation **0.8**, Klassenmarge **0.1**, klassenübergreifende NMS-IoU **0.45**. Nach Einsicht des neuen Tests wurde nichts nachjustiert.

Python und Browser lesen dieselbe versionierte Konfiguration. Ein direkter automatisierter Vergleich prüft beide Decoder auf identischen Daten einschließlich Clipping, Gleichständen, Überlappung und Unsicherheit. Die globale eindeutige Zuordnung nutzt IoU und Bewegung; zweifelhafte Zuordnungen übertragen keine bestätigte Reife. Fehlende oder unsichere Treffer verlieren die Empfehlung sofort. Ein Neustart setzt die Nummern auf 1 zurück. Die App prüft den SHA-256 der geladenen Gewichte gegen die Modellbeschreibung.

Kompakte Nummernschilder vermeiden überdeckte Beschriftungen auf schmalen Bildschirmen. Freie Rahmenpositionen und bei Bedarf Zuordnungslinien halten die Zahlen lesbar; der vollständige Status steht je Frucht in der Ergebnisliste. Ein Mobilfall prüft zwanzig dicht überlappende Rahmen. Die Startüberwachung nutzt den tatsächlichen Laufzeitprovider: Eine kalte WebAssembly-Sitzung erhält ein eigenes Zeitlimit und wird nicht als hängende GPU behandelt.

## Bisheriger Einzelbildtest mit neuem Modell

Die 101 bisherigen Testbilder wurden separat ausgewertet: rohe Präzision 91,1 %, roher Recall 91,1 %, sicher richtig beschriftet 77,2 %. Diese Werte sind keine Aussage über bestätigte Einzelbananen in Büscheln. Die alte Einzelaufnahme-Metrik wird nicht mit dem neuen Büscheltest vermischt.

## Geschwindigkeit und Geräte

ONNX Runtime CPU unter Linux, zwei Threads: Median 11.5 ms, p95 33.3 ms für reine Modellaufrufe. Kamera, Vorverarbeitung, Tracking und Darstellung sind nicht enthalten. Der separate WASM-Bericht misst Node/Linux. Beide Zahlen ersetzen keine Messung von mindestens drei vollständigen Kameraauswertungen/s auf Android/Chrome oder iPhone/Safari.

Chromium- und WebKit-Prüfungen verwenden mobile Ansichtsgrößen in Linux. Kameratests verwenden kontrollierte Video-/Worker-Doubles; der Beispieltest führt das echte lokale ONNX-Modell aus. Tatsächliche physische Gerätetests: keine. Einzelheiten, Browserergebnisse und reproduzierbare Geräteprüfschritte stehen in `acceptance.md` und `device-validation.json`. Die Versions- und Geräteabnahme wird vor einer GitHub-Pages-Veröffentlichung geprüft.

## Artefakte

- [Neue Empfehlungen und Ergebnisse pro Szene](./individual-report.json)
- [Eingefrorene Mitgliedschaft, Einzelboxen, Bildhashes, Quellen und Lizenzen](./individual-benchmark.json)
- [Unveränderte Vorhersagen des neuen Tests](./individual-predictions.json)
- [Validierungswahl einschließlich aller NMS-Kandidaten](./individual-validation-selection.json)
- [Tatsächlicher Trainingsverlauf der Fortsetzung](./individual-training-history.json)
- [Trainingsplan und unveränderte Testhashes](./individual-training-plan.json)
- [Bildgruppen- und Duplikatprüfung](./individual-split-audit.json)
- [Separater bisheriger Einzelbildtest](./test-report.json)
- [WASM-Referenzlauf](./web-runtime.json)
- [Geräte- und Browserprüfprotokoll](./acceptance.md)
- [Ergebnis des fehlgeschlagenen ersten Experiments](./experiment-v1/individual-report.json)

Modell-SHA-256: `18b602c250e26962462ab38c788d115fe8e84b8831430d43d570e1c05170edb2`. Benchmark-SHA-256: `3745ee5c1cc3722291ec45cc260c45b209d3db673ee31ff05da843bd7a4dd2eb`. Trainingsplan-SHA-256: `50f337b1183141bcfbc33ed78c23606034c7f3ccba73af513ab62cd579ac359b`.

## Quellen

Mendeley: Rohit et al. (2026), [Annotated Banana Ripeness Stage Dataset](https://data.mendeley.com/datasets/56c75hvm7r/1), CC BY 4.0. COCO: [Common Objects in Context 2017](https://cocodataset.org/), Bildautoren und individuelle CC-BY-/CC-BY-SA-2.0-Lizenzen mit Flickr-Verweisen im eingefrorenen Manifest. Wikimedia Commons: individuelle CC-BY-/CC-BY-SA-/gemeinfreie Quellen mit Autor und Lizenz je Foto im Manifest. Quellfotos sind lokal heruntergeladen; nur ausdrücklich gekennzeichnete Beispiele werden mit Lizenzhinweisen mitgeliefert. YOLOX: Megvii, Apache 2.0.

„Jetzt essreif“ beschreibt ausschließlich die sichtbare Schalenreife; Verderb und innere Qualität werden nicht beurteilt.
