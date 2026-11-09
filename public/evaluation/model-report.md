# Bananenblick: Modell und nachgewiesener Prüfstand

**Prototyp. Die Ausgaben stammen aus einem tatsächlich feintrainierten YOLOX-Nano-Modell. Eine Abnahme auf echten Android- und iPhone-Kameras steht aus.**

## Modell und Daten

CPU-Training über 16 Epochen; auf Validierung gewählter Checkpoint: Epoche 15. Eingang 320 × 320, Float32/BGR; fertiger ONNX-Export 3.52 MiB. Testvorhersagen verwenden genau diesen Export, SHA-256 `75ced29a13b890afaf328751289567fe48ea8471865c637a91244506d6ec6419`.

Daten: Rohit et al. (2026), [Annotated Banana Ripeness Stage Dataset, DOI 10.17632/56c75hvm7r.1](https://data.mendeley.com/datasets/56c75hvm7r/1), CC BY 4.0. Die Beispielbilder sind verkleinert und EXIF-normalisiert.

| Split | Bilder | Ähnlichkeitsgruppen | Unreif | Reif | Überreif |
|---|---:|---:|---:|---:|---:|
| Training | 471 | 426 | 154 | 143 | 175 |
| Validierung | 101 | 37 | 33 | 31 | 37 |
| Test | 101 | 40 | 33 | 31 | 37 |

Audit: 0 exakte Bildduplikate entfernt; 192 ähnliche Bildpaare in unteilbare Gruppen gefasst. 0 erkannte gruppenübergreifende Duplikate. pHash-Prüfung berücksichtigt Drehungen und Spiegelungen. 36 Randboxen wurden beschnitten. EXIF-Drehungen wurden vor Skalierung und Training angewendet und mit Zielannotationen visuell geprüft.

Physische Frucht- und Aufnahmeserien-IDs fehlen. Derselbe Bananentyp oder dieselbe physische Frucht kann trotz automatischer Ähnlichkeitsprüfung in verschiedenen Gruppen vorkommen. Die Zahlen belegen daher keine unabhängige Standort-/Frucht-Generalisation. Die manuellen Annotationen sind teils ungenau.

## Einmalige Holdout-Auswertung

Der Detektionsschwellwert 0.50 wurde ausschließlich auf Validierung gewählt. IoU ≥ 0,50 definiert einen Positionstreffer. mAP50 im Test: **93,9 %**. Gemeinsame Erkennung mit richtiger oberster Reifeklasse: **95,0 % Recall**, **95,0 % Precision**. Als sichere Reifestufe in der Oberfläche beschriftet: **92,1 %**; **3** Treffer erhalten „unsicher“.

| Reifestufe | Testobjekte | Präzision roh | Recall roh | Sicher beschriftet | Falsche Reife | Übersehen | Unsicher |
|---|---:|---:|---:|---:|---:|---:|---:|
| unreif | 33 | 90,9 % | 90,9 % | 81,8 % | 0 | 3 | 3 |
| reif | 31 | 100,0 % | 100,0 % | 100,0 % | 0 | 0 | 0 |
| überreif | 37 | 94,6 % | 94,6 % | 94,6 % | 0 | 2 | 0 |

„Roh“ zählt die höchste Modellklasse auch bei einer Enthaltung der Oberfläche. „Sicher beschriftet“ zählt nur richtige Position und Reifestufe oberhalb der Konfidenz-/Margengrenze. „Falsche Reife“ zählt falsch klassifizierte, aber passend lokalisierte Testobjekte; „übersehen“ hat keinen passenden Rahmen. Falschalarme werden in der Präzision berücksichtigt. Vollständige Matrizen inklusive „unsicher“ und „nicht erkannt“ stehen im JSON-Bericht.

## Geschwindigkeit

**Cloud-CPU-Referenz, keine Smartphone-Messung:** ONNX Runtime CPU mit drei Threads, Median 12.0 ms, p95 49.6 ms über 101 Testbilder. Das sind rechnerisch 83.3 reine Modellaufrufe/s. Kamera, Pixelumwandlung, NMS und Darstellung sind darin nicht enthalten. Das Ziel von mindestens drei vollständigen Kameraauswertungen/s ist auf echten Handys noch nicht nachgewiesen.

## Begrenzte Stresskontrollen

10 vorab festgelegte Kontrollen, davon 4 ohne Zielbanane. Falschalarme in diesen wenigen Negativkontrollen: 0. Die positiven Varianten sind synthetische Montagen, Helligkeitsänderungen und eine Dateidrehung aus Holdout-Fotos. Ein externes reales NASA-Foto ohne Banane ist Public Domain laut offizieller scikit-image-Dokumentation. Kein Nachtraining oder Tuning auf diese Kontrollen.

| Kontrolle | Zielobjekte | Rahmen | Richtige rohe Treffer | Sicher richtige Treffer |
|---|---:|---:|---:|---:|
| Drei Reifestufen · synthetische Montage | 3 | 3 | 3 | 3 |
| Teilweise Überdeckung · synthetische Montage | 2 | 2 | 2 | 2 |
| Unreif · dunkel | 1 | 1 | 1 | 1 |
| Reif · hell | 1 | 1 | 0 | 0 |
| Überreif · dunkel | 1 | 1 | 1 | 1 |
| Um 90° gedrehte Bilddatei | 1 | 1 | 1 | 1 |
| Leeres Kontrollbild · schwarz | 0 | 0 | 0 | 0 |
| Leeres Kontrollbild · neutralgrau | 0 | 0 | 0 | 0 |
| Leeres Kontrollbild · weiß | 0 | 0 | 0 | 0 |
| Reales Foto ohne Banane | 0 | 0 | 0 | 0 |

Die aufgehellte reife Probe verfehlt das Positionskriterium IoU ≥ 0,50 trotz korrekter oberster Reifeklasse. Die Kontrollen sind somit nicht durchgehend erfolgreich; robuste Erkennung bei starken Lichtänderungen ist nicht belegt.

Diese kleinen Kontrollen sind kein unabhängiger Praxistest. Echte gemischte Bananentheken, dicht liegende Büschel, tatsächliche Überlappungen, Wechsellicht, Bewegung, Handyrotation und repräsentative Negativszenen bleiben offen.

## Prüfartefakte

- [Vollständiger Modell-Testbericht](./test-report.json)
- [Split-Audit mit Mitgliedschaft, Hashes und Gruppen](./split-audit.json)
- [Einzelvorhersagen im Test](./test-predictions.json)
- [Validierung und Schwellwertwahl](./validation-selection.json)
- [Trainingsverlauf](./validation-history.json)
- [Stresskontrollen mit Bildern](./stress-report.json)
- [Zielannotation nach EXIF-Normalisierung](./annotation-orientation-qa.jpg)
- [Modellkarte](../models/model-card.json)
- [Trainiertes ONNX-Modell](../models/banana-yolox-nano.onnx)

Die CPU-Trainingsskripte, Quellverweise, verwendete YOLOX-Lizenz, Reproduktionsanleitung und Prüftests liegen im Quellprojekt unter `ml/`. Der finale CPU-Checkpoint liegt unter `ml/artifacts/banana-yolox-nano.pt`. Kamerabilder sind kein Teil dieser Trainings-/Testartefakte.
