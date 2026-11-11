# Trainingsversuch 0.4 · tatsächliche Ergebnisse

**Die statischen Zielwerte sind verfehlt. Der Änderungsstand behält Modell 0.3; öffentlich bleibt Version 0.1.0.** Dieser Bericht ist keine abgeschlossene Veröffentlichung oder echte Smartphone-Prüfung.

## Unabhängiger neuer Fotobestand

Vor dem Training in Commit `b1de06fd32e1759957b709bc1a40bbb972928890` festgelegt: 30 reale Commons-Büschelfotos und 30 weitere reale COCO-Negativfotos. Die dokumentierten Ausschnitte bilden nähere Ansichten nach. Der Bestand enthält 111 annotierte reife Einzelbananen. Er wurde nach Checkpoint- und Schwellwertwahl einmal ausgewertet. Keine Montage zählt als Prüfszene.

| Bestätigte essreife Einzelbananen | Ergebnis | Ziel |
|---|---:|---:|
| Präzision | 100,0 % | mindestens 95 % |
| Recall | 1,8 % | mindestens 80 % |
| Richtige Empfehlungen | 2 | |
| Falsche Empfehlungen | 0 | |
| Ausgelassene reife Früchte | 109 | |
| Falsche Empfehlungen auf Negativfotos | 0 | |

Passende Einzelbox: IoU ≥0,5 und eindeutige Zuordnung. Die tatsächliche TypeScript-Objektverfolgung bewertet je Foto drei identische gespeicherte Inferenzergebnisse. Enthaltungen und fehlende Bestätigungen zählen als ausgelassene Empfehlungen. Dies belegt weder zeitliche Kamerastabilität noch Smartphone-Durchsatz.

## Training und Auswahl

592 Trainingsfotos, 177 Validierungsfotos und 101 separat ausgewiesene bisherige Einzelbild-Testfotos. Die bereits eingesehenen alten 60 Prüffotos sind jetzt ausdrücklich Validierungsdaten. 17 weitere echte Trainingsfotos, achtfache Gewichtung realer Zusatzfotos, 150 statt 600 Montage-Slots je Epoche, echte Fotoausschnitte mit auf den sichtbaren Bereich begrenzten Zielboxen. YOLOX-Nano, 448 × 448, zwei CPU-Threads.

20 vollständige Epochen im ersten Lauf, danach vier gesondert vorab festgelegte L1-Epochen ab dem ausgewählten Erstcheckpoint. Der Holdout blieb bei beiden Phasen unverändert und nicht inferiert. Der Erstcheckpoint blieb zusätzlich zu den vier neuen Checkpoints wählbar. Insgesamt 24 Epochen tatsächlich gerechnet; die Validierung wählte Epoche 13.

Beim festen Validierungspunkt (Detektion 0,25; Klassifikation 0,6; Marge 0,15; NMS 0,65) änderte sich F1 von 0.508 auf 0.549. Diese Zahlen betreffen dieselbe erweiterte Validierung und sind keine Ergebnisse des neuen Tests.

Die abschließende Schwellwertwahl untersuchte 1320 Kombinationen ausschließlich auf der Validierung. Gewählt: `{"detection": 0.5, "classification": 0.85, "margin": 0.1, "nmsIou": 0.45}`. Validierungspräzision 100,0 %, Validierungsrecall 14,1 %. Das Kriterium verlangt mindestens 95 % Präzision und maximiert danach Recall; erst danach wurde der neue Test inferiert. Nach Einsicht des Tests erfolgte keine Anpassung.

## Nachweise und Grenzen

- [Eingefrorene Fotos, Einzelboxen, Quellen, Lizenzen und Hashes](benchmark.json)
- [Quellen- und Duplikatprüfung](split-audit.json)
- [Tatsächlicher Trainingsverlauf](training-history.json)
- [Plan der zusätzlichen Lokalisierungsphase](refinement-plan.json)
- [Ergebnis und Checkpointwahl dieser Phase](refinement-summary.json)
- [Vollständige Validierungswahl](validation-selection.json)
- [Ergebnisse pro Prüffoto](individual-report.json)
- [Gespeicherte Vorhersagen](individual-predictions.json)
- [Drei vorab ausgewählte Beispiele mit tatsächlichen Nummern und Statuszeilen](examples.md)
- [Separater bisheriger Einzelbildtest](test-report.json)

Die Einzelannotation stammt vom Coding-Assistenten; eine unabhängige Expertenprüfung fehlt. Einzelne Internetfotos besitzen keine verlässlichen physischen Frucht- oder Aufnahmeserien-IDs. Der automatische Audit findet keine Quellgruppen-, Pixel- oder pHash-Überschneidung mit dem neuen Holdout. Ausschnitte belegen keine Genauigkeit auf den gesamten ursprünglichen Fotos. COCO-Negativfotos waren im generischen COCO-Vortraining enthalten.

Tatsächlich geprüfte physische Android-/Chrome- und iPhone-/Safari-Geräte: **keine**. Linux-Browseremulation ersetzt diese Prüfung nicht. Die reine ONNX-CPU-Inferenz im Fototest brauchte im Median 10.9 ms und p95 30.5 ms; Aufnahme, Vorverarbeitung und Darstellung sind nicht enthalten. Das belegt keine drei vollständigen Smartphone-Auswertungen pro Sekunde.

Modell-SHA-256: `9aa2bb21476b8b7b4e6fd8d79444e8971f15773e34204a8ae51f65c3a420f5d4`. Benchmark-SHA-256: `f0ebd7939a01ccb232b3b8f554e885b45ee6aaf43cb0e408633eb4a1d244d73e`.
