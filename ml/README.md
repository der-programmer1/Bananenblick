# Lokales YOLOX-Nano · Einzelbananen-Entwurf

Die Architektur und die lokale ONNX-Auswertung bleiben erhalten. Dieser Stand enthält ein tatsächlich ausgeführtes Nachtraining und ein fehlgeschlagenes neues Prüfergebnis. Er ist kein freigegebenes Modell für zuverlässige Büschelerkennung.

## Tatsächlich ausgeführter Lauf

Linux, Python 3.12, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, drei CPU-Threads, Seed 20261110. Die gepinnten weiteren Pakete stehen in `requirements.txt`. Kein CUDA und keine Bild-API. Acht zusätzliche Epochen dauerten insgesamt etwa 333 Sekunden. Höchstes Validierungs-mAP50 wählte Epoche 1. Der ausgewählte Stand hat lediglich die Head-Aufwärmphase durchlaufen; das ist ein wesentlicher Grund, dieses Experiment nicht als erfolgreiches Einzelbananen-Nachtraining auszugeben.

Ursprüngliche Gewichte für diesen Lauf: `ml/artifacts/banana-yolox-nano.pt` aus Commit `6b682c4`. Der aktuell enthaltene Checkpoint ist der neue ausgewählte Stand. Modellhash, Initialcheckpoint-Hash und tatsächliche Laufdaten stehen in `public/models/model-card.json` und `public/evaluation/individual-training-history.json`.

```bash
python3 -m venv ml/.venv
ml/.venv/bin/pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
ml/.venv/bin/pip install -r ml/requirements.txt
ml/.venv/bin/python ml/prepare_data.py
ml/.venv/bin/python ml/build_individual_data.py
ml/.venv/bin/python ml/verify.py
mkdir -p ml/work
# Den ursprünglichen Ausgangscheckpoint für eine Reproduktion wiederherstellen:
git show 6b682c4:ml/artifacts/banana-yolox-nano.pt > ml/work/base-v0.1.pt
ml/.venv/bin/python ml/train.py --data ml/work/individual --output ml/work/individual-training --pretrained ml/work/base-v0.1.pt --epochs 8 --batch-size 8 --threads 3 --seed 20261110 --nms-iou .45 --freeze-manifest ml/individual-benchmark.json
```

Die Vorbereitung benötigt mehrere GB freien Platz. `prepare_data.py` lädt das Mendeley-Archiv mit festem SHA-256 und erzeugt den ursprünglichen gruppierten Split. `build_individual_data.py` lädt bei Bedarf die offiziellen COCO-2017-Annotationen und die im eingefrorenen Manifest genannten Bilder. Es überprüft Bildhashes und Mitgliedschaft und überschreibt keine geänderte eingefrorene Annotation. Originalfotos, virtuelle Umgebung und Arbeitsdateien unter `work/` werden nicht committed.

## Daten und Audit

- Ursprünglich: 471 Training / 101 Validierung / 101 Test, Mendeley, CC BY 4.0; EXIF-Normalisierung und pHash-/Namensgruppen wie im ursprünglichen `split-audit.json`.
- Ergänzt: zwölf positive und zwanzig negative reale COCO-Trainingsfotos; vier positive und fünf negative Validierungsfotos. Überlappungen, wechselnde Hintergründe, Licht und gemischte Schalenreife sind vertreten.
- Neuer Holdout: dreißig reale Mehrfruchtszenen und dreißig negative Fotos. Einige positive Szenen zeigen arrangierte Früchte; die strenge Forderung nach dreißig unabhängig abgenommenen Büschelszenen ist noch offen.
- `individual_annotations.json` enthält geprüfte sichtbare Einzelboxen in Prozent statt der ausgewählten groben COCO-Sammelboxen. Klassen 0/1/2: unreif/reif/überreif. Klasse 3 im Test: ungewöhnliche Schale visuell nicht bestimmbar; keine sichere Empfehlung.
- Eine exakt doppelte Quellannotation in Mendeley-Bild 298 wurde entfernt. Der ursprüngliche Split-Audit beschreibt weiterhin die ursprünglichen Daten, nicht die vollständige Ergänzung.
- Der zusätzliche Split wird nach COCO-Bild-ID getrennt. Bildhashes, Flickr-Verweise, individuelle CC-BY-/CC-BY-SA-2.0-Lizenzen und die vollständige Mitgliedschaft stehen in `individual-benchmark.json`, festgeschrieben in Commit `28b35eb` vor Training.

Die Annotation wurde vom Coding-Assistenten visuell geprüft; unabhängige menschliche Gegenprüfung fehlt. Physische Frucht- und Aufnahmeserien-IDs fehlen. COCO-Trainingsfotos waren außerdem bereits Teil des generischen COCO-Vortrainings. Es ist kein vollständig unabhängiger Frucht-/Standort-/Vortrainings-Holdout.

## Export und einmaliger Test

`release_individual.py` exportiert den Validierungscheckpoint, prüft ONNX/PyTorch numerisch und wählt ausschließlich anhand der Validierung NMS und Schwellwerte. Für diesen Lauf wurden 72 Kombinationen geprüft (NMS 0,40–0,75; Detektion 0,25/0,35/0,50; sichere Klassifikation 0,60/0,70/0,80; Marge 0,15). Gewählt: Detektion 0,50; Klassifikation 0,60; Marge 0,15; NMS 0,40. Das kleine Validierungsplus ist für dichte Büschel nicht ausreichend repräsentativ.

Danach lief der neue Holdout einmal mit dem fertigen ONNX-Artefakt. `scripts/evaluate-recommendations.ts` nutzt den tatsächlichen Browser-Tracker auf drei identischen Resultaten pro Foto. IoU ≥0,50; unsichere oder nicht dreimal bestätigte Treffer zählen als ausgelassene Empfehlung. Ergebnis: **0 TP / 1 FP / 169 FN, Präzision 0 %, Recall 0 %**. Das Modell erkennt überwiegend den gesamten Büschel oder nichts. Die Ziele sind verfehlt.

```bash
# Dieser einmalige Export wurde bereits ausgeführt:
ml/.venv/bin/python ml/release_individual.py
# Das Skript verweigert ein Überschreiben des bereits eingesehenen neuen Tests.
# Vorhandene Vorhersagen ohne erneutes Training nachvollziehen:
node --import tsx scripts/evaluate-recommendations.ts public/evaluation/individual-predictions.json ml/work/reproduced-recommendations.json
ml/.venv/bin/python ml/write_summary.py
```

Nach weiterem modell- oder testgetriebenem Tuning ist ein neuer, vorab festgelegter unberührter Holdout nötig. Das vorhandene Fehlresultat nicht löschen oder als unabhängiges neues Testergebnis überschreiben. Historische Berichte des Modells 0.1 tragen den Präfix `legacy-v0.1-`.

## Vertrag zwischen Python und Browser

Eingang `images`: Float32 `[1,3,320,320]`, BGR, 0…255, links oben eingeordnet, Padding rechts/unten 114. Ausgang `predictions`: `[1,2100,8]` mit bereits dekodiertem `cx,cy,w,h,objectness,unreif,reif,überreif`. Kein zweiter Grid-Decode.

`lib/postprocess-config.json` wird von TypeScript und dem Python-Modul `postprocess.py` gelesen. Gemeinsame Regeln: Score als Objektwert × größter Klassenwert, Clipping, Mindestboxgröße, klassenübergreifende NMS, stabile Reihenfolge bei Gleichstand, höchstens 40 Treffer, Klassenmarge und Unsicherheit. `npm test` vergleicht echte Python- und TypeScript-Ausgaben direkt. Ein abgesenkter Anzeige-Schwellwert umgeht die Mindestkonfidenz der Reife-Bestätigung nicht.

Die eigentliche Empfehlung ist ein Trackingzustand: drei sichere, eindeutig zugeordnete Reifebewertungen; bei Unsicherheit, fehlender Frucht, Unterbrechung oder zu alten Liveframes wird sie gelöscht. Ein statisches Beispiel benötigt ebenfalls drei tatsächliche Modellaufrufe, bleibt aber unabhängig von der Rechenzeit dasselbe Foto.

Browser- und reale Gerätetests sind in `public/evaluation/acceptance.md` dokumentiert. Linux-/Node-Zahlen sind keine Smartphone-Benchmarks. Die private Sites-Veröffentlichung gehört nicht zu diesem Lauf. Weitere Herkunft: `provenance.json`; YOLOX-Lizenz: `vendor/YOLOX-LICENSE`.
