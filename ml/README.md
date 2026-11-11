# Lokales YOLOX-Nano · nummerierte Einzelbananen

Der aktuelle Stand verwendet die bestehende Architektur mit vollständigem Backbone-Nachtraining und engen Einzelboxen. Das tatsächliche Ergebnis, die gewählten Schwellwerte und die offene Geräteabnahme stehen im [Modellbericht](../public/evaluation/model-report.md). Kamerabilder und Inferenz bleiben auf dem Gerät. Es gibt keine Bild- oder Inferenz-API.

## Daten und eingefrorener Test

- Ausgangsdaten: 471 Training / 101 Validierung / 101 Test, Mendeley, CC BY 4.0. EXIF-Normalisierung und ursprüngliche Gruppen-/Duplikatprüfung sind in `public/evaluation/split-audit.json` dokumentiert. Eine identisch doppelte Quellannotation wurde im ergänzten Training entfernt.
- Erstes Experiment: 503 Training / 110 Validierung, zwölf zusätzliche echte positive und zwanzig negative COCO-Fotos. Der alte 60-Foto-Test verfehlte die Ziele. Seine unveränderten Berichte bleiben unter `public/evaluation/experiment-v1/` erhalten.
- Neues Training: echte zusätzliche Commons-Fotos mit 13 eng annotierten Ausschnitten. Die schon eingesehenen alten 30 Negativfotos sind ausdrücklich Trainingsdaten. 29 alte positive Fotos wurden in der Fortsetzung ebenfalls Trainingsdaten; ein visuell nicht beurteilbares Foto blieb in der Validierung. Sechs neue COCO-Ausschnitte bilden zusätzliche Validierung.
- Endgültiger Split: **575 Training / 117 Validierung / 101 bisheriger Einzelbildtest / 60 neue Prüffotos**.
- Der neue Test umfasst **30 reale Commons-Büschelfotos und 30 echte COCO-Negativfotos**. `individual-benchmark-v2.json` und `individual_annotations_v2.json` wurden vor Training in Commit `59efe9e` eingefroren. Bei der Fortsetzung in Commit `18308d4` blieben Fotos, Mitgliedschaft und Annotationen des Tests unverändert. Der eingefrorene Fortsetzungsplan steht in `individual-training-v3.json`.
- Klassen: 0 unreif, 1 reif, 2 überreif; 3 bedeutet visuell nicht sicher bestimmbar und darf keine sichere Reifeempfehlung ergeben. Die zusätzliche Annotation beschreibt einzelne sichtbare Früchte statt grober Sammelboxen.

Die Einzelboxen wurden vom Coding-Assistenten visuell geprüft; eine unabhängige menschliche Gegenprüfung fehlt. Quellen, Bildhashes, Autoren, individuelle Lizenzen und Ausschnitte stehen in den Manifesten. COCO-Trainingsfotos waren im generischen Vortraining enthalten. Die neuen Commons-Testbilder sind davon getrennt; öffentliche Quellen liefern jedoch keine physischen Frucht- oder Aufnahmeserien-IDs. Die automatische zusätzliche Gruppen-, Pixel- und pHash-Prüfung ist in `public/evaluation/individual-split-audit.json` dokumentiert.

## Vorbereitung und Training reproduzieren

Linux, Python 3.12, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, zwei CPU-Threads. Alle weiteren Pakete stehen gepinnt in `requirements.txt`. Die Vorbereitung benötigt mehrere GB freien Speicherplatz und lädt ausschließlich öffentliche Originalquellen.

```bash
python3 -m venv ml/.venv
ml/.venv/bin/pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
ml/.venv/bin/pip install -r ml/requirements.txt
npm ci
ml/.venv/bin/python ml/prepare_data.py
ml/.venv/bin/python ml/build_individual_data.py
ml/.venv/bin/python ml/build_data_v2.py
ml/.venv/bin/python ml/build_data_v3.py
ml/.venv/bin/python ml/prepare_sprites.py
ml/.venv/bin/python ml/verify.py
ml/.venv/bin/python ml/audit_individual_data.py
ml/.venv/bin/python ml/train_individual_v3.py
```

`build_individual_data.py` lädt die ursprünglichen COCO-Annotationen und benötigten Bilder. Die weiteren Builder rekonstruieren die eingefrorenen Quellen und prüfen Hashes; Änderungen an eingefrorenen Annotationen werden nicht still überschrieben. Arbeitsbilder, virtuelle Umgebung und Trainingsdateien unter `ml/work/` sind nicht committed.

Die Fortsetzung beginnt mit dem tatsächlich ausgewählten Pilotcheckpoint `artifacts/banana-individual-v3-initial.pt`; sein SHA-256 ist im eingefrorenen Trainingsplan festgelegt. Damit kann das Training ohne erneuten Pilotlauf reproduziert werden. Der endgültige ausgewählte Checkpoint steht unter `artifacts/banana-yolox-nano.pt`.

Der Pilot hat 13 vollständige Epochen mit Seed 20261112 durchlaufen; höchstes Validierungs-F1 wählte Epoche 11. Er wurde während Epoche 14 beendet, um bereits eingesehene reale Büschel in das weitere Training aufzunehmen. Keine Inferenz des neuen Tests war erfolgt. Die tatsächlichen 13 Epochen bleiben unter `public/evaluation/pilot-v2/training-history.json` erhalten. Eine optionale Reproduktion des Piloten lautet:

```bash
git show 6b682c4:ml/artifacts/banana-yolox-nano.pt > ml/work/base-v0.1.pt
ml/.venv/bin/python ml/train_individual_v2.py --stop-after 13
```

`--stop-after 13` behält den ursprünglichen Kosinusplan mit maximal 24 Epochen bei und beendet nur den Lauf früher. Die Fortsetzung verwendet Seed 20261113, höchstens zwölf Epochen, AdamW, EMA, vollständiges Backbone ab Epoche 1 und späte L1-Lokalisierung. Das höchste bestätigte Reife-F1 auf der Validierung wählt den Checkpoint. Tatsächliche Laufzeit, gewählte Epoche und alle Validierungswerte stehen in `public/models/model-card.json` und `public/evaluation/individual-training-history.json`.

Zusätzliche Trainingsaugmentation: 467 aus ursprünglichen Trainingsfotos gewonnene GrabCut-Fruchtmasken; 600 Montagen je Epoche mit drei bis neun Früchten, Überlappungen, Drehungen und Beleuchtungsänderungen. Verdeckte Zielboxen werden auf sichtbare Pixel verkleinert; nahezu vollständig verdeckte Früchte werden ausgelassen. Zu 30 % werden einfarbige Hintergründe verwendet. Echte zusätzliche positive Fotos werden sechsmal je Epoche gezogen. **Keine Montage zählt als realer Test.**

## Validierungswahl und einmaliger Test

`release_v2.py` exportiert den ausgewählten Fortsetzungscheckpoint mit ONNX-Opset 17, fusioniert Conv/BatchNorm und prüft PyTorch/ONNX numerisch. Die ausschließlich auf Validierung geprüften Kombinationen umfassen NMS-IoU 0,45–0,90, Detektionsschwelle 0,20–0,50, Klassifikationsschwelle 0,40–0,90 und Klassenmarge 0,10–0,40. Die tatsächliche Browser-Objektverfolgung bewertet jede Kombination nach drei Bestätigungen. Auswahl: mindestens 95 % Präzision, dann höchster Recall; falls kein Punkt genügt, höchstes F1 mit dokumentiertem Fehlschlag.

Erst danach wird der neue eingefrorene Test einmal mit dem fertigen ONNX-Artefakt ausgewertet. IoU ≥0,50; unsichere oder unbestätigte reife Früchte zählen als ausgelassene Empfehlung. Der bisherige 101-Foto-Test wird separat ausgewiesen. Das Exportskript verweigert ein Überschreiben eines bereits eingesehenen neuen Tests. Eine erneute Berechnung der schon gespeicherten Empfehlungsmetriken benötigt kein Training:

```bash
node --import tsx scripts/evaluate-recommendations.ts public/evaluation/individual-predictions.json ml/work/reproduced-recommendations.json
ml/.venv/bin/python ml/write_summary.py
```

Nach weiterem testgetriebenem Tuning ist ein neuer, vorab festgelegter unberührter Holdout erforderlich. Bereits eingesehene Fehlresultate dürfen nicht gelöscht und als unabhängige neue Testergebnisse ausgegeben werden. Historische Berichte des Modells 0.1 tragen den Präfix `legacy-v0.1-`; das fehlgeschlagene Experiment 0.2 bleibt ausdrücklich archiviert.

## Gemeinsamer Laufzeitvertrag

Eingang `images`: Float32 `[1,3,448,448]`, BGR, 0…255, links oben eingeordnet, Padding rechts/unten 114. Ausgang `predictions`: `[1,4116,8]` mit bereits dekodiertem `cx,cy,w,h,objectness,unreif,reif,überreif`. Kein zweiter Grid-Decode.

`lib/postprocess-config.json` wird von TypeScript und `postprocess.py` gelesen. Gemeinsame Regeln: Objektwert × größter Klassenwert, Clipping, Mindestboxgröße, klassenübergreifende NMS, stabile Reihenfolge bei Gleichstand, höchstens 40 Treffer, Klassenmarge und Unsicherheit. `npm test` vergleicht echte Python- und TypeScript-Ausgaben direkt. Die App kann die Sicherheitsschwelle erhöhen; sie unterschreitet den auf Validierung kalibrierten Mindestwert für Empfehlungen nicht.

Die Empfehlung ist ein Trackingzustand: drei sichere, eindeutig zugeordnete Reifebewertungen; bei Unsicherheit, fehlender Frucht, Unterbrechung oder zu alten Liveframes wird sie gelöscht. Ein statisches Beispiel benötigt ebenfalls drei tatsächliche Modellaufrufe, bleibt aber unabhängig von der Rechenzeit dasselbe Foto.

## Nächstes Experiment: mehr echte Fotos und Ausschnitte

Der neue Prüfbestand in `individual-benchmark-v4.json` wird vor dem nächsten Training committed. Er enthält **30 weitere reale Commons-Büschelfotos und 30 weitere COCO-Negativfotos**. Die ausgewählten Ausschnitte bilden eine nähere Kameraansicht nach; sie belegen keine Genauigkeit auf dem gesamten ursprünglichen Foto. Alle sichtbaren reifen Instanzen bleiben Ziele, auch wenn sie teilweise verdeckt sind. Unsichere Modellantworten zählen weiterhin als ausgelassene Empfehlungen. Bildquellen, Autoren, Lizenzen, Einzelboxen und Dateihashes sind im Manifest enthalten. Die Codex-Annotationen besitzen keine unabhängige Expertenabnahme.

17 zusätzliche echte Trainingsfotos ergänzen den bisherigen Bestand. Die bereits eingesehenen alten 60 Prüffotos werden ausdrücklich zu Validierungsdaten. Neuer Stand: **592 Training / 177 Validierung / 101 alter Einzelbildtest / 60 neuer Holdout**. Verwandte Ansichten bleiben aus anderen Splits ausgeschlossen; der Quellen-, Pixel- und pHash-Audit vergleicht die unbeschnittenen Quellen.

Das Modell verwendet weiter YOLOX-Nano mit 448 Pixeln. Die Fortsetzung reduziert die Montage-Slots von 600 auf 150 je Epoche, gewichtet echte positive Fotos achtfach und zeigt in der Hälfte der echten Mehrfruchtfälle einen zufälligen Ausschnitt. Zielboxen werden auf sichtbare Bereiche begrenzt; fast vollständig entfernte Früchte bleiben nicht als volle Zielbox stehen. Der unveränderte Startcheckpoint liegt unter `artifacts/banana-individual-v4-initial.pt`. Der neue Holdout wird während Training und Schwellwertwahl nicht inferiert.

```bash
ml/.venv/bin/python ml/build_data_v4.py
ml/.venv/bin/python ml/audit_individual_data.py --data ml/work/individual-v4 --output ml/work/v4-split-audit.json
ml/.venv/bin/python ml/verify.py
ml/.venv/bin/python ml/train_individual_v4.py
ml/.venv/bin/python ml/refine_individual_v4.py
ml/.venv/bin/python ml/evaluate_individual_v4.py
node --import tsx scripts/prepare-benchmark-examples.ts
ml/.venv/bin/python ml/render_benchmark_examples.py
```

Die erste Phase endete nach 20 Epochen; Validierungs-F1 wählte Epoche 13. Weil der Frühstopp vor den spät geplanten L1-Epochen lag und Lokalisierungsfehler überwiegen, ist eine gesondert vor der Ausführung festgelegte Feinabstimmung mit vier L1-Epochen ergänzt. Plan: `individual-refinement-v4.json`; unveränderter Startcheckpoint: `artifacts/banana-individual-v4-stage1.pt`. Auch dieser Plan wird vor Ausführung committed; die Fotos und Annotationen des neuen Holdouts bleiben unverändert und weiterhin nicht inferiert. Der alte ausgewählte Checkpoint bleibt zusätzlich zu den vier neuen Epochen wählbar. Die Feinabstimmung darf direkt vom mitgelieferten Checkpoint und dem aufgezeichneten Erstverlauf aus reproduziert werden; ein erneuter Pilotlauf ist dafür nicht nötig. Laufzeiten sind Messwerte des tatsächlichen Laufs und keine bitweise reproduzierbaren Zeitangaben.

Die Auswertung wartet auf den vollständigen Trainingslauf einschließlich dieser vier Epochen. Sie wählt Schwellwerte ausschließlich auf der Validierung und inferiert danach den neuen Holdout genau einmal. Das Skript verweigert ein erneutes Überschreiben des Tests. Fehlgeschlagene Kandidaten bleiben unter `public/evaluation/experiment-v4/` archiviert und ersetzen die App-Gewichte nicht. Nur ein statisch abgenommener Kandidat wird für die weiteren App- und Geräteprüfungen übernommen.

Dieses Experiment allein gibt keine Veröffentlichung frei. Die bestehende Modellprüfung 0.3 und alle fehlgeschlagenen Ergebnisse bleiben erhalten. Die echten Android-/Chrome- und iPhone-/Safari-Prüfungen bleiben erforderlich.

`npm run test:browser` baut den Produktionsstand und prüft ihn unter dem Pages-Unterpfad `/Bananenblick/`, einschließlich langsamen Laufzeitdownloads, verspäteter Kamerafreigabe nach Stopp, Modellhashfehlern und GPU-Hang mit WASM-Neustart. Chromium/WebKit verwenden mobile Linux-Emulation; echte Geräteprüfungen werden dadurch nicht ersetzt. Physische Android-/Chrome- und iPhone-/Safari-Nachweise bleiben gesondert in `public/evaluation/device-validation.json` dokumentiert. Die private Sites-Veröffentlichung gehört nicht zu dieser Änderung.
