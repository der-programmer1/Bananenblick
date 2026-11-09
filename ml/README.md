# Echte lokale Bananenerkennung

Die Web-App führt ein hier auf öffentlichen Bildern feintrainiertes **YOLOX-Nano** aus. Das Modell ist ein echter Detektor mit drei Klassen, kein Farbregelwerk und keine feste Demonstrationsanimation. Alle Reifeklassen und Rahmen kommen aus den ONNX-Ausgaben.

## Reproduzieren auf einer CPU

Getestet mit Linux, Python 3.12, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu und drei CPU-Threads. Es wird kein CUDA-Treiber und keine GPU benötigt. Andere CPUs können deutlich länger brauchen. Datenvorbereitung benötigt zeitweise ca. 2,1 GB freien Platz; PyTorch und virtuelle Umgebung zusätzlich ca. 1 GB. Das Originalarchiv wird nie ins Repository aufgenommen.

```bash
cd ml
python3 -m venv .venv
.venv/bin/pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python download_pretrained.py
.venv/bin/python prepare_data.py
.venv/bin/python verify.py
.venv/bin/python train.py --pretrained work/yolox_nano.pth --epochs 16 --batch-size 8 --threads 3
.venv/bin/python export_evaluate.py
.venv/bin/python stress_evaluate.py
.venv/bin/python write_summary.py
```

Windows: `.venv\Scripts\python.exe` statt `.venv/bin/python`. Für die PyTorch-Installation den passenden offiziellen CPU-Wheel-Index benutzen. Die Python-Skripte selbst setzen kein Bash voraus.

`prepare_data.py --archive /absoluter/pfad/BANANA_DATASET.zip` benutzt ein bereits vorhandenes Archiv. Download und ursprüngliche COCO-Gewichte werden per SHA-256 geprüft. Das trainierte Zwischenmodell liegt in `work/training/best.pt`, der mitgelieferte trainierbare Checkpoint in `artifacts/banana-yolox-nano.pt`, der fertige, selbständige Export in `../public/models/banana-yolox-nano.onnx`. CPU-Lernlauf und Trainingsmetrikverlauf werden protokolliert. `work/` ist absichtlich von Git ausgeschlossen.

## Daten und Trennung

Quelle: Rohit, Natiq-Ul-Haque; Raihan, Abu; Mim, Afia Sarkar; Fahima Tabassum, Fahima; Haque, Aziza (2026), **Annotated Banana Ripeness Stage Dataset**, Mendeley Data V1, [DOI 10.17632/56c75hvm7r.1](https://data.mendeley.com/datasets/56c75hvm7r/1), **CC BY 4.0**. Bilder wurden für die App verkleinert und als JPEG neu gespeichert; ursprüngliche Quelldateinamen bleiben im Audit erhalten.

Der archivierte Datensatz enthält 673 Bilder und keine YAML-Datei. Die Klassenbelegung 0=unripe, 1=ripe, 2=overripe wird von der DOI-Seite übernommen und mit den Dateinamenmehrheiten gegengeprüft. Ein paar Annotationen überschreiten den Bildrand; sie werden an sichtbaren Grenzen beschnitten und einzeln protokolliert. Die Quellannotation ist manuell und teils ungenau; es fand keine vollständige Neuannotation statt. Es werden keine Klassen aus Farbregeln neu vergeben.

Die EXIF-Ausrichtung wird **vor Hashing, Skalierung und Training auf die Pixel angewendet**. Die Quellannotation verwendet bereits die dargestellten, EXIF-orientierten Koordinaten. Überlagerte Zielboxen wurden je Reifestufe sowie für EXIF1/3/6/8 an Beispielen visuell gegengeprüft. `../public/evaluation/annotation-orientation-qa.jpg` zeigt vier Trainingsbilder mit roten **Zielannotationen**, keine Modellvorhersagen. Das ist wichtig: Ein JPEG ohne angewendete EXIF-Drehung und anschließend ohne Metadaten gespeichert würde die Frucht und ihre Annotation gegeneinander verdrehen.

Die vorhandenen Originalsplits werden vor jeglicher Augmentation zusammengeführt. Exakte Duplikate über Größe und dekodierte RGB-Pixel werden entfernt. Bildpaare mit pHash-Hamming-Abstand <=6 (alle vier Vierteldrehungen und ihre Spiegelungen) und bekannte Augmentationsnamensfamilien werden zu transitiven Gruppen verbunden. Eine Gruppe liegt vollständig in genau einem der neu erzeugten 70/15/15-Splits. Der gespeicherte Seed ist 20261109. Jede Klasse ist in jedem Split vertreten. `verify.py` prüft nachträglich, dass weder Pixelhash noch Gruppe noch erkannte pHash-Nachbarschaft einen Split überschreiten.

Das ist keine Garantie gegen alle Aufnahmeähnlichkeiten: Die Quelle liefert weder IDs der physischen Früchte noch Aufnahmeserien. Ein unabhängiger Standort-/Frucht-Holdout ist somit nicht belegt. Das Audit dokumentiert diese Grenze. Der tatsächliche Split mit Gruppen und Quelldateinamen steht in `../public/evaluation/split-audit.json`.

## Training und Modellauswahl

Die Originalarchitektur stammt aus [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), Commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`, Apache-2.0. Die verwendeten Module und Lizenz sind unter `vendor/` enthalten. Tiefe 0,33, Breite 0,25, Depthwise-Convolutions, drei Ausgabeklassen, 320×320 Eingabe. Architektur, SimOTA-Zuordnung und Loss stammen aus YOLOX; der eigene Trainingsloop verwendet durchgehend CPU-Tensoren. Ein minimales Utils-Subset und die Autocast-API wurden angepasst. Details und URLs: `provenance.json`.

Startpunkt sind die offiziellen COCO-Gewichte; die Banana-Klasse (COCO Index 46) initialisiert alle drei Klassenkanäle. Danach lernen die Ausgaben aus den annotierten Reifestufen. Die ersten beiden Epochen fixieren das Backbone; anschließend wird das gesamte Modell trainiert. AdamW, Seed, Lernrate, Cosinus-Absenkung, EMA und das Ende mit L1-Loss sind im Trainingsskript festgelegt. Nur Trainingsdaten erhalten moderate Helligkeits-/Sättigungsänderungen, Spiegelung, Größen-/Positionsänderung und bis zu 25 % Mosaike. Mosaikbilder sind keine echten gemischten Testszenen.

Das höchste mAP50 **auf Validierungsdaten** bestimmt den Checkpoint. Anschließend wird unter festen Kandidaten der Detektionsschwellwert mit dem höchsten Validierungs-F1 gewählt; Gleichstände bevorzugen höhere Präzision und danach den höheren Schwellwert. Im Export wird der Schwellwert erneut auf Validierungsdaten mit dem fertigen ONNX-Artefakt und der bildbegrenzten Browser-NMS überprüft. Testbilder werden erst danach einmal mit dem fertigen ONNX-Modell ausgewertet. Der Exportvergleich prüft die numerische Übereinstimmung zwischen PyTorch und ONNX Runtime. Ein erneutes Überschreiben des Testberichts ist standardmäßig blockiert. Nach einer testgetriebenen Modelländerung ist ein neuer, unberührter Holdout nötig.

## Browservertrag

- Modell: `/models/banana-yolox-nano.onnx`, alle Gewichte in einer Datei.
- Eingang `images`: Float32 `[1,3,320,320]`, **BGR**, Werte **0…255**, CHW.
- Seitenverhältnis beibehalten, Skalierungsfaktor `min(320/Breite, 320/Höhe)`. Zielbreite/-höhe ganzzahlig abrunden. Bild links oben bei (0,0), nur rechts/unten mit 114 auffüllen. Keine Zentrierung und keine Division durch 255.
- Ausgang `predictions`: Float32 `[1,2100,8]`; `cx, cy, w, h, objectness, unreif, reif, überreif`. Koordinaten sind bereits in 320er-Pixeln dekodiert; kein zweiter YOLOX-Grid-Decode.
- Detektionswert: `objectness * max(klassenwerte)`. Klassenübergreifende NMS mit IoU 0,45. Gewählte Schwellwerte stehen in `model-card.json`.
- `unsicher`, wenn Detektionswert (Objektwert × größter Klassenwert) <0,60 oder Abstand der beiden höchsten Klassenwerte <0,15. Diese konservative Regel ist keine kalibrierte Wahrscheinlichkeit.

## Bewertung und Grenzen

Die tatsächlichen Werte stehen in `../public/evaluation/test-report.json`: Precision, Recall und AP50 je Klasse, falsche Reifestufen, übersehene Früchte, eine Konfusionsmatrix inklusive „nicht erkannt“ und eine Matrix inklusive „unsicher“. Ein Vorhersagefehler bei passender Position zählt nicht als richtig erkannte Reifestufe. Falschalarme werden bei Precision gezählt. AP50 nutzt 101 Recall-Stützstellen, IoU >=0,50, Kandidaten ab 0,01 und maximal 40 NMS-Rahmen pro Bild. Unklassifizierte Lokalisierungsquote und gemeinsame Erkennung mit korrekter Reife werden getrennt ausgewiesen. `recall` zählt den obersten Modellklassenwert, auch wenn die App sich wegen geringer Konfidenz enthält. `visibleRecall` und `visibleTruePositive` zählen nur Treffer, die tatsächlich als Reifestufe statt „unsicher“ erscheinen. Damit bleibt die Enthaltungsquote nachvollziehbar.

Geschwindigkeit im Bericht ist ausschließlich die ONNX-Inferenz auf der beschriebenen Cloud-CPU, ohne Kamera, Pixelumwandlung, NMS und Darstellung. Sie beweist keine Smartphone-Geschwindigkeit.

Die kleinen Einzelaufnahmen mit einfachen Hintergründen belegen keine sichere Thekenerkennung. Insbesondere echte gemischte Reifestufen, dicht liegende Büschel, Überlappungen, Fremdobjekte ohne Banane, wechselndes Licht und eine unabhängige Smartphone-Kamera sind noch zu prüfen. Reale Android-/iPhone-Tests und deren Netzwerkaufzeichnung bleiben explizit ausstehend; diese Pipeline kann sie nicht simulieren oder als bestanden bescheinigen.

## Begrenzte Stresskontrollen

`stress_evaluate.py` schreibt separat `../public/evaluation/stress-report.json`. Die festen Kontrollen enthalten eine synthetische Montage der drei Reifestufen, rechnerische Teilüberdeckung, Helligkeitsvarianten, 90°-Dateidrehung, homogene Negativbilder und ein reales Public-Domain-Foto ohne Banane (NASA, verteilt von scikit-image). Quelle und Lizenz dieses Fotos stehen im Stressbericht. Es stammt nicht aus Training oder Validierung. Die kleinen Kontrollen dürfen keine unabhängige Praxistestquote vortäuschen; ihre Ausgaben führen hier weder zu Nachtraining noch zu geänderten Schwellwerten.
