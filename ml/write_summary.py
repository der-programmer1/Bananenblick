#!/usr/bin/env python3
"""Generate the current report from actual measured results and frozen provenance."""
from pathlib import Path
import json
from decimal import Decimal,ROUND_HALF_UP
ROOT=Path(__file__).resolve().parent;SITE=ROOT.parent
load=lambda p:json.loads((SITE/p).read_text())
card=load('public/models/model-card.json');report=load('public/evaluation/individual-report.json')
legacy=load('public/evaluation/test-report.json')['test'];training=card['training'];thresholds=card['thresholds']
is_v4=card['version']=='0.4.0'
plan=load('ml/individual-benchmark-v4.json' if is_v4 else 'ml/individual-training-v3.json');selection=load('public/evaluation/individual-validation-selection.json')
pct=lambda x:f"{Decimal(str(x*100)).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP)} %".replace('.',',')
status='Die Zielwerte im statischen Fototest sind erreicht.' if report['passed'] else 'Die Zielwerte für essreife Einzelbananen sind nicht erreicht.'
benchmark_file='individual-benchmark-v4.json' if is_v4 else 'individual-benchmark-v2.json'
benchmark_intro=(f"Der neue Bestand wurde am {report['frozenAt']} in Commit `{report['frozenCommit'][:7]}` vor Training festgelegt: **30 reale Commons-Büschelfotos und 30 COCO-Negativfotos**. Die dokumentierten Fotoausschnitte bilden nähere Ansichten nach; das Ergebnis belegt keine Genauigkeit auf den vollständigen Quellfotos."
    if is_v4 else f"Der neue Bestand wurde am {report['frozenAt']} in Commit `59efe9e` vor dem vollständigen Nachtraining festgelegt: **30 reale Büschelfotos von Wikimedia Commons und 30 reale Fotos ohne Bananen aus COCO 2017**. Die 60 Fotos und ihre Annotationen blieben auch bei der Trainingsfortsetzung in Commit `18308d4` unverändert.")
training_brief=(f"{training['pilot']['completedEpochs']} abgeschlossene Pilotepochen und {training['epochs']} Fortsetzungsepochen" if 'pilot' in training else f"{training['epochs']} abgeschlossene Fortsetzungsepochen")
training_intro=(f"Der Pilot wurde nach {training['pilot']['completedEpochs']} vollständigen Epochen beendet; Validierungs-F1 wählte Pilotepoche {training['pilot']['selectedEpoch']}. Darauf folgten {training['epochs']} Fortsetzungsepochen" if 'pilot' in training else f"Der unveränderte Startcheckpoint 0.3 wurde in {training['epochs']} vollständigen Epochen weitertrainiert")
if 'refinement' in training:
    training_intro+=f" ({training['refinement']['stage1CompletedEpochs']} im ersten Lauf; {training['refinement']['epochs']} weitere L1-Epochen ab dem ausgewählten Erstcheckpoint, mit unverändertem Holdout)"
data_intro=("17 weitere reale Commons-Fotos ergänzen das Training. Die zuvor eingesehenen 60 Prüffotos der Version 0.3 sind ausdrücklich Validierungsdaten; sie werden nicht als unabhängiger neuer Test ausgegeben." if is_v4 else "Aus dem zuvor fehlgeschlagenen und bereits eingesehenen v1-Test wurden 29 positive Fotos ausdrücklich Trainingsdaten und 30 Negativfotos ebenfalls Trainingsdaten; das visuell unbestimmbare positive Foto blieb in der Validierung. Sechs zusätzliche COCO-Ausschnitte erhielten vor der Fortsetzung neue Einzelboxen ausschließlich für die Validierung.")
augmentation_intro=("150 zusätzliche Trainingsmontagen je Epoche und achtfache Gewichtung echter positiver Fotos. Die Hälfte der realen Mehrfruchtfälle erhält einen zufälligen Ausschnitt mit auf den sichtbaren Bereich begrenzten Einzelboxen." if is_v4 else "600 zusätzliche Trainingsmontagen je Epoche und sechsfache Gewichtung echter positiver Zusatzfotos.")
additional_experiment=''
if not is_v4 and (SITE/'public/evaluation/individual-v4-report.json').exists():
    experiment=load('public/evaluation/individual-v4-report.json')
    additional_experiment=(f"Der weitere, getrennt dokumentierte Trainingsversuch 0.4 prüfte einen neuen, in Commit `{experiment['frozenCommit'][:7]}` vor Training festgelegten Bestand mit 30 echten Büschelfotos (dokumentierte Ausschnitte) und 30 neuen Negativfotos. Ergebnis: **{experiment['truePositive']} TP, {experiment['falsePositive']} FP, {experiment['falseNegative']} FN; Präzision {pct(experiment['precision'])}, Recall {pct(experiment['recall'])}**. Er hat die Zielwerte ebenfalls nicht erreicht und ersetzt die Gewichte 0.3 im Änderungsstand nicht. Die öffentliche Pages-Version bleibt 0.1.0. [Bericht mit Grenzen](./experiment-v4/report.md), [Ergebnisse](./individual-v4-report.json), [Validierungswahl](./experiment-v4/validation-selection.json), [Trainingsverlauf](./experiment-v4/training-history.json), [Beispiele aus gespeicherten Ausgaben](./experiment-v4/examples.md).")
lines=[
f"# Bananenblick {card['version']}: Modellprüfung",'',
f'**{status} Echte Android-/iPhone-Prüfungen und die Veröffentlichungsabnahme stehen aus.**','',
'## Unberührter Prüfbestand und Ergebnis','',
benchmark_intro+' Keine Montage zählt als Prüfszene. Modell und Schwellwerte wurden ausschließlich mit Trainings- und Validierungsdaten gewählt, danach wurde der neue Bestand einmal ausgewertet.','',
'| Empfehlung nach drei Bestätigungen | Ergebnis | Ziel |','|---|---:|---:|',
f"| Präzision | {pct(report['precision'])} | ≥ 95 % |",
f"| Recall sichtbarer reifer Einzelbananen | {pct(report['recall'])} | ≥ 80 % |",
f"| Richtige / falsche Empfehlungen | {report['truePositive']} / {report['falsePositive']} | |",
f"| Ausgelassene reife Einzelbananen | {report['falseNegative']} | |",
f"| Falsche Empfehlungen auf Negativfotos | {report['negativeFalseRecommendations']} | |",'',
'IoU ≥ 0,50 definiert eine passende Einzelbox. Unsichere, nicht dreimal bestätigte oder nicht lokalisierte reife Früchte zählen als ausgelassene Empfehlungen. Die exakte TypeScript-Objektverfolgung erhält je Foto drei identische Inferenzergebnisse in 200-ms-Abständen. Dies prüft die Empfehlung auf statischen Fotos; es belegt keine zeitliche Stabilität einer echten Kamera. Zweifelhafte Zuordnungen dürfen auch bei statischen überlappenden Boxen keine Bestätigungen ansammeln.','',
'Die sichtbaren Einzelboxen und Reifestufen wurden vom Coding-Assistenten visuell annotiert und geprüft. Eine unabhängige menschliche Gegenprüfung fehlt. Ungewöhnliche oder nicht sicher erkennbare Schalen sind als nicht beurteilbar markiert. Verdeckte Früchte zählen nach sichtbarem Umfang, vollständig verdeckte Früchte nicht als sichtbare Instanzen. Quellen, Einzelboxen, Ausschnitte, Dateihashes und Lizenzen sind im eingefrorenen Manifest dokumentiert.','',
'## Training und Auswahl','',
f"YOLOX-Nano, {card['inputSize']} × {card['inputSize']} BGR/Float32, vollständiges Backbone ab der ersten Epoche. {training_intro} mit Seed {training['seed']}; höchstes bestätigtes Reife-F1 auf der Validierung wählte Epoche {training['selectedEpoch']}. CPU/PyTorch, zwei Threads, keine Bild- oder Inferenz-API.",'',
f"Die endgültigen Splits enthalten **{training['data']['train']} Training / {training['data']['val']} Validierung / {training['data']['test']} bisheriger Test / {training['data']['acceptance']} neuer Test**. 471 Mendeley-Trainingsfotos werden durch echte COCO- und Commons-Fotos ergänzt. {data_intro} Es gibt keine Überlappung der Bildgruppen zwischen Training, Validierung und dem neuen Test.",'',
augmentation_intro+' Die Montagen verwenden ausschließlich aus ursprünglichen Trainingsfotos gewonnene Fruchtmasken. Überdeckungen verkleinern die sichtbaren Zielboxen; fast vollständig verdeckte Früchte werden ausgelassen. Echte Trainingshintergründe und zu 30 % einfarbige Hintergründe, Drehungen und Beleuchtungsänderungen ergänzen die realen Fotos. Diese Montagen sind Trainingsaugmentation und kein Nachweis von realer Erkennungsleistung.','',
'Das erste Experiment 0.2 mit einem früh ausgewählten Head-Checkpoint verfehlte die Ziele (0 richtige Empfehlungen, 1 falsche, 169 ausgelassene reife Früchte). Seine Originalberichte bleiben unter `experiment-v1/` erhalten. Dieser bereits gesehene Bestand wird nicht als unabhängiger neuer Test ausgegeben. COCO-Trainingsfotos waren im generischen Vortraining enthalten; die 30 Commons-Prüffotos sind davon getrennt. Physische Frucht- und Aufnahmeserien-IDs sind bei den öffentlichen Quellen nicht verfügbar.','',
f"{len(selection['candidates'])} Kombinationen wurden ausschließlich auf der Validierung geprüft: NMS-IoU 0,45/0,55/0,65/0,75/0,85/0,90, Detektion 0,20–0,50, sichere Klassifikation 0,40–0,90 und Klassenmarge 0,10–0,40. Die Auswahl verlangt ≥95 % Präzision und maximiert dann Recall; falls kein Punkt genügt, wählt sie höchstes F1 und dokumentiert das Verfehlen. Gewählt: Detektion **{thresholds['detection']}**, sichere Klassifikation **{thresholds['classification']}**, Klassenmarge **{thresholds['margin']}**, klassenübergreifende NMS-IoU **{thresholds['nmsIou']}**. Nach Einsicht des neuen Tests wurde nichts nachjustiert.",'',
'Python und Browser lesen dieselbe versionierte Konfiguration. Ein direkter automatisierter Vergleich prüft beide Decoder auf identischen Daten einschließlich Clipping, Gleichständen, Überlappung und Unsicherheit. Die globale eindeutige Zuordnung nutzt IoU und Bewegung; zweifelhafte Zuordnungen übertragen keine bestätigte Reife. Fehlende oder unsichere Treffer verlieren die Empfehlung sofort. Ein Neustart setzt die Nummern auf 1 zurück. Die App prüft den SHA-256 der geladenen Gewichte gegen die Modellbeschreibung.','',
'Kompakte Nummernschilder vermeiden überdeckte Beschriftungen auf schmalen Bildschirmen. Freie Rahmenpositionen und bei Bedarf Zuordnungslinien halten die Zahlen lesbar; der vollständige Status steht je Frucht in der Ergebnisliste. Ein Mobilfall prüft zwanzig dicht überlappende Rahmen. Die Startüberwachung nutzt den tatsächlichen Laufzeitprovider: Eine kalte WebAssembly-Sitzung erhält auch beim Wechsel von WebGPU während Start oder Auswertung ihr eigenes Zeitlimit. Sitzungsaufbau und laufende Auswertung haben getrennte Zeitbudgets; während des Wechsels werden alte Empfehlungen gelöscht.','',
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
if additional_experiment:
    index=lines.index('## Quellen')
    lines[index:index]=['## Weiterer Trainingsversuch', '', additional_experiment, '']
text='\n'.join(lines)
(SITE/'public/evaluation/model-report.md').write_text(text)
(SITE/'documentation/model-validation.md').write_text(text.replace('](./', '](../public/evaluation/'))
print('Current model report generated from actual results.')

readme=f'''# Bananenblick · nummerierte Einzelbananen

**Stand {card['version']} – Veröffentlichungsabnahme noch offen.** Die bestehende öffentliche App steht unter [der-programmer1.github.io/Bananenblick](https://der-programmer1.github.io/Bananenblick/). Der neue Stand liegt im Änderungsbranch. {status} Tatsächliche Android-/iPhone-Prüfungen fehlen.

Jeder aktuelle Treffer erhält eine eindeutige Nummer. Rahmen und Ergebnisliste verwenden dieselbe Nummer; Antippen wählt die zugehörige Frucht aus. Die globale eindeutige Zuordnung hält Nummern bei leichter Bewegung stabil. Beim Neustart beginnt die Nummerierung wieder bei 1.

Ausschließlich drei aufeinanderfolgende sichere Bewertungen als „reif“ ergeben einen grünen Rahmen und „✓ Jetzt essreif“. Unsicherheit, ein verschwundener Treffer oder eine zweifelhafte Zuordnung unterbrechen die Bestätigung. Andere Ergebnisse heißen „Unreif“, „Überreif“ oder „Nicht sicher beurteilbar“. Der Überblick nennt die empfohlenen Nummern oder zeigt „Aktuell keine Banane sicher als essreif erkannt“. Bewertet wird ausschließlich die sichtbare Schalenreife.

## Tatsächlicher Modellstand

Die bestehende YOLOX-Nano-Pipeline wurde mit engen Einzelboxen vollständig nachtrainiert. {training_brief}, {card['inputSize']} × {card['inputSize']}, zwei CPU-Threads. Datensätze der Fortsetzung: **{training['data']['train']} Training / {training['data']['val']} Validierung / 101 bisheriger Test / 60 neue Prüffotos**. Überlappung, wechselndes Licht, gemischte Schalenreife und echte Negativfotos sind ergänzt; Trainingsmontagen verwenden ausschließlich ursprüngliche Trainingsfrüchte.

{benchmark_intro} Ergebnis nach drei Bestätigungen: **{report['truePositive']} richtige Empfehlungen, {report['falsePositive']} falsche Empfehlungen, {report['falseNegative']} ausgelassene reife Einzelbananen; Präzision {pct(report['precision'])}, Recall {pct(report['recall'])}**. Ziel: ≥95 % Präzision, ≥80 % Recall. Unsicherheit zählt als ausgelassene Empfehlung.

Die Annotation wurde vom Coding-Assistenten visuell geprüft; eine unabhängige menschliche Gegenprüfung fehlt. COCO-Trainingsfotos waren im generischen Vortraining enthalten. Das frühere fehlgeschlagene Experiment bleibt archiviert; seine bereits eingesehenen Szenen sind ausdrücklich Trainings-/Validierungsdaten und kein neuer unabhängiger Test.

- [Modellbericht](documentation/model-validation.md)
- [Neue Empfehlungsmetriken](public/evaluation/individual-report.json)
- [Eingefrorene Szenen, Boxen, Hashes und Quellen](ml/{benchmark_file})
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

Der [AppShelf-Eintrag](https://der-programmer1.github.io/AppShelf/apps/bananenblick/) verlinkt die bestehende öffentliche Version 0.1.0 als Prototyp. Das noch nicht abgenommene Büschel-Update ist damit nicht als veröffentlicht gekennzeichnet.

Quellen: Rohit et al. (2026), [Mendeley Banana Ripeness Stage Dataset](https://data.mendeley.com/datasets/56c75hvm7r/1), CC BY 4.0; [COCO 2017](https://cocodataset.org/) und Wikimedia Commons mit Quellen und individuellen Lizenzen im Manifest. Tellerbeispiel: [goodiesfirst, „putahracsa fruit plate“](https://www.flickr.com/photos/scaredykat/4430551771/), CC BY 2.0, verkleinert. Gemischter Büschel: [Esther Musasike, „Bunch of banana.jpg“](https://commons.wikimedia.org/wiki/File:Bunch_of_banana.jpg), CC BY-SA 4.0, Ausschnitt und verkleinert. YOLOX: Megvii, Apache 2.0. Ursprüngliche Herkunft: offizieller Sites-/Vinext-Starter; das Standard-Buildziel ist der statische Pages-Export.
'''
if additional_experiment:
    readme=readme.replace('## Lokal prüfen', '## Weiterer Trainingsversuch\n\n'+additional_experiment.replace('](./', '](public/evaluation/')+'\n\n## Lokal prüfen')
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

acceptance_path=SITE/'public/evaluation/acceptance.md'
old_protocol=acceptance_path.read_text().partition('## Noch erforderlich: echte Geräte')[2]
if not old_protocol:raise RuntimeError('Keep the physical-device protocol when updating the report.')
old_protocol=old_protocol.replace('Die öffentliche HTTPS-Seite direkt öffnen; Emulation und Node-Zahlen zählen nicht als Gerätetest.', 'Den zu prüfenden Build unter HTTPS oder über eine lokale USB-/Portweiterleitung öffnen und den Modellhash kontrollieren. Die öffentliche Altversion zählt nicht als Prüfung der neuen Gewichte. Emulation und Node-Zahlen zählen nicht als Gerätetest.')
acceptance=f'''# Prüfprotokoll · Bananenblick {card['version']}

**{status} Veröffentlichungsabnahme offen; keine Veröffentlichung dieses Stands.** Der bisherige GitHub-Pages-Stand bleibt bestehen.

## Tatsächlich ausgeführt

| Prüfung | Ergebnis und Grenze |
|---|---|
| TypeScript | Bestanden |
| Automatisierte Tests | 22 bestanden: Überlappung, globale eindeutige Zuordnung, Bewegung, Unsicherheit, drei Bestätigungen, fehlende Treffer, lesbare Nummern in dichten Büscheln, Liste/Rahmen, Python-/TypeScript-Parität |
| Python-Pipeline | 12 Tests bestanden, einschließlich Verdeckung, sichtbarer Zielboxen, echter Fotoausschnitte mit erhaltenen Klassen, 448-Pixel-Spiegelung, ursprünglicher Split-Integrität und Metriksemantik |
| Chromium unter Linux, mobile Pixel-7-Ansichtsgröße | Sieben Browserfälle bestanden am Produktionsbuild unter `/Bananenblick/`: kalter WASM-Start, GPU-/WASM-Wechsel beim Start und während Auswertung (kontrollierte Uhr), verspätete Kamerafreigabe nach Stopp, falscher Modellhash, kontrollierte Kamera/Worker mit Auswahl/Bewegung/Unsicherheit/Verschwinden/Hoch-/Querformat/Stopp/Nummernreset/GPU-Hang und zwanzig dicht überlappenden Nummernschildern, zusätzlich echtes ONNX-Beispiel mit verzögertem Download und drei Durchläufen |
| WebKit unter Linux, mobile iPhone-13-Ansichtsgröße | Dieselben sieben Browserfälle bestanden; WebKit-WPE ist kein echtes iPhone/Safari |
| Tatsächliche physische Geräte | **Keine** Android-/Chrome- oder iPhone-/Safari-Geräte geprüft |
| WASM unter Node/Linux | Echter ONNX-Aufruf, endliche `[1,4116,8]`-Ausgabe und bildabhängige Ergebnisse; Werte in `web-runtime.json` |
| Statischer Pages-Build | Bestanden; relative Assets, Worker, Modellhashes und gemeinsame Schwellwerte geprüft; Browser laufen auf diesem Build |
| Neue Datenprüfung | Keine Bildgruppenüberschneidung, identischen Quellbilder oder pHash-Nahduplikatkandidaten zwischen neuem Test und übrigen Splits |
| Neuer realer Büscheltest | 30 reale Commons-Büschelszenen und 30 COCO-Negativszenen vor Training eingefroren; **{report['truePositive']} TP, {report['falsePositive']} FP, {report['falseNegative']} FN**, Präzision **{pct(report['precision'])}**, Recall **{pct(report['recall'])}** |

{benchmark_intro} Bilder, Annotationen, Lizenzen und Hashes stehen in `individual-benchmark.json`. Die zusätzliche Einzelannotation wurde vom Coding-Assistenten visuell geprüft; eine unabhängige menschliche Gegenprüfung fehlt. Öffentliche Quellen liefern keine physischen Frucht-/Aufnahmeserien-IDs. Das alte fehlgeschlagene Experiment wurde ausdrücklich Training/Validierung und ist unter `experiment-v1/` archiviert.

Die Empfehlungsmetrik nutzt den echten TypeScript-Tracker mit drei identischen Ergebnissen je statischem Foto. Unsichere Ergebnisse zählen als ausgelassene Empfehlungen. Auf den 30 Negativfotos entstanden {report['negativeFalseRecommendations']} falsche Empfehlungen. Zu große Gesamtrahmen ohne passende Einzelbox zählen nicht als richtige Erkennung. Nach Sichtung des neuen Tests erfolgte kein Nachjustieren.

Browser-Kameratests verwenden kontrollierte Video-/Worker-Doubles. Chromium verwendet einen lokalen Canvas-Medienstream; WebKit simuliert Medienquelle und Trackzustände. Das zusätzliche Beispiel verwendet unveränderte Bildpixel und den echten Worker. Modellhash und Browserversion werden vom Lauf erfasst und in `device-validation.json` dokumentiert. Ein künstlich um 13 Sekunden verzögerter WASM-Download prüft, dass Netzwerklatenz keinen unnötigen GPU-Neustart auslöst. Kein solcher Test belegt den physischen Kameraindikator, echte Smartphone-Leistung oder echten Kamera-Netzwerkverkehr.

## Modellabnahme

Ziel: ≥95 % Präzision und ≥80 % Recall für bestätigte Empfehlungen sichtbarer reifer Einzelbananen bei IoU ≥0,5; Enthaltungen zählen als ausgelassene Empfehlungen. {status} Der Bericht enthält die tatsächlich gemessenen Werte. Nach weiterem testgetriebenem Tuning muss ein neuer Holdout vor Training festgelegt werden; der jetzige Test ist nun eingesehen.

## Noch erforderlich: echte Geräte'''+old_protocol
if additional_experiment:
    acceptance=acceptance.replace('## Modellabnahme','## Weiterer Trainingsversuch\n\n'+additional_experiment+'\n\n## Modellabnahme')
acceptance_path.write_text(acceptance)

if (SITE/'public/evaluation/individual-v4-report.json').exists():
    trial=load('public/evaluation/individual-v4-report.json')
    trial_card=load('public/evaluation/experiment-v4/model-card.json')
    trial_selection=load('public/evaluation/experiment-v4/validation-selection.json')['selected']
    trial_training=trial_card['training']
    baseline=trial['baselineValidation'];selected=trial['selectedEpochValidation']
    trial_text=f'''# Trainingsversuch 0.4 · tatsächliche Ergebnisse

**{'Die statischen Zielwerte sind erreicht; die Geräteabnahme bleibt offen.' if trial['passed'] else 'Die statischen Zielwerte sind verfehlt. Der Änderungsstand behält Modell 0.3; öffentlich bleibt Version 0.1.0.'}** Dieser Bericht ist keine abgeschlossene Veröffentlichung oder echte Smartphone-Prüfung.

## Unabhängiger neuer Fotobestand

Vor dem Training in Commit `{trial['frozenCommit']}` festgelegt: 30 reale Commons-Büschelfotos und 30 weitere reale COCO-Negativfotos. Die dokumentierten Ausschnitte bilden nähere Ansichten nach. Der Bestand enthält {trial['truePositive']+trial['falseNegative']} annotierte reife Einzelbananen. Er wurde nach Checkpoint- und Schwellwertwahl einmal ausgewertet. Keine Montage zählt als Prüfszene.

| Bestätigte essreife Einzelbananen | Ergebnis | Ziel |
|---|---:|---:|
| Präzision | {pct(trial['precision'])} | mindestens 95 % |
| Recall | {pct(trial['recall'])} | mindestens 80 % |
| Richtige Empfehlungen | {trial['truePositive']} | |
| Falsche Empfehlungen | {trial['falsePositive']} | |
| Ausgelassene reife Früchte | {trial['falseNegative']} | |
| Falsche Empfehlungen auf Negativfotos | {trial['negativeFalseRecommendations']} | |

Passende Einzelbox: IoU ≥0,5 und eindeutige Zuordnung. Die tatsächliche TypeScript-Objektverfolgung bewertet je Foto drei identische gespeicherte Inferenzergebnisse. Enthaltungen und fehlende Bestätigungen zählen als ausgelassene Empfehlungen. Dies belegt weder zeitliche Kamerastabilität noch Smartphone-Durchsatz.

## Training und Auswahl

592 Trainingsfotos, 177 Validierungsfotos und 101 separat ausgewiesene bisherige Einzelbild-Testfotos. Die bereits eingesehenen alten 60 Prüffotos sind jetzt ausdrücklich Validierungsdaten. 17 weitere echte Trainingsfotos, achtfache Gewichtung realer Zusatzfotos, 150 statt 600 Montage-Slots je Epoche, echte Fotoausschnitte mit auf den sichtbaren Bereich begrenzten Zielboxen. YOLOX-Nano, 448 × 448, zwei CPU-Threads.

20 vollständige Epochen im ersten Lauf, danach vier gesondert vorab festgelegte L1-Epochen ab dem ausgewählten Erstcheckpoint. Der Holdout blieb bei beiden Phasen unverändert und nicht inferiert. Der Erstcheckpoint blieb zusätzlich zu den vier neuen Checkpoints wählbar. Insgesamt {trial_training['epochs']} Epochen tatsächlich gerechnet; die Validierung wählte Epoche {trial_training['selectedEpoch']}.

Beim festen Validierungspunkt (Detektion 0,25; Klassifikation 0,6; Marge 0,15; NMS 0,65) änderte sich F1 von {baseline['f1']:.3f} auf {selected['f1']:.3f}. Diese Zahlen betreffen dieselbe erweiterte Validierung und sind keine Ergebnisse des neuen Tests.

Die abschließende Schwellwertwahl untersuchte 1320 Kombinationen ausschließlich auf der Validierung. Gewählt: `{json.dumps(trial['thresholds'],ensure_ascii=False)}`. Validierungspräzision {pct(trial_selection['precision'])}, Validierungsrecall {pct(trial_selection['recall'])}. Das Kriterium verlangt mindestens 95 % Präzision und maximiert danach Recall; erst danach wurde der neue Test inferiert. Nach Einsicht des Tests erfolgte keine Anpassung.

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

Tatsächlich geprüfte physische Android-/Chrome- und iPhone-/Safari-Geräte: **keine**. Linux-Browseremulation ersetzt diese Prüfung nicht. Die reine ONNX-CPU-Inferenz im Fototest brauchte im Median {trial['inferenceMs']['median']:.1f} ms und p95 {trial['inferenceMs']['p95']:.1f} ms; Aufnahme, Vorverarbeitung und Darstellung sind nicht enthalten. Das belegt keine drei vollständigen Smartphone-Auswertungen pro Sekunde.

Modell-SHA-256: `{trial['modelSha256']}`. Benchmark-SHA-256: `{trial['benchmarkSha256']}`.
'''
    (SITE/'public/evaluation/experiment-v4/report.md').write_text(trial_text)
