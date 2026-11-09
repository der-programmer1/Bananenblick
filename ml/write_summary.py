#!/usr/bin/env python3
"""Render the actual reports into a concise German model validation document."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent;SITE=ROOT.parent
r=json.loads((SITE/'public/evaluation/test-report.json').read_text());t=r['test'];s=r['speed']
pct=lambda x:f'{x*100:.1f} %'.replace('.',',')
lines=[
'# Bananenblick: Modell und nachgewiesener Prüfstand',
'',
'**Prototyp. Die Ausgaben stammen aus einem tatsächlich feintrainierten YOLOX-Nano-Modell. Eine Abnahme auf echten Android- und iPhone-Kameras steht aus.**',
'',
'## Modell und Daten',
'',
f"CPU-Training über {r['training']['epochsRequested']} Epochen; auf Validierung gewählter Checkpoint: Epoche {r['training']['epoch']}. Eingang 320 × 320, Float32/BGR; fertiger ONNX-Export {r['export']['bytes']/1024/1024:.2f} MiB. Testvorhersagen verwenden genau diesen Export, SHA-256 `{r['modelSha256']}`.",
'',
'Daten: Rohit et al. (2026), [Annotated Banana Ripeness Stage Dataset, DOI 10.17632/56c75hvm7r.1](https://data.mendeley.com/datasets/56c75hvm7r/1), CC BY 4.0. Die Beispielbilder sind verkleinert und EXIF-normalisiert.',
'',
'| Split | Bilder | Ähnlichkeitsgruppen | Unreif | Reif | Überreif |',
'|---|---:|---:|---:|---:|---:|'
]
for name,title in [('train','Training'),('val','Validierung'),('test','Test')]:
 q=r['splits'][name];c=q['objects_per_class'];lines.append(f"| {title} | {q['images']} | {q['groups']} | {c['unreif']} | {c['reif']} | {c['überreif']} |")
lines += ['',f"Audit: {r['audit']['exactDuplicatesRemoved']} exakte Bildduplikate entfernt; {r['audit']['nearDuplicatePairsGrouped']} ähnliche Bildpaare in unteilbare Gruppen gefasst. {r['audit']['detectedCrossSplitDuplicates']} erkannte gruppenübergreifende Duplikate. pHash-Prüfung berücksichtigt Drehungen und Spiegelungen. {r['audit']['boundaryBoxesClipped']} Randboxen wurden beschnitten. EXIF-Drehungen wurden vor Skalierung und Training angewendet und mit Zielannotationen visuell geprüft.",'',
'Physische Frucht- und Aufnahmeserien-IDs fehlen. Derselbe Bananentyp oder dieselbe physische Frucht kann trotz automatischer Ähnlichkeitsprüfung in verschiedenen Gruppen vorkommen. Die Zahlen belegen daher keine unabhängige Standort-/Frucht-Generalisation. Die manuellen Annotationen sind teils ungenau.','',
'## Einmalige Holdout-Auswertung','',
f"Der Detektionsschwellwert {t['threshold']:.2f} wurde ausschließlich auf Validierung gewählt. IoU ≥ 0,50 definiert einen Positionstreffer. mAP50 im Test: **{pct(t['mAP50'])}**. Gemeinsame Erkennung mit richtiger oberster Reifeklasse: **{pct(t['recall'])} Recall**, **{pct(t['precision'])} Precision**. Als sichere Reifestufe in der Oberfläche beschriftet: **{pct(t['visibleRecall'])}**; **{t['uncertainCount']}** Treffer erhalten „unsicher“.",'',
'| Reifestufe | Testobjekte | Präzision roh | Recall roh | Sicher beschriftet | Falsche Reife | Übersehen | Unsicher |',
'|---|---:|---:|---:|---:|---:|---:|---:|'
]
for c in t['perClass']:lines.append(f"| {c['class']} | {c['support']} | {pct(c['precision'])} | {pct(c['recall'])} | {pct(c['visibleRecall'])} | {c['wrongRipeness']} | {c['missed']} | {c['uncertain']} |")
lines += ['',
'„Roh“ zählt die höchste Modellklasse auch bei einer Enthaltung der Oberfläche. „Sicher beschriftet“ zählt nur richtige Position und Reifestufe oberhalb der Konfidenz-/Margengrenze. „Falsche Reife“ zählt falsch klassifizierte, aber passend lokalisierte Testobjekte; „übersehen“ hat keinen passenden Rahmen. Falschalarme werden in der Präzision berücksichtigt. Vollständige Matrizen inklusive „unsicher“ und „nicht erkannt“ stehen im JSON-Bericht.','',
'## Geschwindigkeit','',
f"**Cloud-CPU-Referenz, keine Smartphone-Messung:** ONNX Runtime CPU mit drei Threads, Median {s['medianMs']:.1f} ms, p95 {s['p95Ms']:.1f} ms über {s['sampleCount']} Testbilder. Das sind rechnerisch {s['evaluationsPerSecond']:.1f} reine Modellaufrufe/s. Kamera, Pixelumwandlung, NMS und Darstellung sind darin nicht enthalten. Das Ziel von mindestens drei vollständigen Kameraauswertungen/s ist auf echten Handys noch nicht nachgewiesen.",'',
'## Begrenzte Stresskontrollen',''
]
stress_path=SITE/'public/evaluation/stress-report.json'
if stress_path.exists():
 st=json.loads(stress_path.read_text());q=st['summary'];lines += [f"{q['cases']} vorab festgelegte Kontrollen, davon {q['negativeControls']} ohne Zielbanane. Falschalarme in diesen wenigen Negativkontrollen: {q['falseAlarmsOnNegativeControls']}. Die positiven Varianten sind synthetische Montagen, Helligkeitsänderungen und eine Dateidrehung aus Holdout-Fotos. Ein externes reales NASA-Foto ohne Banane ist Public Domain laut offizieller scikit-image-Dokumentation. Kein Nachtraining oder Tuning auf diese Kontrollen.",'','| Kontrolle | Zielobjekte | Rahmen | Richtige rohe Treffer | Sicher richtige Treffer |','|---|---:|---:|---:|---:|']
 for case in st['results']:
  tp=sum(c['truePositive'] for c in case['metrics']['perClass']);vtp=sum(c['visibleTruePositive'] for c in case['metrics']['perClass']);lines.append(f"| {case['title']} | {case['expectedObjects']} | {case['detections']} | {tp} | {vtp} |")
 if any(case['expectedObjects']>sum(c['truePositive'] for c in case['metrics']['perClass']) for case in st['results']):
  lines += ['', 'Die aufgehellte reife Probe verfehlt das Positionskriterium IoU ≥ 0,50 trotz korrekter oberster Reifeklasse. Die Kontrollen sind somit nicht durchgehend erfolgreich; robuste Erkennung bei starken Lichtänderungen ist nicht belegt.']
else:lines += ['Noch nicht ausgeführt; keine Stressprüfung behauptet.']
lines += ['',
'Diese kleinen Kontrollen sind kein unabhängiger Praxistest. Echte gemischte Bananentheken, dicht liegende Büschel, tatsächliche Überlappungen, Wechsellicht, Bewegung, Handyrotation und repräsentative Negativszenen bleiben offen.','',
'## Prüfartefakte','',
'- [Vollständiger Modell-Testbericht](/evaluation/test-report.json)',
'- [Split-Audit mit Mitgliedschaft, Hashes und Gruppen](/evaluation/split-audit.json)',
'- [Einzelvorhersagen im Test](/evaluation/test-predictions.json)',
'- [Validierung und Schwellwertwahl](/evaluation/validation-selection.json)',
'- [Trainingsverlauf](/evaluation/validation-history.json)',
'- [Stresskontrollen mit Bildern](/evaluation/stress-report.json)',
'- [Zielannotation nach EXIF-Normalisierung](/evaluation/annotation-orientation-qa.jpg)',
'- [Modellkarte](/models/model-card.json)',
'- [Trainiertes ONNX-Modell](/models/banana-yolox-nano.onnx)',
'',
'Die CPU-Trainingsskripte, Quellverweise, verwendete YOLOX-Lizenz, Reproduktionsanleitung und Prüftests liegen im Quellprojekt unter `ml/`. Der finale CPU-Checkpoint liegt unter `ml/artifacts/banana-yolox-nano.pt`. Kamerabilder sind kein Teil dieser Trainings-/Testartefakte.',
''
]
text='\n'.join(lines)
(SITE/'docs/model-validation.md').write_text(text)
(SITE/'public/evaluation/model-report.md').write_text(text)
print('Wrote model validation summary')
