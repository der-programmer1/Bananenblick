# Prüfprotokoll · Bananenblick 0.3.0

**Die Zielwerte für essreife Einzelbananen sind nicht erreicht. Veröffentlichungsabnahme offen; keine Veröffentlichung dieses Stands.** Der bisherige GitHub-Pages-Stand bleibt bestehen.

## Tatsächlich ausgeführt

| Prüfung | Ergebnis und Grenze |
|---|---|
| TypeScript | Bestanden |
| Automatisierte Tests | 22 bestanden: Überlappung, globale eindeutige Zuordnung, Bewegung, Unsicherheit, drei Bestätigungen, fehlende Treffer, lesbare Nummern in dichten Büscheln, Liste/Rahmen, Python-/TypeScript-Parität |
| Python-Pipeline | 9 Tests bestanden, einschließlich Verdeckung, sichtbarer Zielboxen, 448-Pixel-Spiegelung, ursprünglicher Split-Integrität und Metriksemantik |
| Chromium unter Linux, mobile Pixel-7-Ansichtsgröße | Fünf Browserfälle bestanden am Produktionsbuild unter `/Bananenblick/`: kalter WASM-Start, verspätete Kamerafreigabe nach Stopp, falscher Modellhash, kontrollierte Kamera/Worker mit Auswahl/Bewegung/Unsicherheit/Verschwinden/Hoch-/Querformat/Stopp/Nummernreset/GPU-Hang und zwanzig dicht überlappenden Nummernschildern, zusätzlich echtes ONNX-Beispiel mit verzögertem Download und drei Durchläufen |
| WebKit unter Linux, mobile iPhone-13-Ansichtsgröße | Dieselben fünf Browserfälle bestanden; WebKit-WPE ist kein echtes iPhone/Safari |
| Tatsächliche physische Geräte | **Keine** Android-/Chrome- oder iPhone-/Safari-Geräte geprüft |
| WASM unter Node/Linux | Echter ONNX-Aufruf, endliche `[1,4116,8]`-Ausgabe und bildabhängige Ergebnisse; Werte in `web-runtime.json` |
| Statischer Pages-Build | Bestanden; relative Assets, Worker, Modellhashes und gemeinsame Schwellwerte geprüft; Browser laufen auf diesem Build |
| Neue Datenprüfung | Keine Bildgruppenüberschneidung, identischen Quellbilder oder pHash-Nahduplikatkandidaten zwischen neuem Test und übrigen Splits |
| Neuer realer Büscheltest | 30 reale Commons-Büschelszenen und 30 COCO-Negativszenen vor Training eingefroren; **6 TP, 1 FP, 90 FN**, Präzision **85,7 %**, Recall **6,3 %** |

Der neue Prüfbestand wurde in Commit `59efe9e` eingefroren und bei der Fortsetzung in Commit `18308d4` unverändert gelassen. Bilder, Annotationen, Lizenzen und Hashes stehen in `individual-benchmark.json`. Die zusätzliche Einzelannotation wurde vom Coding-Assistenten visuell geprüft; eine unabhängige menschliche Gegenprüfung fehlt. Öffentliche Quellen liefern keine physischen Frucht-/Aufnahmeserien-IDs. Das alte fehlgeschlagene Experiment wurde ausdrücklich Training/Validierung und ist unter `experiment-v1/` archiviert.

Die Empfehlungsmetrik nutzt den echten TypeScript-Tracker mit drei identischen Ergebnissen je statischem Foto. Unsichere Ergebnisse zählen als ausgelassene Empfehlungen. Auf den 30 Negativfotos entstanden 0 falsche Empfehlungen. Zu große Gesamtrahmen ohne passende Einzelbox zählen nicht als richtige Erkennung. Nach Sichtung des neuen Tests erfolgte kein Nachjustieren.

Browser-Kameratests verwenden kontrollierte Video-/Worker-Doubles. Chromium verwendet einen lokalen Canvas-Medienstream; WebKit simuliert Medienquelle und Trackzustände. Das zusätzliche Beispiel verwendet unveränderte Bildpixel und den echten Worker. Modellhash und Browserversion werden vom Lauf erfasst und in `device-validation.json` dokumentiert. Ein künstlich um 13 Sekunden verzögerter WASM-Download prüft, dass Netzwerklatenz keinen unnötigen GPU-Neustart auslöst. Kein solcher Test belegt den physischen Kameraindikator, echte Smartphone-Leistung oder echten Kamera-Netzwerkverkehr.

## Modellabnahme

Ziel: ≥95 % Präzision und ≥80 % Recall für bestätigte Empfehlungen sichtbarer reifer Einzelbananen bei IoU ≥0,5; Enthaltungen zählen als ausgelassene Empfehlungen. Die Zielwerte für essreife Einzelbananen sind nicht erreicht. Der Bericht enthält die tatsächlich gemessenen Werte. Nach weiterem testgetriebenem Tuning muss ein neuer Holdout vor Training festgelegt werden; der jetzige Test ist nun eingesehen.

## Noch erforderlich: echte Geräte

Je ein physisches Android-Handy mit Chrome und ein physisches iPhone mit Safari. Gerät, OS, Browserversion, Modellhash, Datum und Testperson dokumentieren. Den zu prüfenden Build unter HTTPS oder über eine lokale USB-/Portweiterleitung öffnen und den Modellhash kontrollieren. Die öffentliche Altversion zählt nicht als Prüfung der neuen Gewichte. Emulation und Node-Zahlen zählen nicht als Gerätetest.

1. Kamera verweigern und erneut freigeben. Rückkamera starten, während einer offenen Freigabeanfrage stoppen. Nach dem Stoppen muss der System-Kameraindikator erlöschen; verspätet freigegebene Tracks dürfen nicht weiterlaufen.
2. Mehrere einzelne Früchte und überlappende Büschel filmen. Leicht bewegen, Konfidenzreihenfolge wechseln, eine Frucht entfernen. Nummern müssen eindeutig bleiben; Empfehlung bei fehlender oder unsicherer Frucht sofort nach dem nächsten Ergebnis entfallen. Nach Neustart Nummern ab 1. Nummer antippen und Zuordnung im Bild prüfen.
3. Drei aufeinanderfolgende sichere Reifebewertungen abwarten. Nur dann grüner Rahmen und „✓ Jetzt essreif“. Bei „Unreif“, „Überreif“ und „Nicht sicher beurteilbar“ kein Häkchen. Hinweise bei mangelhafter Erkennung prüfen.
4. Mehrfach zwischen Hoch-/Querformat wechseln, Früchte an Bildränder bewegen, Vollbild öffnen/schließen. Rahmen müssen Bildkoordinaten folgen; Liste und Rahmen müssen dieselben Nummern zeigen.
5. Hintergrundwechsel, Sperren und Unterbrechung der Kamera prüfen. Stoppen und erneuter Start müssen funktionieren.
6. Nach zehn Sekunden Aufwärmen mindestens 60 Sekunden messen. **Mindestens drei vollständige Auswertungen/s**, einschließlich Aufnahme, Vorverarbeitung, Worker, Nachverarbeitung und Darstellung. Inferenzmedian/p95, Framealter, Provider und Browser-Messbericht sichern. WebGPU und WASM getrennt dokumentieren, soweit verfügbar.
7. Netzwerk per Chrome Remote Debugging bzw. Safari Web Inspector prüfen: nach statischen Downloads dürfen keine Kamerapixel in Fetch/XHR/WebSocket oder anderen Requests vorkommen. Lokale Speicher auf Kameraaufnahmen prüfen. Keine Videoaufzeichnung in die App einbauen.

Ergebnisse in `device-validation.json` erfassen. Nur tatsächlich geprüfte physische Geräte als `physicalDevice: true` eintragen. Der Veröffentlichungsworkflow prüft die dort belegten Mindestbedingungen und die Modellmetriken. Er lässt diese aktuell nicht erfüllte Abnahme nicht als erfolgreiche Veröffentlichung durchlaufen.
