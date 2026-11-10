# Prüfprotokoll · Bananenblick 0.2

**Status: Modellabnahme fehlgeschlagen; keine Veröffentlichung dieses Entwurfs.** Stand: 10. November 2026. Die bisherige GitHub-Pages-Version bleibt bestehen.

## Tatsächlich ausgeführt

| Prüfung | Ergebnis und Grenze |
|---|---|
| TypeScript | Bestanden |
| Automatisierte Tests | 20 bestanden: Überlappung, globale eindeutige Zuordnung, Bewegung, Unsicherheit, drei Bestätigungen, fehlende Treffer, Nummern und Verknüpfung von Liste/Rahmen, Python-/TypeScript-Parität |
| Python-Pipeline | 6 Tests bestanden, einschließlich ursprünglicher Split-Integrität und Metriksemantik |
| Chromium unter Linux, mobile Pixel-7-Ansichtsgröße | Zwei Browserfälle bestanden: kontrollierte Kamera/Worker, Auswahl, Bewegung, Unsicherheit, Verschwinden, Hoch-/Querformat, Stoppen, Nummernreset; zusätzlich echtes ONNX-Beispiel mit drei Durchläufen |
| WebKit unter Linux, mobile iPhone-13-Ansichtsgröße | Dieselben zwei Browserfälle bestanden; WebKit-WPE ist kein echtes iPhone/Safari |
| Tatsächliche physische Geräte | **Keine** Android-/Chrome- oder iPhone-/Safari-Geräte geprüft |
| WASM-Auswertung unter Node/Linux | Echter ONNX-Aufruf, endliche `[1,2100,8]`-Ausgabe und bildabhängige Ergebnisse; vollständige Werte in `web-runtime.json` |
| Statischer Pages-Build | Lokal ausgeführt; relative Assets, Modellhashes und gemeinsame Schwellwerte werden geprüft |
| Neuer Mehrfrucht-Prüfbestand | 30 reale Mehrfruchtszenen und 30 reale Negativszenen vor Training festgelegt; **0 TP, 1 FP, 169 FN**, Zielwerte nicht erreicht |

Die neuen Fotos enthalten Büschel und arrangierte Früchte. Eine unabhängige Abnahme von mindestens 30 eigentlichen Büschelszenen ist noch nicht belegt. Die Einzelboxen wurden vom Coding-Assistenten visuell geprüft; keine unabhängige menschliche Gegenprüfung. COCO-Fotos aus dem Trainingssplit waren im ursprünglichen generischen Vortraining enthalten. Die Prüfung ist daher weder ein unabhängiger Standort-/Frucht-Nachweis noch ein vollständig neuer Test gegenüber dem COCO-Vortraining.

Die Empfehlungsmetrik nutzt den echten TypeScript-Tracker mit drei identischen Ergebnissen je statischem Foto. Unsichere Ergebnisse zählen als ausgelassene Empfehlungen. Es gab eine falsche Empfehlung auf den 30 Negativfotos. Zu große Gesamtrahmen zählen bei fehlender passender Einzelbox nicht als richtige Erkennung.

Browser-Kameratests verwenden Test-Doubles. Der Chromium-Fall verwendet einen lokalen Canvas-Medienstream mit kontrollierter Video-Uhr; im WebKit-Fall werden die Medienquelle und Trackzustände simuliert. Der echte Modelltest verwendet unveränderte Beispielbilder, denselben Worker und dieselbe ONNX-Datei wie die App. Kein solcher Test belegt den physischen Kameraindikator, echte Smartphone-Leistung oder echten Kamera-Netzwerkverkehr.

## Noch erforderliche Modellarbeit

1. Mehr reale, eng annotierte Einzelbananen in Büscheln, gemischten Reifestufen, unterschiedlichen Perspektiven und Beleuchtungen sammeln und unabhängig gegenprüfen. Große Sammelboxen aufteilen; verdeckte oder visuell nicht beurteilbare Früchte ausdrücklich markieren.
2. Nach weiterem Tuning einen neuen Holdout **vor Training** festlegen. Der aktuelle Bestand wurde eingesehen und darf nicht erneut als unberührter Test dienen.
3. Mindestens 30 reale Büschelszenen und 30 reale Negativszenen prüfen. Ziel: ≥95 % Präzision und ≥80 % Recall für tatsächlich bestätigte Empfehlungen sichtbarer reifer Einzelbananen bei IoU ≥0,5; Enthaltungen zählen als ausgelassene Empfehlungen.

## Noch erforderlich: echte Geräte

Je ein physisches Android-Handy mit Chrome und ein physisches iPhone mit Safari. Gerät, OS, Browserversion, Modellhash, Datum und Testperson dokumentieren. Die öffentliche HTTPS-Seite direkt öffnen; Emulation und Node-Zahlen zählen nicht als Gerätetest.

1. Kamera verweigern und erneut freigeben. Rückkamera starten, während einer offenen Freigabeanfrage stoppen. Nach dem Stoppen muss der System-Kameraindikator erlöschen; verspätet freigegebene Tracks dürfen nicht weiterlaufen.
2. Mehrere einzelne Früchte und überlappende Büschel filmen. Leicht bewegen, Konfidenzreihenfolge wechseln, eine Frucht entfernen. Nummern müssen eindeutig bleiben; Empfehlung bei fehlender oder unsicherer Frucht sofort nach dem nächsten Ergebnis entfallen. Nach Neustart Nummern ab 1. Nummer antippen und Zuordnung im Bild prüfen.
3. Drei aufeinanderfolgende sichere Reifebewertungen abwarten. Nur dann grüner Rahmen und „✓ Jetzt essreif“. Bei „Unreif“, „Überreif“ und „Nicht sicher beurteilbar“ kein Häkchen. Hinweise bei mangelhafter Erkennung prüfen.
4. Mehrfach zwischen Hoch-/Querformat wechseln, Früchte an Bildränder bewegen, Vollbild öffnen/schließen. Rahmen müssen Bildkoordinaten folgen; Liste und Rahmen müssen dieselben Nummern zeigen.
5. Hintergrundwechsel, Sperren und Unterbrechung der Kamera prüfen. Stoppen und erneuter Start müssen funktionieren.
6. Nach zehn Sekunden Aufwärmen mindestens 60 Sekunden messen. **Mindestens drei vollständige Auswertungen/s**, einschließlich Aufnahme, Vorverarbeitung, Worker, Nachverarbeitung und Darstellung. Inferenzmedian/p95, Framealter, Provider und Browser-Messbericht sichern. WebGPU und WASM getrennt dokumentieren, soweit verfügbar.
7. Netzwerk per Chrome Remote Debugging bzw. Safari Web Inspector prüfen: nach statischen Downloads dürfen keine Kamerapixel in Fetch/XHR/WebSocket oder anderen Requests vorkommen. Lokale Speicher auf Kameraaufnahmen prüfen. Keine Videoaufzeichnung in die App einbauen.

Ergebnisse in `device-validation.json` erfassen. Nur tatsächlich geprüfte physische Geräte als `physicalDevice: true` eintragen. Der Veröffentlichungsworkflow prüft die dort belegten Mindestbedingungen und die Modellmetriken. Er lässt diese aktuell nicht erfüllte Abnahme nicht als erfolgreiche Veröffentlichung durchlaufen.
