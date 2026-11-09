# Bananenblick – live auf GitHub Pages

**[App öffnen](https://der-programmer1.github.io/Bananenblick/)** · [Erfolgreicher GitHub-Workflow](https://github.com/der-programmer1/Bananenblick/actions/runs/42658377737)

**Die Website ist veröffentlicht.** Dieses Paket enthält die öffentlich nutzbare App, das tatsächlich trainierte ONNX-Modell, den kompletten React-/TypeScript-Quellcode, die CPU-Trainingspipeline, Prüfdaten und den GitHub-Pages-Workflow. Die App braucht keine Anmeldung und keinen API-Schlüssel; Kamerabilder werden auf dem jeweiligen Gerät ausgewertet.

## In drei Schritten veröffentlichen

1. **ZIP entpacken und Dateien hochladen.** Auf GitHub ein neues öffentliches Repository mit dem Hauptbranch **main** erstellen. Den **Inhalt** des entpackten Ordners `bananenblick` in die oberste Ebene des Repositorys übernehmen. Dort müssen anschließend `README.md`, `package.json`, `docs/`, `app/`, `lib/`, `public/` und `.github/workflows/pages.yml` liegen. Die ZIP-Datei selbst hochzuladen startet keine Website. Versteckte Ordner wie `.github` mit übernehmen.
2. **GitHub Pages einstellen.** Im Repository **Settings → Pages → Build and deployment → Source → GitHub Actions** auswählen.
3. **Workflow starten.** Unter **Actions → Bananenblick auf GitHub Pages → Run workflow** den Branch **main** wählen und starten. Wenn alle Schritte grün sind, zeigt der Schritt „Veröffentlichen“ die HTTPS-Adresse. Diese Adresse kann jeder in einem aktuellen Safari- oder Chrome-Browser öffnen und „Kamera starten“ antippen.

Bei späteren Änderungen baut und veröffentlicht jeder Push auf `main` automatisch. Installation, TypeScript-Prüfung, alle neun Tests, Build und Veröffentlichung sind im [GitHub-Actions-Lauf](https://github.com/der-programmer1/Bananenblick/actions/runs/42658377737) erfolgreich abgeschlossen. Die ZIP wurde außerdem frisch entpackt und lokal gebaut; der Workflow ist mit actionlint geprüft. Die Anleitung unten beschreibt das Einrichten einer eigenen Kopie.

Es muss kein Repositoryname im Code eingetragen und das Modell nicht neu trainiert werden. Relative Asset-Pfade unterstützen sowohl `https://NAME.github.io/REPOSITORY/` als auch eine eigene Domain. Der fertige `docs/`-Ordner ist vollständig enthalten.

## Dateien bequem übernehmen

Mit GitHub Desktop: Den entpackten Projektordner über **File → Add local repository** hinzufügen, bei Bedarf **create a repository here** wählen, alle Dateien committen und mit **Publish repository** veröffentlichen. Ein öffentliches Repository auswählen. Anschließend die drei Launch-Schritte oben ab Schritt 2 ausführen.

Alternativ im entpackten Projektordner mit Git arbeiten. Die URL in der vorletzten Zeile durch die Adresse deines neuen, leeren Repositorys ersetzen:

```bash
git init -b main
git add .
git commit -m "Bananenblick GitHub Pages Launch"
git remote add origin https://github.com/DEIN-NAME/DEIN-REPOSITORY.git
git push -u origin main
```

Die Anmeldung erfolgt über die eigene GitHub-Anmeldung. Die ZIP enthält keine Zugangsdaten. Der Workflow verwendet das von GitHub bereitgestellte `GITHUB_TOKEN` mit bereits festgelegten Rechten. Es müssen keine Repository-Secrets angelegt werden.

## Wenn GitHub keine Website zeigt

- In **Settings → Pages** muss die Quelle **GitHub Actions** ausgewählt sein.
- Der Hauptbranch heißt **main**; der Workflow wird auf diesem Branch ausgeführt.
- `package.json` und `.github/workflows/pages.yml` müssen in der obersten Repositoryebene liegen. Ein zusätzlicher Projekt-Unterordner im Repository verhindert den Start.
- Falls der erste automatische Lauf vor dem Aktivieren von Pages gestartet ist, danach über **Run workflow** erneut starten.
- Unter **Actions** zeigt der fehlgeschlagene Schritt die konkrete Meldung. Für Kamera und Module die veröffentlichte HTTPS-Adresse direkt im Browser öffnen.

## Lokal weiterentwickeln

Node.js **24 LTS** und npm:

```bash
npm ci
npm run dev
```

Prüfen und neu bauen:

```bash
npm run check
npm test
npm run build
```

`npm run build` aktualisiert `docs/`. Das Skript stellt die ONNX-Runtime-Dateien aus der fest gepinnten npm-Abhängigkeit wieder her und prüft anschließend Unterpfade, Modell-Hash und erforderliche Assets. Die relative Basis `./` vermeidet eine Anpassung an Benutzer- und Repositorynamen. Der Modell-Worker erhält die Basisadresse explizit von der Seite, damit Modell und WASM auch unter `/DEIN-REPOSITORY/` geladen werden.

Die fertige WASM-Laufzeit ist vollständig in `docs/ort/` enthalten. Die zusätzliche Arbeitskopie `public/ort/` wird beim Entwickeln und Bauen erzeugt und daher nicht doppelt in die ZIP gepackt. Sämtliche nötigen Dateien für die sofortige Veröffentlichung sind bereits in `docs/` vorhanden.

Nicht per Doppelklick als `file://` öffnen: Module und Kamera benötigen einen Webserver. `npm start` bietet eine lokale Vorschau des fertigen Builds an.

## Modell und Teststand

Das mitgelieferte YOLOX-Nano wurde mit **16 CPU-Epochen** trainiert; der Checkpoint wurde ausschließlich auf Validierungsdaten gewählt. Es erkennt drei Reifestufen und kennzeichnet unsichere Vorhersagen. WebGPU wird verwendet, wenn verfügbar; sonst WebAssembly. Frames werden flüchtig verarbeitet, ohne Warteschlange alter Kamerabilder. Beim Stoppen, Hintergrundwechsel und Verlassen der Seite wird die Kamera freigegeben.

Getrennte Daten: **471 Training / 101 Validierung / 101 Test**. Erkannte ähnliche Bilder wurden vor dem Split gruppiert; zwischen Splits gibt es keine erkannten Duplikate. Aufnahmeserien und physische Frucht-IDs sind nicht bekannt und begrenzen die Aussagekraft.

- Test: **95,05 %** gemeinsame Erkennungsquote mit richtiger oberster Reifeklasse; **92,08 %** sicher sichtbar richtig beschriftet; mAP50 **93,95 %**.
- CPU-Referenz: 12,0 ms Median mit ONNX Runtime CPU und drei Threads. Separater ONNX-Runtime-Web-WASM-Lauf unter Node/Linux: 54,3 ms Median. Beides sind **keine Handy-Benchmarks**.
- [Lesbarer Modellbericht](documentation/model-validation.md)
- [Modellkarte](public/models/model-card.json), [Testbericht](public/evaluation/test-report.json), [Stresskontrollen](public/evaluation/stress-report.json), [Web-Laufzeit](public/evaluation/web-runtime.json)
- [Prüfprotokoll und noch offene Geräteabnahme](public/evaluation/acceptance.md)
- [CPU-Training reproduzieren](ml/README.md), finaler Checkpoint in `ml/artifacts/banana-yolox-nano.pt`

Die sichtbare Schalenreife ist keine Aussage über Verderb oder innere Fruchtqualität. Echte Android-/iPhone-Livestreamtests, Geschwindigkeit von mindestens drei vollständigen Auswertungen/s, Rahmenpositionen bei Bewegung und Rotation sowie die Netzwerkprüfung auf echten Geräten sind noch offen. Einzelbananen in dichten Büscheln sind mit diesem Datensatz nicht belastbar abgenommen. Die Browser-Steuerung stand hier nicht zur Verfügung; der Pages-Unterpfad wird durch Build-/URL-Prüfungen überprüft, nicht durch eine behauptete Handyabnahme.

## Öffentlich nutzbar, Bilder lokal

Diese Ausgabe enthält **keine Anmeldung**. Bei normalem GitHub Pages ist die Website öffentlich erreichbar. Die App-Dateien, das Modell und die Beispielbilder können heruntergeladen werden. Kamerabilder bleiben im Arbeitsspeicher des jeweiligen Nutzers. GitHub erhält die üblichen Abrufe der statischen Website-Dateien, keine Kamera-Uploads aus dieser App.

Keine eigenen Fotos als Ersatz in `public/` ablegen, wenn sie nicht öffentlich sein sollen: Alles in `docs/` wird bei Veröffentlichung ausgeliefert. Das ist eine Eigenschaft statischen Hostings, keine Kamerafunktion der App.

## Herkunft und ergänzendes Sites-Buildziel

Die ursprüngliche Oberfläche stammt aus dem offiziellen Sites-Vinext-Starter; die wiederverwendete React-/TypeScript-App liegt unverändert in ihrer Struktur unter `app/` und `lib/`. GitHub Pages kann keinen Cloudflare-Worker ausführen. Deshalb rendert `app/pages-entry.tsx` dieselbe Oberfläche als statische Vite-App. Der GitHub-Pages-Build ist der Standard. Optional bleiben `npm run dev:sites`, `npm run build:sites`, `vite.config.ts` und `app/layout.tsx` für eine eigene Sites-/Vinext-Umgebung erhalten. Die Hosting-Datei ist neutral und enthält **keine fremde Projekt-ID**. Für eine Sites-Publikation muss eine eigene Site eingerichtet werden.

Trainingsdaten: Rohit et al. (2026), [Annotated Banana Ripeness Stage Dataset](https://data.mendeley.com/datasets/56c75hvm7r/1), CC BY 4.0. YOLOX-Architektur: Megvii, Apache 2.0, Lizenz in `ml/vendor/YOLOX-LICENSE`. Weitere Herkunftsangaben stehen in `ml/provenance.json`, Font-Lizenz in `public/fonts/OFL.txt`. Original-Trainingsbilder werden bei Bedarf über das Skript mit Hash-Prüfung heruntergeladen; sie sind wegen der Größe nicht Bestandteil dieser ZIP. Enthalten sind das fertige Modell, die Skripte und die verwendeten Beispielbilder.

Offizielle Hosting-Anleitung: [GitHub Pages – Veröffentlichungsquelle](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site), [GitHub Pages – eigene Workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
