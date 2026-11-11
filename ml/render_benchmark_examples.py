"""Draw explicitly labelled report figures from cached predictions, not camera screenshots."""
from pathlib import Path
import json
import textwrap
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

ROOT=Path(__file__).resolve().parent
SITE=ROOT.parent
FOLDER=SITE/'public/evaluation/experiment-v4'


def main():
    figures=json.loads((ROOT/'work/training-v4/example-frames.json').read_text())
    report=json.loads((FOLDER/'individual-report.json').read_text())
    font=lambda size:ImageFont.truetype('DejaVuSans.ttf',size)
    page=['# Beispiele des Trainingsversuchs 0.4','',
        'Diese Abbildungen zeigen gespeicherte Ausgaben des Kandidaten auf drei vor der Testinferenz festgelegten Fotos. '
        'Sie sind keine Screenshots der öffentlichen App und keine Geräteprüfung. Die Nummern, drei Bestätigungen, '
        'Statuszeilen und Nummernpositionen verwenden die gleichen Funktionen wie die App. Es werden gespeicherte '
        'Detektionen dreimal ausgewertet; das Modell wird hier nicht erneut ausgeführt oder nachjustiert.','',
        f"Gesamter Prüfbestand: **{report['truePositive']} TP / {report['falsePositive']} FP / {report['falseNegative']} FN**. "
        'Die Beispiele ersetzen diese vollständige Auswertung nicht.','',f"Modell-SHA-256: `{report['modelSha256']}`.",'']
    for number,figure in enumerate(figures,1):
        record=figure['record'];rect=figure['rect'];tracks=figure['tracks']
        summary=textwrap.wrap(figure['summary'],width=78)
        height=58+rect['height']+32+len(summary)*24+max(1,len(tracks))*26+100
        canvas=Image.new('RGB',(800,height),'#f7f7f2');draw=ImageDraw.Draw(canvas)
        draw.text((20,12),f"Kandidat 0.4 · gespeicherte Modellausgabe · Beispiel {number}",font=font(18),fill='#1c2827')
        photo=Image.open(ROOT/'work/individual-v4'/record['file']).convert('RGB').resize((rect['width'],rect['height']))
        canvas.paste(photo,(20,58))
        by_id={label['id']:label for label in figure['labels']}
        for track in tracks:
            color='#16874b' if track['readyToEat'] else '#687679' if track['ripeness'] in ['unsicher','reif'] else '#bd8841'
            box=track['box'];x=20+box['x']*rect['width'];y=58+box['y']*rect['height']
            draw.rectangle((x,y,x+box['width']*rect['width'],y+box['height']*rect['height']),outline=color,width=3 if track['readyToEat'] else 2)
            label=by_id[track['id']];lx,ly=20+label['x'],58+label['y']
            draw.line((lx+label['width']/2,ly+label['height']/2,20+label['anchorX'],58+label['anchorY']),fill=color,width=1)
            draw.rectangle((lx,ly,lx+label['width'],ly+label['height']),fill=color)
            draw.text((lx+4,ly+2),str(track['id'])+(' ✓' if track['readyToEat'] else ''),font=font(13),fill='white')
        y=58+rect['height']+20
        for line in summary:
            draw.text((20,y),line,font=font(17),fill='#1c2827');y+=24
        for track in tracks:
            draw.text((20,y),f"Nr. {track['id']}: {track['status']}",font=font(16),fill='#1c2827');y+=26
        if not tracks:
            draw.text((20,y),'Keine aktuellen Treffer.',font=font(16),fill='#687679');y+=26
        metadata=record['sourceMetadata']
        draw.text((20,y+10),'Statisches Foto; drei identische gespeicherte Ergebnisse. Kein Kameratest.',font=font(12),fill='#687679')
        credit=f"Foto: {metadata['artist']} · {metadata['license']} · Commons-ID {record['commonsId']}"
        for i,line in enumerate(textwrap.wrap(credit,width=105)[:2]):
            draw.text((20,y+30+i*16),line,font=font(12),fill='#687679')
        png_info=PngImagePlugin.PngInfo()
        for key,value in {'Artist':metadata['artist'],'Source':metadata['sourceUrl'],'License':metadata['license'],
            'LicenseURL':metadata['licenseUrl'],'Changes':'Cropped, scaled and annotated with cached model results.',
            'ModelSHA256':report['modelSha256']}.items():png_info.add_text(key,str(value))
        filename=f'example-{number}.png';canvas.save(FOLDER/filename,pnginfo=png_info)
        page.extend([f"## {number}. {metadata['title'].removeprefix('File:')}",'',f"![Gespeicherte Modellausgabe mit Nummern und Ergebnisliste]({filename})",'',
            f"Foto: {metadata['artist']}, [{metadata['title']}]({metadata['sourceUrl']}), "
            f"[{metadata['license']}]({metadata['licenseUrl']}). Ausschnitt `{record['cropPercent']}` Prozent; "
            'verkleinert, Rahmen und Texte ergänzt. Die Abbildung verwendet dieselbe Bildlizenz.',''])
    (FOLDER/'examples.md').write_text('\n'.join(page))
    print('Three report figures generated from saved detections; no new inference.')


if __name__=='__main__':main()
