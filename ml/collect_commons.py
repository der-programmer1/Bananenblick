"""Collect licensed real-photo candidates for visual annotation, without model predictions."""
from pathlib import Path
import concurrent.futures,hashlib,html,io,json,re,urllib.request
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parent;WORK=ROOT/'work'
clean=lambda s:html.unescape(re.sub('<[^>]*>','',s or '')).strip()

def main():
    raw=json.loads((WORK/'commons-candidates.json').read_text());folder=WORK/'commons';folder.mkdir(exist_ok=True)
    existing={r['id']:r for r in json.loads((WORK/'commons-records.json').read_text())} if (WORK/'commons-records.json').exists() else {}
    records=[]
    for p in raw['query']['pages'].values():
        info=p.get('imageinfo',[{}])[0];meta=info.get('extmetadata',{});license=clean(meta.get('LicenseShortName',{}).get('value',''))
        if not license or 'NC' in license or 'ND' in license or not any(k in license for k in ['CC BY','CC0','Public domain']):continue
        if p['pageid'] in existing:records.append(existing[p['pageid']]);continue
        records.append({'id':p['pageid'],'title':p['title'],'imageUrl':info.get('thumburl') or info.get('url'),'sourceUrl':info.get('descriptionurl'),
            'license':license,'licenseUrl':meta.get('LicenseUrl',{}).get('value',''),'artist':clean(meta.get('Artist',{}).get('value','')),
            'description':clean(meta.get('ImageDescription',{}).get('value','')),'date':clean(meta.get('DateTimeOriginal',{}).get('value','')),
            'file':f"commons/{p['pageid']}.jpg"})
    def fetch(r):
        if (WORK/r['file']).exists() and 'sha256' in r:return r
        req=urllib.request.Request(r['imageUrl'],headers={'User-Agent':'Bananenblick-Research/0.2 (https://github.com/der-programmer1/Bananenblick)'})
        data=urllib.request.urlopen(req,timeout=40).read();r['sourceSha256']=hashlib.sha256(data).hexdigest()
        im=ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB');im.thumbnail((640,640));im.save(WORK/r['file'],quality=94)
        r.update(width=im.width,height=im.height,sha256=hashlib.sha256((WORK/r['file']).read_bytes()).hexdigest());return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,records))
    (WORK/'commons-records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    for page in range((len(records)+19)//20):
        sheet=Image.new('RGB',(1500,1100),'#eee');draw=ImageDraw.Draw(sheet)
        for k,r in enumerate(records[page*20:(page+1)*20]):
            im=Image.open(WORK/r['file']);im.thumbnail((290,240));x=k%5*300;y=k//5*275
            sheet.paste(im,(x+(300-im.width)//2,y+25));draw.text((x+3,y+3),str(page*20+k)+' id='+str(r['id']),fill='black')
        sheet.save(WORK/f'commons-contact-{page}.jpg')
    print('Licensed candidates:',len(records),flush=True)

if __name__=='__main__':main()
