"""Prepare a NEW, frozen real-bunch benchmark; the failed v1 test becomes validation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,random,shutil,urllib.request
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parent;WORK=ROOT/'work';DATA=WORK/'individual-v2'
FROZEN=ROOT/'individual-benchmark-v2.json'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    labels=json.loads((ROOT/'individual_annotations_v2.json').read_text())
    train_labels=json.loads((ROOT/'individual_training_v2.json').read_text())['photos']
    if (WORK/'commons-records.json').exists():commons={str(r['id']):r for r in json.loads((WORK/'commons-records.json').read_text())}
    elif FROZEN.exists():commons={str(r['commonsId']):r['sourceMetadata'] for rs in json.loads(FROZEN.read_text())['splits'].values() for r in rs if 'commonsId' in r}
    else:raise ValueError('Collect and visually annotate Commons photos before freezing.')
    DATA.mkdir(exist_ok=True);(DATA/'images').mkdir(exist_ok=True)
    base=WORK/'individual'
    shutil.copytree(base/'images',DATA/'images',dirs_exist_ok=True)
    sets={s:json.loads((base/f'{s}.json').read_text()) for s in ['train','val','test']}
    old=json.loads((base/'acceptance.json').read_text())
    # The already inspected failed test is expressly NOT reused as a new test.
    sets['train'] += [r for r in old if r['scene']=='no-banana']
    sets['val'] += [r for r in old if r['scene']!='no-banana']
    sets['acceptance']=[];supplemental={'train':[],'acceptance':[]}
    assert not set(train_labels)&set(labels['test'])
    assert not set(map(int,train_labels))&set(labels['excludeRelatedFromTraining'])
    for split,entries in [('train',train_labels),('acceptance',labels['test'])]:
        for page_id,entry in entries.items():
            source=commons[page_id];cached=WORK/source['file']
            if not cached.exists():
                cached.parent.mkdir(parents=True,exist_ok=True)
                request=urllib.request.Request(source['imageUrl'],headers={'User-Agent':'Bananenblick-Research/0.2 (https://github.com/der-programmer1/Bananenblick)'})
                raw=urllib.request.urlopen(request,timeout=60).read()
                assert hashlib.sha256(raw).hexdigest()==source['sourceSha256']
                im=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB');im.thumbnail((640,640));im.save(cached,quality=94)
            assert sha(cached)==source['sha256']
            crop=entry['crop'] if split=='train' else [0,0,100,100]
            boxes=entry['boxes'] if split=='train' else entry
            im=Image.open(cached).convert('RGB');w,h=im.size
            im=im.crop((round(crop[0]*w/100),round(crop[1]*h/100),round(crop[2]*w/100),round(crop[3]*h/100)))
            name=f'commons-{page_id}.jpg';im.save(DATA/'images'/name,quality=94)
            record={'id':'commons-'+page_id,'commonsId':int(page_id),'group':'commons-'+page_id,'file':'images/'+name,
                'width':im.width,'height':im.height,'boxes':[[c,(x1+x2)/200,(y1+y2)/200,(x2-x1)/100,(y2-y1)/100] for c,x1,y1,x2,y2 in boxes],
                'scene':'real-bunch','cropPercent':crop,'sourceMetadata':source,'sha256_file':sha(DATA/'images'/name)}
            assert all(0<=c<=3 and 0<=x1<x2<=100 and 0<=y1<y2<=100 for c,x1,y1,x2,y2 in boxes)
            sets[split].append(record);supplemental[split].append(record)
    coco=json.loads((WORK/'coco-val2017.json').read_text());bananas={a['image_id'] for a in coco['annotations'] if a['category_id']==52}
    food={a['image_id'] for a in coco['annotations'] if a['category_id'] in [47,49,50,51,53,54,55,56,57,58,59,60]}
    old_frozen=json.loads((ROOT/'individual-benchmark.json').read_text())
    used={int(r['id'].split('-')[-1]) for records in old_frozen['splits'].values() for r in records}
    candidates=[i for i in coco['images'] if i['id'] in food and i['id'] not in bananas|used and i['license'] in [4,5]]
    random.Random(20261112).shuffle(candidates);licenses={l['id']:l for l in coco['licenses']}
    for image in candidates[:30]:
        path=DATA/'images'/('negative-v2-'+image['file_name'])
        source=WORK/'coco/val2017'/image['file_name']
        if not source.exists():urllib.request.urlretrieve(image['coco_url'].replace('http:','https:'),source)
        im=Image.open(source).convert('RGB');im.thumbnail((640,640));im.save(path,quality=94)
        r={'id':f"negative-v2-{image['id']}",'group':f"coco-{image['id']}",'file':'images/'+path.name,'width':im.width,'height':im.height,
           'boxes':[],'scene':'no-banana','source':image['coco_url'].replace('http:','https:'),'flickrSource':image['flickr_url'],
           'license':licenses[image['license']],'originalSha256':sha(source),'sha256_file':sha(path)}
        sets['acceptance'].append(r);supplemental['acceptance'].append(r)
    assert len(sets['acceptance'])==60 and sum(r['scene']=='real-bunch' for r in sets['acceptance'])==30
    groups={s:{r['group'] for r in rs} for s,rs in sets.items()}
    for a,b in [('train','val'),('train','acceptance'),('val','acceptance'),('train','test')]:assert not groups[a]&groups[b]
    for split,records in sets.items():(DATA/f'{split}.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    frozen={'version':2,'frozenAt':datetime.now(timezone.utc).isoformat(),'realBunchScenes':30,'negativeScenes':30,
        'annotationSha256':sha(ROOT/'individual_annotations_v2.json'),'trainingAnnotationSha256':sha(ROOT/'individual_training_v2.json'),
        'splits':supplemental,'validationIncludesPreviouslySeenV1Benchmark':True,
        'protocol':{'inputSize':448,'requiredConfirmations':3,'precisionTarget':.95,'recallTarget':.8,'iou':.5,
            'modelSelection':'highest validation confirmed-ripe F1, full backbone training from epoch 1; never select on the v2 test',
            'realPositiveSamplingRepeat':6,'syntheticScenesPerEpoch':600,'maxEpochs':24,'earlyStoppingPatience':6},
        'limits':['Visible coding-assistant annotations, no independent expert adjudication.','Photographs, not live-camera sequences; physical device validation is separate.','Previous failed v1 test is now validation and is not claimed as an independent new result.']}
    if FROZEN.exists():
        previous=json.loads(FROZEN.read_text());assert previous['splits']==frozen['splits'];assert previous['annotationSha256']==frozen['annotationSha256']
        assert previous['trainingAnnotationSha256']==frozen['trainingAnnotationSha256']
    else:FROZEN.write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n')
    # Human-readable negative contact sheet for visual absence check before training.
    sheet=Image.new('RGB',(1200,1000),'#eee');dr=ImageDraw.Draw(sheet)
    for k,r in enumerate(r for r in sets['acceptance'] if r['scene']=='no-banana'):
        im=Image.open(DATA/r['file']);im.thumbnail((190,175));x=k%6*200;y=k//6*200;sheet.paste(im,(x+(200-im.width)//2,y+20));dr.text((x+2,y+1),r['id'],fill='black')
    sheet.save(WORK/'negative-v2-contact.jpg')
    print(json.dumps({s:len(rs) for s,rs in sets.items()}),flush=True)

if __name__=='__main__':main()
