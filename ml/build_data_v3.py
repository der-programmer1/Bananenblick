"""Continue learning from the old, seen v1 cases; preserve the unseen v2 holdout."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,shutil,urllib.request
from PIL import Image

ROOT=Path(__file__).resolve().parent; WORK=ROOT/'work'; BASE=WORK/'individual-v2'; DATA=WORK/'individual-v3'
FROZEN=ROOT/'individual-training-v3.json'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    DATA.mkdir(exist_ok=True);shutil.copytree(BASE/'images',DATA/'images',dirs_exist_ok=True)
    sets={s:json.loads((BASE/f'{s}.json').read_text()) for s in ['train','val','test','acceptance']}
    old={r['id'] for r in json.loads((WORK/'individual/acceptance.json').read_text()) if r['scene']!='no-banana' and all(b[0]<3 for b in r['boxes'])}
    moved=[r for r in sets['val'] if r['id'] in old]
    assert len(moved)==29
    sets['train']+=moved;sets['val']=[r for r in sets['val'] if r['id'] not in old]
    labels=json.loads((ROOT/'validation_annotations_v3.json').read_text())['photos']
    coco=json.loads((WORK/'coco-train2017.json').read_text());images={str(i['id']):i for i in coco['images']};licenses={l['id']:l for l in coco['licenses']}
    original_groups={r['group'] for rs in sets.values() for r in rs}
    supplemental=[]
    for key,entry in labels.items():
        source=images[key];assert f'coco-{key}' not in original_groups
        path=WORK/'coco/train2017'/source['file_name']
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            urllib.request.urlretrieve(source['coco_url'].replace('http:','https:'),path)
        crop=entry['crop'];boxes=entry['boxes'];im=Image.open(path).convert('RGB');w,h=im.size
        im=im.crop((round(w*crop[0]/100),round(h*crop[1]/100),round(w*crop[2]/100),round(h*crop[3]/100)))
        name=f'validation-v3-{key}.jpg';im.save(DATA/'images'/name,quality=94)
        assert all(0<=c<=3 and 0<=x1<x2<=100 and 0<=y1<y2<=100 for c,x1,y1,x2,y2 in boxes)
        r={'id':f'validation-v3-{key}','group':f'coco-{key}','file':'images/'+name,'width':im.width,'height':im.height,
           'boxes':[[c,(x1+x2)/200,(y1+y2)/200,(x2-x1)/100,(y2-y1)/100] for c,x1,y1,x2,y2 in boxes],
           'scene':'real-bunch','cropPercent':crop,'source':source['coco_url'].replace('http:','https:'),'flickrSource':source['flickr_url'],
           'license':licenses[source['license']],'originalSha256':sha(path),'sha256_file':sha(DATA/'images'/name)}
        sets['val'].append(r);supplemental.append(r)
    for split,rs in sets.items():(DATA/f'{split}.json').write_text(json.dumps(rs,ensure_ascii=False,indent=2))
    groups={s:{r['group'] for r in rs} for s,rs in sets.items()}
    for a,b in [('train','val'),('train','acceptance'),('val','acceptance'),('train','test')]:assert not groups[a]&groups[b]
    assert sha(DATA/'acceptance.json')==sha(BASE/'acceptance.json')
    manifest={'version':3,'frozenAt':datetime.now(timezone.utc).isoformat(),'benchmarkSha256':sha(ROOT/'individual-benchmark-v2.json'),
      'newHoldoutUnchanged':True,'annotationSha256':sha(ROOT/'validation_annotations_v3.json'),
      'counts':{s:len(rs) for s,rs in sets.items()},'movedSeenV1CasesToTraining':[r['id'] for r in moved],'newValidation':supplemental,
      'splitSha256':{s:sha(DATA/f'{s}.json') for s in sets},
      'protocol':{'inputSize':448,'maxEpochs':12,'earlyStoppingPatience':5,'learningRate':.0002,
          'modelSelection':'highest validation confirmed-ripe F1; v2 test never used for selection',
          'initialCheckpoint':'best EMA checkpoint of full-backbone pilot selected on its validation F1',
          'realPositiveSamplingRepeat':6,'syntheticScenesPerEpoch':600,'plainBackgroundProbability':.3,
          'fullBackboneFromEpoch':1,'requiredConfirmations':3,'precisionTarget':.95,'recallTarget':.8,'iou':.5},
      'limits':['Old v1 test was already inspected and is explicitly training data here.','Generic COCO pretraining included COCO photos; Commons holdout is separate.','Visible assistant annotation; no independent human adjudication.']}
    if FROZEN.exists():
        old_manifest=json.loads(FROZEN.read_text());assert old_manifest['splitSha256']==manifest['splitSha256'];assert old_manifest['annotationSha256']==manifest['annotationSha256']
    else:FROZEN.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest['counts']))

if __name__=='__main__':main()
