"""Check source groups and perceptual near-duplicates without running a model."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageOps
from prepare_data import phash

ROOT=Path(__file__).resolve().parent;WORK=ROOT/'work';DATA=WORK/'individual-v3'

def signature(record):
    path=WORK/record['sourceMetadata']['file'] if 'sourceMetadata' in record else DATA/record['file']
    im=ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    gray=np.asarray(im.convert('L').resize((32,32)))
    hashes=[phash(np.fliplr(np.rot90(gray,k)) if flip else np.rot90(gray,k)) for k in range(4) for flip in [False,True]]
    return {'id':record['id'],'group':record['group'],'sourcePixelSha256':hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest(),'hashes':hashes}

def main():
    sets={s:json.loads((DATA/f'{s}.json').read_text()) for s in ['train','val','test','acceptance']}
    groups={s:{r['group'] for r in rs} for s,rs in sets.items()};group_overlaps=[]
    for a,b in [('train','val'),('train','test'),('train','acceptance'),('val','acceptance')]:
        group_overlaps.extend({'left':a,'right':b,'group':g} for g in groups[a]&groups[b])
    signatures={s:[signature(r) for r in rs] for s,rs in sets.items()};near=[];exact=[]
    for right in ['train','val','test']:
        for a in signatures['acceptance']:
            for b in signatures[right]:
                if a['sourcePixelSha256']==b['sourcePixelSha256']:exact.append({'acceptance':a['id'],'otherSplit':right,'other':b['id']})
                distance=min((x^y).bit_count() for x in a['hashes'] for y in b['hashes'])
                if distance<=6:near.append({'acceptance':a['id'],'otherSplit':right,'other':b['id'],'distance':distance})
    report={'method':'Source group IDs, exact displayed source pixel hashes, 63-bit DCT pHash over eight rotations/reflections; near-duplicate threshold <=6. Training/validation Commons crops are compared using their uncropped source.',
        'scope':'No model predictions. Public sources do not supply physical-fruit or camera-session IDs; this audit cannot establish independent fruit populations.',
        'counts':{s:len(rs) for s,rs in sets.items()},'groupOverlaps':group_overlaps,'exactSourceDuplicates':exact,'nearDuplicateCandidates':near,
        'passedAutomaticChecks':not group_overlaps and not exact and not near}
    target=ROOT.parent/'public/evaluation/individual-split-audit.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    assert not group_overlaps and not exact,'Actual source overlap invalidates the held-out claim.'

if __name__=='__main__':main()
