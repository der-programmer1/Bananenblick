#!/usr/bin/env python3
"""Download, validate and group-split publicly licensed YOLO banana annotations.

All source splits are pooled. Decoded-pixel duplicates are removed; pHash-near
images and filename augmentation families are assigned as one indivisible group.
No augmentation is performed before splitting. Test labels are not used to tune.
"""
from pathlib import Path
import argparse,collections,hashlib,io,json,random,re,urllib.request,zipfile
import numpy as np
from PIL import Image,ImageOps
from scipy.fft import dctn

ROOT=Path(__file__).resolve().parent
URL='https://data.mendeley.com/public-files/datasets/56c75hvm7r/files/e6fed512-91b4-413e-a7ea-7962ecf01a3c/file_downloaded'
SHA='aa635da1a7e8c4e09056a7a72d8625b2facccbc7f9d9e6a8b3b62c0e3712161e'
NAMES=['unreif','reif','überreif']

def normalize_display_image(source):
    return ImageOps.exif_transpose(source).convert('RGB')

def phash(im):
    g=np.asarray(im,dtype=float)
    v=dctn(g,type=2,norm='ortho')[:8,:8].flatten()[1:]
    return int(''.join('1' if x>np.median(v) else '0' for x in v),2)

def family(stem):
    s=re.sub(r'\.(?:jpg|png|jpeg)[._-]rf[._-].*','',stem,flags=re.I)
    s=re.sub(r'(?:_aug(?:mented)?|_flip|_rotate|_rot|_brightness|_zoom)[_-]?\d*.*','',s,flags=re.I)
    return s

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=ROOT/'work/BANANA_DATASET.zip');p.add_argument('--output',type=Path,default=ROOT/'work/data');p.add_argument('--seed',type=int,default=20261109);a=p.parse_args()
    a.archive.parent.mkdir(parents=True,exist_ok=True)
    if not a.archive.exists():
        req=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=120) as src,a.archive.open('wb') as out:
            while chunk:=src.read(1024*1024):out.write(chunk)
    archive_hash=hashlib.file_digest(a.archive.open('rb'),'sha256').hexdigest()
    if archive_hash!=SHA:raise ValueError(f'Archive checksum mismatch: {archive_hash}')
    out=a.output; (out/'images').mkdir(parents=True,exist_ok=True)
    z=zipfile.ZipFile(a.archive);allnames=z.namelist()
    yaml={n:z.read(n).decode() for n in allnames if n.endswith(('.yaml','.yml'))}
    print('Dataset configuration:',json.dumps(yaml),flush=True)
    # Verify the actual archive label mapping. Publisher's description may differ.
    mapping=None
    for content in yaml.values():
        m=re.search(r'names:\s*\[([^\]]+)\]',content)
        if m:
            labels=[x.strip().strip("'\"").lower() for x in m.group(1).split(',')]
            aliases={'unripe':0,'ripe':1,'overripe':2,'unriped':0,'riped':1,'over-riped':2,'over ripe':2}
            if all(x in aliases for x in labels):mapping=[aliases[x] for x in labels]
    if mapping is None:
        # This published archive has no YAML. Its DOI page explicitly documents
        # 0=Unripe, 1=Ripe, 2=Overripe. Cross-check filename/category majorities.
        mapping=[0,1,2]
        evidence=collections.defaultdict(collections.Counter)
        for n in allnames:
            if n.endswith('.txt'):
                prefix=Path(n).stem.split('_')[0].lower()
                for line in z.read(n).decode().splitlines():
                    if line.strip():evidence[prefix][int(line.split()[0])]+=1
        expected={'unripe':0,'ripe':1,'overripe':2}
        for name,cls in expected.items():
            if name not in evidence or evidence[name].most_common(1)[0][0]!=cls:raise ValueError('Archive mapping is inconsistent with publisher mapping.')
        yaml={'publisher_mapping': 'DOI 10.17632/56c75hvm7r.1 documents 0 unripe / 1 ripe / 2 overripe; archive contains no YAML.', 'filename_crosscheck': {k:dict(v) for k,v in evidence.items()}}
    image_names=[n for n in allnames if n.lower().endswith(('.jpg','.jpeg','.png','.webp')) and not n.startswith('__MACOSX')]
    labels_by_stem=collections.defaultdict(list)
    for n in allnames:
        if n.lower().endswith('.txt'):labels_by_stem[Path(n).stem].append(n)
    records=[];seen={};exact=[];bad=[];clipped=[];orientations=collections.Counter()
    for num,name in enumerate(sorted(image_names)):
        path=Path(name);label_candidates=labels_by_stem[path.stem]
        preferred=str(path.parent.parent/'labels'/f'{path.stem}.txt')
        label_name=preferred if preferred in label_candidates else label_candidates[0] if len(label_candidates)==1 else None
        if not label_name:bad.append({'image':name,'reason':'missing_or_ambiguous_label'});continue
        with Image.open(io.BytesIO(z.read(name))) as source:
            raw_size=list(source.size);orientation=int(source.getexif().get(274,1));orientations[orientation]+=1
            # YOLO boxes in this dataset refer to the EXIF-oriented display.
            # Normalize pixels first; do not rotate the already display-oriented labels.
            im=normalize_display_image(source);w,h=im.size
            pixels=np.asarray(im);digest=hashlib.sha256(str(im.size).encode()+pixels.tobytes()).hexdigest()
            if digest in seen:exact.append({'duplicate':name,'kept':seen[digest]});continue
            boxes=[]
            for line in z.read(label_name).decode().strip().splitlines():
                parts=line.split()
                if len(parts)!=5:raise ValueError(f'Invalid YOLO label {label_name}: {line}')
                cls=int(parts[0]);box=list(map(float,parts[1:]));
                if cls<0 or cls>=len(mapping) or not all(np.isfinite(box)) or min(box[2:])<=0:raise ValueError(f'Invalid label: {label_name}: {line}')
                cx,cy,bw,bh=box
                x1,y1,x2,y2=np.clip([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2],0,1)
                corrected=[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]
                if max(abs(x-y) for x,y in zip(box,corrected))>1e-6:
                    clipped.append({'label':label_name,'before':box,'after':corrected})
                box=corrected
                if min(box[2:])<=0:raise ValueError(f'Empty box after clipping: {label_name}')
                boxes.append([mapping[cls],*box])
            seen[digest]=name
            gray=np.asarray(im.convert('L').resize((32,32)))
            hashes=[phash(np.fliplr(np.rot90(gray,k)) if flip else np.rot90(gray,k)) for k in range(4) for flip in [False,True]]
            filename=f'{len(records):04d}-{digest[:12]}.jpg'
            im.thumbnail((640,640),Image.Resampling.LANCZOS);im.save(out/'images'/filename,quality=94)
            records.append({'id':len(records),'file':f'images/{filename}','source':name,'sha256_pixels':digest,'source_size':[w,h],'raw_size':raw_size,'exif_orientation':orientation,'width':im.width,'height':im.height,'boxes':boxes,'phash':hashes,'family':family(path.stem)})
        if num%50==0:print('processed',num,'/',len(image_names),flush=True)
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(i)]=find(j)
    near=[]
    for i in range(len(records)):
        for j in range(i):
            distance=min((x^y).bit_count() for x in records[i]['phash'] for y in records[j]['phash'])
            same_family=records[i]['family']==records[j]['family']
            if distance<=6 or same_family:
                union(i,j);near.append({'a':i,'b':j,'hamming':distance,'same_family':same_family})
    groups=collections.defaultdict(list)
    for i in range(len(records)):groups[find(i)].append(i)
    # Group-stratified greedy allocation, balancing image/object class counts.
    rng=random.Random(a.seed);items=list(groups.values());rng.shuffle(items);items.sort(key=len,reverse=True)
    totals=np.zeros(4)
    for r in records:
        totals[0]+=1
        for b in r['boxes']:totals[1+int(b[0])]+=1
    ratios=np.array([.70,.15,.15]);counts=np.zeros((3,4));assigned=[[],[],[]]
    for group in items:
        v=np.zeros(4);v[0]=len(group)
        for i in group:
            for b in records[i]['boxes']:v[1+int(b[0])]+=1
        scores=[]
        for s in range(3):
            candidate=counts.copy();candidate[s]+=v
            target=ratios[:,None]*totals
            scores.append(np.sum(((candidate-target)/np.maximum(target,1))**2))
        s=int(np.argmin(scores));counts[s]+=v;assigned[s].extend(group)
    split_names=['train','val','test'];split_records={}
    for s,ids in zip(split_names,assigned):
        split_records[s]=[dict(records[i],group=find(i)) for i in sorted(ids)]
        (out/f'{s}.json').write_text(json.dumps(split_records[s],ensure_ascii=False,indent=2))
    # Audit every group and all detected near pairs, with no cross-split membership.
    owner={i:s for s,ids in zip(split_names,assigned) for i in ids}
    cross=[x for x in near if owner[x['a']]!=owner[x['b']]]
    assert not cross
    summary={s:{'images':len(rs),'objects_per_class':dict(zip(NAMES,[sum(b[0]==c for r in rs for b in r['boxes']) for c in range(3)])),'negative_images':sum(not r['boxes'] for r in rs),'groups':len({r['group'] for r in rs})} for s,rs in split_records.items()}
    audit={'seed':a.seed,'archive_sha256':archive_hash,'class_mapping_source':yaml,'class_mapping_to_internal':mapping,'source_images':len(image_names),'unique_decoded_images':len(records),'exact_duplicates_removed':exact,'near_duplicate_pairs':near,'group_count':len(groups),'cross_split_near_duplicate_pairs':cross,'rejected':bad,'boundary_clipping':clipped,'exif_orientations':dict(orientations),'orientation_handling':'ImageOps.exif_transpose before hashing/resizing; annotations already use display-oriented coordinates; verified visually with three stages.','splits':summary,'limitations':['pHash <=6 plus matching augmentation filenames detects many, but not all, near duplicates.','Capture-session/physical-fruit IDs are unavailable: independent fruit-level and location-level generalization is not established.','No source image is synthesized, and test/validation transformations do not feed training.']}
    audit['membership']={s:[{'id':r['id'],'source':r['source'],'group':r['group'],'pixelSha256':r['sha256_pixels'],'phashHex':[format(x,'016x') for x in r['phash']],'exifOrientation':r['exif_orientation']} for r in rs] for s,rs in split_records.items()}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
