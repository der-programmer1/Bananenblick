"""Freeze the individual-fruit benchmark BEFORE training; never read predictions here.

Run prepare_data.py first. COCO annotation JSON files must be downloaded from
https://images.cocodataset.org/annotations/annotations_trainval2017.zip.
Photographs are fetched from their recorded official COCO URLs, with license and
Flickr source links retained. No source images are committed by this script.
"""
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures, hashlib, json, random, re, shutil, urllib.request
from PIL import Image

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'work'
OUT = WORK / 'individual'
ANNOTATIONS = ROOT / 'individual_annotations.json'
FROZEN = ROOT / 'individual-benchmark.json'

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    labels = json.loads(ANNOTATIONS.read_text())
    train_candidates = json.loads((WORK / 'coco-train-candidates.json').read_text())
    # Candidate index ordering is also persisted in the benchmark for reproduction.
    datasets = {s: json.loads((WORK / f'coco-{s}2017.json').read_text()) for s in ['train', 'val']}
    maps = {s: {i['id']: i for i in d['images']} for s, d in datasets.items()}
    licenses = {l['id']: l for l in datasets['val']['licenses']}
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(WORK/'data/images', OUT/'images', dirs_exist_ok=True)
    splits = {s: json.loads((WORK / f'data/{s}.json').read_text()) for s in ['train', 'val', 'test']}
    # One original image contains exactly duplicated labels for the same fruit.
    for records in splits.values():
        for r in records:
            r['boxes'] = [list(b) for b in dict.fromkeys(tuple(b) for b in r['boxes'])]
    supplemental = {'train': [], 'val': [], 'acceptance': []}
    sources = []

    def record(source_set, image_id, boxes, split, negative=False):
        image = maps[source_set][image_id]
        folder = WORK / f'coco/{source_set}2017'
        folder.mkdir(parents=True, exist_ok=True)
        original = folder / image['file_name']
        if not original.exists():
            urllib.request.urlretrieve(image['coco_url'].replace('http:', 'https:'), original)
        im = Image.open(original).convert('RGB'); im.thumbnail((640, 640))
        name = f'coco-{source_set}-{image_id:012}.jpg'
        im.save(OUT/'images'/name, quality=94)
        flickr = re.search(r'/([0-9]+)_', image.get('flickr_url', ''))
        r = {'id': f'coco-{source_set}-{image_id}', 'file': 'images/'+name,
             'source': image['coco_url'].replace('http:', 'https:'), 'width': im.width, 'height': im.height,
             'boxes': [[c, (x1+x2)/200, (y1+y2)/200, (x2-x1)/100, (y2-y1)/100] for c,x1,y1,x2,y2 in boxes],
             'group': f'coco-{image_id}', 'sha256_file': digest(OUT/'images'/name),
             'originalSha256': digest(original), 'license': licenses[image['license']],
             'flickrSource': f'https://www.flickr.com/photo.gne?id={flickr[1]}' if flickr else image.get('flickr_url'),
             'scene': 'no-banana' if negative else 'multiple-real-fruits', 'sourceSet': source_set}
        supplemental[split].append(r); sources.append(r)

    for label_key, split, source_set, indexed in [
        ('testTrainIndices', 'acceptance', 'train', True), ('testValIds', 'acceptance', 'val', False),
        ('trainTrainIndices', 'train', 'train', True), ('trainValIds', 'train', 'val', False),
        ('valTrainIndices', 'val', 'train', True),
    ]:
        for key, boxes in labels[label_key].items():
            image_id = train_candidates[int(key)]['id'] if indexed else int(key)
            record(source_set, image_id, boxes, split)
    positive_ids = {r['id'] for r in sources}
    for source_set, count, split in [('train', 20, 'train'), ('val', 5, 'val'), ('val', 30, 'acceptance')]:
        ds = datasets[source_set]
        banana_ids = {a['image_id'] for a in ds['annotations'] if a['category_id'] == 52}
        food_ids = {a['image_id'] for a in ds['annotations'] if a['category_id'] in [47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60]}
        candidates = [i for i in ds['images'] if i['id'] not in banana_ids and i['id'] in food_ids and i['license'] in [4,5]
                      and f"coco-{source_set}-{i['id']}" not in positive_ids]
        random.Random(20261110 + len(sources)).shuffle(candidates)
        for image in candidates[:count]:
            record(source_set, image['id'], [], split, True)
            positive_ids.add(f"coco-{source_set}-{image['id']}")
    for split, records in supplemental.items():
        if split != 'acceptance': splits[split].extend(records)
        else: splits[split] = records
    memberships = [{r['group'] for r in rs} for rs in supplemental.values()]
    assert not any(memberships[i] & memberships[j] for i in range(3) for j in range(i))
    for split, records in splits.items():
        (OUT/f'{split}.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    shutil.copyfile(WORK/'data/audit.json', OUT/'audit.json')
    frozen = {
        'version': 1, 'frozenAt': datetime.now(timezone.utc).isoformat(),
        'annotationSha256': digest(ANNOTATIONS), 'trainingStarted': False,
        'positiveScenes': sum(r['scene'] != 'no-banana' for r in supplemental['acceptance']),
        'negativeScenes': sum(r['scene'] == 'no-banana' for r in supplemental['acceptance']),
        'criteria': {'precision': .95, 'recall': .8, 'iou': .5, 'consecutiveConfirmations': 3,
                     'uncertainCountsAsMissedRecommendation': True},
        'limitations': [
            'COCO train photographs were exposed to the original COCO pretrained detector; this is a holdout from banana fine tuning, not from generic COCO pretraining.',
            'Single coding-assistant annotation review; independent human ripeness/box adjudication is still required.',
            'Static real photographs cannot establish live-camera temporal or device performance.'
        ],
        'candidateOrdering': [i['id'] for i in train_candidates],
        'splits': supplemental,
    }
    assert frozen['positiveScenes'] >= 30 and frozen['negativeScenes'] >= 30
    if FROZEN.exists():
        previous = json.loads(FROZEN.read_text())
        assert previous['annotationSha256'] == frozen['annotationSha256'], 'Frozen labels changed; use a new benchmark version.'
        assert previous['splits'] == frozen['splits'], 'Frozen membership/images changed.'
    else: FROZEN.write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    print(json.dumps({s: len(rs) for s, rs in splits.items()}), flush=True)

if __name__ == '__main__': main()
