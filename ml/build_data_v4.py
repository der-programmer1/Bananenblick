"""Freeze the next real-photo holdout before training; preserve all previous results."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import random
import shutil
import urllib.request
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
DATA = WORK / "individual-v4"
FROZEN = ROOT / "individual-benchmark-v4.json"


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def source_photo(metadata, max_side):
    cached = WORK / metadata["file"]
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(metadata["imageUrl"], headers={
            "User-Agent": "Bananenblick-Research/0.4 (https://github.com/der-programmer1/Bananenblick)",
        })
        with urllib.request.urlopen(request, timeout=40) as response:
            raw = response.read()
        assert hashlib.sha256(raw).hexdigest() == metadata["sourceSha256"]
        photo = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        photo.thumbnail((max_side, max_side))
        photo.save(cached, quality=94)
    assert sha(cached) == metadata["sha256"], f"Source photo changed: {cached}"
    return Image.open(cached).convert("RGB")


def main():
    DATA.mkdir(exist_ok=True)
    (DATA / "images").mkdir(exist_ok=True)
    base = WORK / "individual-v3"
    shutil.copytree(base / "images", DATA / "images", dirs_exist_ok=True)
    sets = {s: json.loads((base / f"{s}.json").read_text()) for s in ["train", "val", "test"]}
    seen = json.loads((base / "acceptance.json").read_text())
    # The already inspected v2 benchmark now helps validation. It is never
    # presented as an independent test for a model developed after its inspection.
    sets["val"] += seen
    sets["acceptance"] = []
    labels = json.loads((ROOT / "individual_annotations_v4.json").read_text())
    extra_training = json.loads((ROOT / "individual_training_v4.json").read_text())["photos"]
    assert len(labels["test"]) == 30
    assert not set(labels["test"]) & set(extra_training)
    assert not set(map(str, labels["excludeRelatedFromOtherSplits"])) & set(extra_training)
    if FROZEN.exists():
        original = json.loads(FROZEN.read_text())
        sources = {str(r["commonsId"]): r["sourceMetadata"] for rows in original["supplemental"].values()
                   for r in rows if "commonsId" in r}
    else:
        sources = {str(r["id"]): r for name in ["commons-records.json", "commons-v4-records.json"]
                   for r in json.loads((WORK / name).read_text())}
    supplemental = {"train": [], "acceptance": []}
    for split, entries in [("train", extra_training), ("acceptance", labels["test"])]:
        for key, entry in entries.items():
            source = sources[key]
            max_side = 800 if source["file"].startswith("commons-v4/") else 640
            photo = source_photo(source, max_side)
            width, height = photo.size
            crop = entry["crop"]
            photo = photo.crop(tuple(round(v * d / 100) for v, d in zip(crop, [width, height, width, height])))
            filename = f"v4-commons-{key}.jpg"
            photo.save(DATA / "images" / filename, quality=94)
            boxes = entry["boxes"]
            assert all(0 <= c <= 3 and 0 <= x < X <= 100 and 0 <= y < Y <= 100 for c, x, y, X, Y in boxes)
            record = {
                "id": "v4-commons-" + key, "commonsId": int(key), "group": "commons-" + key,
                "file": "images/" + filename, "width": photo.width, "height": photo.height,
                "boxes": [[c, (x+X)/200, (y+Y)/200, (X-x)/100, (Y-y)/100] for c, x, y, X, Y in boxes],
                "scene": "real-bunch" if len(boxes) > 1 else "single-fruit", "cropPercent": crop,
                "sourceMetadata": source, "sourceNormalizationMaxSide": max_side,
                "sha256_file": sha(DATA / "images" / filename),
            }
            sets[split].append(record)
            supplemental[split].append(record)
    coco = json.loads((WORK / "coco-val2017.json").read_text())
    banana_ids = {r["image_id"] for r in coco["annotations"] if r["category_id"] == 52}
    food_ids = {r["image_id"] for r in coco["annotations"] if r["category_id"] in [47,49,50,51,53,54,55,56,57,58,59,60]}
    used_groups = {r["group"] for rows in sets.values() for r in rows}
    candidates = [r for r in coco["images"] if r["id"] in food_ids - banana_ids
                  and f"coco-{r['id']}" not in used_groups and r["license"] in [4,5]]
    random.Random(20261114).shuffle(candidates)
    licenses = {r["id"]: r for r in coco["licenses"]}
    for item in candidates[:30]:
        original_path = WORK / "coco/val2017" / item["file_name"]
        if not original_path.exists():
            original_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(item["coco_url"].replace("http:", "https:"), original_path)
        photo = Image.open(original_path).convert("RGB")
        photo.thumbnail((800,800))
        filename = "v4-negative-" + item["file_name"]
        photo.save(DATA / "images" / filename, quality=94)
        record = {
            "id": f"v4-negative-{item['id']}", "group": f"coco-{item['id']}", "file": "images/" + filename,
            "width": photo.width, "height": photo.height, "boxes": [], "scene": "no-banana",
            "source": item["coco_url"].replace("http:", "https:"), "flickrSource": item["flickr_url"],
            "license": licenses[item["license"]], "originalSha256": sha(original_path),
            "sha256_file": sha(DATA / "images" / filename),
        }
        sets["acceptance"].append(record)
        supplemental["acceptance"].append(record)
    groups = {s: {r["group"] for r in rows} for s, rows in sets.items()}
    for left, right in [("train","val"),("train","acceptance"),("val","acceptance"),("test","acceptance")]:
        assert not groups[left] & groups[right], (left, right)
    assert len(sets["acceptance"]) == 60
    assert sum(r["scene"] == "real-bunch" for r in sets["acceptance"]) == 30
    for split, rows in sets.items():
        (DATA / f"{split}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    initial=ROOT/"artifacts/banana-individual-v4-initial.pt"
    if not initial.exists():
        shutil.copyfile(ROOT/"artifacts/banana-yolox-nano.pt",initial)
    protocol = {
        "inputSize":448, "maxEpochs":24, "minimumEpochs":10, "earlyStoppingPatience":7,
        "learningRate":.00015, "seed":20261114, "fullBackboneFromEpoch":1,
        "realPositiveSamplingRepeat":8, "syntheticScenesPerEpoch":150, "plainBackgroundProbability":.3,
        "realBunchCropProbability":.5, "cropMinScale":.55, "minVisibleBoxFraction":.25,
        "initialCheckpoint":"artifacts/banana-individual-v4-initial.pt",
        "initialCheckpointSha256":sha(initial),
        "modelSelection":"highest confirmed-ripe validation F1 at fixed .25/.6/.15/.65 thresholds; no holdout inference before checkpoint/threshold selection",
        "calibration":"validation only, >=95% precision then highest recall; otherwise highest F1 and explicit failure",
        "requiredConfirmations":3, "precisionTarget":.95, "recallTarget":.8, "iou":.5,
    }
    manifest = {
        "version":4, "frozenAt":datetime.now(timezone.utc).isoformat(),
        "realBunchScenes":30, "negativeScenes":30,
        "annotationSha256":sha(ROOT/"individual_annotations_v4.json"),
        "trainingAnnotationSha256":sha(ROOT/"individual_training_v4.json"),
        "counts":{s:len(rows) for s,rows in sets.items()},
        "splitSha256":{s:sha(DATA/f"{s}.json") for s in sets},
        "supplemental":supplemental, "protocol":protocol,
        "previouslyInspectedBenchmarkUsedAs":"validation, not an independent test",
        "limits":[
            "Coding-assistant visible-instance annotations; no independent expert adjudication.",
            "Selected real photo crops approximate a closer camera view; this does not establish full-photo or live-camera generalization.",
            "Public source identity and duplicate checks do not guarantee distinct physical fruits in all internet photographs.",
            "COCO negatives were within generic COCO pretraining; Commons positives were not part of that dataset.",
            "No physical Android/Chrome or iPhone/Safari evidence is created by this dataset or training.",
        ],
    }
    if FROZEN.exists():
        assert original["splitSha256"] == manifest["splitSha256"]
        assert original["annotationSha256"] == manifest["annotationSha256"]
        assert original["trainingAnnotationSha256"] == manifest["trainingAnnotationSha256"]
        assert original["protocol"] == protocol
    else:
        FROZEN.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    negatives = [r for r in sets["acceptance"] if r["scene"] == "no-banana"]
    sheet = Image.new("RGB", (1200,1000), "#eee")
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(negatives):
        image = Image.open(DATA/record["file"])
        image.thumbnail((190,175))
        x,y=i%6*200,i//6*200
        sheet.paste(image,(x,y+20))
        draw.text((x+2,y+2),record["id"],fill="black")
    sheet.save(WORK/"v4-negative-review.jpg")
    print(json.dumps(manifest["counts"]),flush=True)


if __name__ == "__main__":
    main()
