"""Collect licensed photo candidates for visible-instance annotation, without inference.

This is data collection only. A photograph is not a benchmark member until its
visible individual fruits have been reviewed and its split has been frozen.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import html
import io
import json
import re
import urllib.request
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
OUTPUT = WORK / "commons-v4"


def clean(value):
    return html.unescape(re.sub("<[^>]*>", "", value or "")).strip()


def main():
    OUTPUT.mkdir(exist_ok=True)
    used = set()
    for split in ["train", "val", "test", "acceptance"]:
        records = json.loads((WORK / "individual-v3" / f"{split}.json").read_text())
        used.update(r["commonsId"] for r in records if "commonsId" in r)
    # Related views already identified during the previous photograph review.
    used.update([77250022, 109673183, 109673184])
    pages = {}
    for path in sorted(WORK.glob("commons-v4-candidates-*.json")):
        data = json.loads(path.read_text())
        pages.update(data.get("query", {}).get("pages", {}))
    candidates = []
    for page in sorted(pages.values(), key=lambda p: p["pageid"]):
        if page["pageid"] in used:
            continue
        info = page.get("imageinfo", [{}])[0]
        meta = info.get("extmetadata", {})
        license_name = clean(meta.get("LicenseShortName", {}).get("value", ""))
        if (not license_name or "NC" in license_name or "ND" in license_name
                or not any(k in license_name for k in ["CC BY", "CC0", "Public domain"])):
            continue
        candidates.append({
            "id": page["pageid"], "title": page["title"],
            "imageUrl": info.get("thumburl") or info.get("url"),
            "sourceUrl": info.get("descriptionurl"), "license": license_name,
            "licenseUrl": meta.get("LicenseUrl", {}).get("value", ""),
            "artist": clean(meta.get("Artist", {}).get("value", "")),
            "description": clean(meta.get("ImageDescription", {}).get("value", "")),
            "date": clean(meta.get("DateTimeOriginal", {}).get("value", "")),
            "file": f"commons-v4/{page['pageid']}.jpg",
        })

    def fetch(record):
        path = WORK / record["file"]
        cached_metadata = path.with_suffix(".json")
        if path.exists() and cached_metadata.exists():
            cached = json.loads(cached_metadata.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() == cached["sha256"]:
                return cached
        request = urllib.request.Request(record["imageUrl"], headers={
            "User-Agent": "Bananenblick-Research/0.4 (https://github.com/der-programmer1/Bananenblick)",
        })
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                raw = response.read(12 * 1024 * 1024 + 1)
            if len(raw) > 12 * 1024 * 1024:
                raise ValueError("Unexpectedly large candidate")
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            image.thumbnail((800, 800))
            image.save(path, quality=94)
            record.update(width=image.width, height=image.height,
                sourceSha256=hashlib.sha256(raw).hexdigest(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            cached_metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            return record
        except Exception as error:
            print("Candidate unavailable", record["id"], type(error).__name__, flush=True)
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = [r for r in pool.map(fetch, candidates) if r]
    (WORK / "commons-v4-records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    for page in range((len(records) + 19) // 20):
        sheet = Image.new("RGB", (1500, 1160), "#eee")
        draw = ImageDraw.Draw(sheet)
        for k, record in enumerate(records[page * 20:(page + 1) * 20]):
            photo = Image.open(WORK / record["file"])
            photo.thumbnail((290, 245))
            x, y = k % 5 * 300, k // 5 * 290
            sheet.paste(photo, (x + (300 - photo.width) // 2, y + 35))
            draw.text((x + 3, y + 3), f"{page*20+k} id={record['id']}", fill="black")
            draw.text((x + 3, y + 18), record["title"][5:43], fill="black")
        sheet.save(WORK / f"commons-v4-contact-{page}.jpg")
    print("New licensed candidates (not yet a benchmark):", len(records), flush=True)


if __name__ == "__main__":
    main()
