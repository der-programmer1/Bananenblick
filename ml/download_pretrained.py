#!/usr/bin/env python3
"""Fetch the original official COCO checkpoint with a pinned checksum."""
from pathlib import Path
import argparse,hashlib,urllib.request
URL='https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth'
SHA='cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b'
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'work/yolox_nano.pth');a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
if not a.output.exists():
    with urllib.request.urlopen(URL,timeout=120) as src,a.output.open('wb') as dst:
        while chunk:=src.read(1024*1024):dst.write(chunk)
hash=hashlib.file_digest(a.output.open('rb'),'sha256').hexdigest()
if hash!=SHA:raise ValueError(f'Pretrained checkpoint checksum mismatch: {hash}')
print(a.output)
