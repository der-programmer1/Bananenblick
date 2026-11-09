#!/usr/bin/env python3
"""Fixed diagnostic controls. These are NOT independent real-world benchmarks.

Run only after final checkpoint/export/test selection; do not retrain or tune on
these outputs. Positive controls are derived from held-out images. One external
public-domain photograph, which is never used for training, is a negative check.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,urllib.request
import cv2
import numpy as np
import onnxruntime as ort
from train import prepare,postprocess,metrics,gt_from_target
import torch
ROOT=Path(__file__).resolve().parent;SITE=ROOT.parent
ASTRO_URL='https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/astronaut.png'
ASTRO_DOC='https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut'

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'work/data');p.add_argument('--force',action='store_true');a=p.parse_args()
    dest=SITE/'public/evaluation';out=dest/'stress-assets';out.mkdir(parents=True,exist_ok=True)
    if (dest/'stress-report.json').exists() and not a.force:raise RuntimeError('Stress controls already evaluated; no retuning on these outputs.')
    card=json.loads((SITE/'public/models/model-card.json').read_text());rs=json.loads((a.data/'test.json').read_text());chosen=[next(r for r in rs if any(int(b[0])==c for b in r['boxes'])) for c in range(3)]
    images=[cv2.imread(str(a.data/r['file'])) for r in chosen]
    cases=[]
    def add(name,title,im,boxes,kind,detail,sources):cases.append({'id':name,'title':title,'im':im,'boxes':boxes,'kind':kind,'detail':detail,'sources':sources})
    provenance='https://data.mendeley.com/datasets/56c75hvm7r/1'
    # Fixed, deliberately synthetic side-by-side composition, retaining complete
    # source images. There is no claim that it is an actual banana-counter photo.
    canvas=np.full((640,1200,3),190,np.uint8);boxes=[]
    for i,(im,r) in enumerate(zip(images,chosen)):
        h,w=im.shape[:2];scale=min(380/w,600/h);nw,nh=int(w*scale),int(h*scale);ox=i*400+(400-nw)//2;oy=(640-nh)//2
        canvas[oy:oy+nh,ox:ox+nw]=cv2.resize(im,(nw,nh))
        for c,x,y,bw,bh in r['boxes']:boxes.append([c,(x*w*scale+ox)/1200,(y*h*scale+oy)/640,bw*w*scale/1200,bh*h*scale/640])
    add('mixed-montage','Drei Reifestufen · synthetische Montage',canvas,boxes,'synthetic-derived','Drei vollständige Holdout-Fotos nebeneinander; keine echte Thekenszene.',[r['source'] for r in chosen])
    # Overlay of two image panels creates partial occlusion. Full source boxes
    # are retained; this is a diagnostic control, not an occlusion benchmark.
    canvas=np.full((640,800,3),190,np.uint8);boxes=[]
    for i,(im,r) in enumerate(zip(images[:2],chosen[:2])):
        h,w=im.shape[:2];scale=min(430/w,580/h);nw,nh=int(w*scale),int(h*scale);ox=30+i*240;oy=30
        canvas[oy:oy+nh,ox:ox+nw]=cv2.resize(im,(nw,nh))
        for c,x,y,bw,bh in r['boxes']:boxes.append([c,(x*w*scale+ox)/800,(y*h*scale+oy)/640,bw*w*scale/800,bh*h*scale/640])
    add('overlap-montage','Teilweise Überdeckung · synthetische Montage',canvas,boxes,'synthetic-derived','Zwei überlagerte Fotoflächen; volle ursprüngliche Boxen als diagnostische Ziele. Keine unabhängige reale Überlappungsszene.',[r['source'] for r in chosen[:2]])
    for c,label,factor in [(0,'dunkel',.45),(1,'hell',1.55),(2,'dunkel',.6)]:
        im=np.clip(images[c].astype(float)*factor,0,255).astype(np.uint8)
        add(f'brightness-{c}',f'{["Unreif","Reif","Überreif"][c]} · {label}',im,chosen[c]['boxes'],'synthetic-derived',f'Fester Helligkeitsfaktor {factor}; kein Wechsellicht-Kameratest.',[chosen[c]['source']])
    r=chosen[1];rot=cv2.rotate(images[1],cv2.ROTATE_90_CLOCKWISE);rb=[[c,1-y,x,bh,bw] for c,x,y,bw,bh in r['boxes']]
    add('rotation-90','Um 90° gedrehte Bilddatei',rot,rb,'synthetic-derived','Pixel und Zielboxen rechnerisch um 90° gedreht; kein Handy-Drehungstest.',[r['source']])
    for value,label in [(0,'schwarz'),(114,'neutralgrau'),(255,'weiß')]:add(f'empty-{value}',f'Leeres Kontrollbild · {label}',np.full((480,640,3),value,np.uint8),[],'synthetic-negative','Homogene leere Fläche; ersetzt keine reale Negativszenen-Stichprobe.',[])
    # Public-domain NASA photograph, distributed by scikit-image; source license
    # and original provenance are documented by the official project.
    external=ROOT/'work/astronaut.png';external.parent.mkdir(exist_ok=True,parents=True)
    if not external.exists():
        with urllib.request.urlopen(ASTRO_URL,timeout=60) as src:external.write_bytes(src.read())
    external_hash=hashlib.file_digest(external.open('rb'),'sha256').hexdigest()
    astro=cv2.imread(str(external))
    add('no-banana-photograph','Reales Foto ohne Banane',astro,[],'external-real-negative','NASA-Foto aus scikit-image (Public Domain). Ein einzelnes Motiv außerhalb des Thekenbereichs; kein belastbarer Negativdatensatz.',[ASTRO_URL,ASTRO_DOC])
    options=ort.SessionOptions();options.intra_op_num_threads=3;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(SITE/'public/models/banana-yolox-nano.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    results=[]
    for case in cases:
        im=case.pop('im');normalized,bs=prepare(im,case.pop('boxes'))
        tensor=np.ascontiguousarray(normalized.transpose(2,0,1)[None],dtype=np.float32)
        output=session.run(None,{'images':tensor})[0][0]
        predictions=postprocess(output,threshold=card['thresholds']['detection'],image_shape=(im.shape[1],im.shape[0]))
        target=np.zeros((max(100,len(bs)),5),np.float32);target[:len(bs)]=bs;truth=gt_from_target(torch.from_numpy(target))
        filename=case['id']+'.jpg';cv2.imwrite(str(out/filename),im,[cv2.IMWRITE_JPEG_QUALITY,92])
        result=metrics([predictions],[truth],card['thresholds']['detection'])
        results.append(dict(case,image='/evaluation/stress-assets/'+filename,width=im.shape[1],height=im.shape[0],expectedObjects=len(truth),detections=len(predictions),groundTruth=truth,predictions=predictions,metrics=result))
    report={'title':'Feste diagnostische Stresskontrollen','createdAt':datetime.now(timezone.utc).isoformat(),'modelSha256':card['sha256'],'thresholds':card['thresholds'],'coordinateSpace':'boxes are xyxy in the 320x320 top-left-letterboxed model input, before inverse mapping to source image','scope':'Synthetische, aus dem Holdout abgeleitete Varianten und ein externes Negativfoto. Nicht zum Training oder zur Schwellwertwahl verwendet. Kein unabhängiger Test echter Theken-, Büschel-, Lichtwechsel-, Bewegungs- oder Smartphone-Szenen.','results':results,'externalImage':{'source':ASTRO_URL,'licenseSource':ASTRO_DOC,'credit':'NASA, distributed by scikit-image','license':'Public Domain; no known copyright restrictions','sha256':external_hash},'summary':{'cases':len(results),'negativeControls':sum(r['expectedObjects']==0 for r in results),'falseAlarmsOnNegativeControls':sum(r['detections'] for r in results if r['expectedObjects']==0),'syntheticPositiveCases':sum(r['expectedObjects']>0 for r in results)},'limitations':['Die Kontrollen wurden vor Betrachtung der Ausgaben festgelegt. Sie sind klein, nicht repräsentativ und teilweise künstlich zusammengesetzt.','Tatsächliche gemischte Thekenbilder, dicht liegende Büschel und reale Überlappungen bleiben offen.','Es wurde nach diesen Ausgaben nicht nachtrainiert und kein Schwellwert angepasst.']}
    (dest/'stress-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report['summary']))

if __name__=='__main__':main()
