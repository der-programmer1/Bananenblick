#!/usr/bin/env python3
"""Export the validation-selected checkpoint; evaluate held-out test exactly once.

This script deliberately refuses to overwrite the test report unless --force is
specified. Never tune a model after inspecting test results without a new holdout.
"""
from pathlib import Path
import argparse,hashlib,json,os,platform,shutil,time
from datetime import datetime,timezone
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image,ImageDraw
from torch.nn.utils import fuse_conv_bn_eval
from model import create_model
from train import BananaDataset,DataLoader,metrics,postprocess,gt_from_target,prepare
ROOT=Path(__file__).resolve().parent;SITE=ROOT.parent
CLASSES=['unreif','reif','überreif']

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'work/data');p.add_argument('--training',type=Path,default=ROOT/'work/training');p.add_argument('--force',action='store_true');a=p.parse_args()
    evaluation=SITE/'public/evaluation';models=SITE/'public/models';examples=SITE/'public/examples'
    for d in [evaluation,models,examples]:d.mkdir(parents=True,exist_ok=True)
    if (evaluation/'test-report.json').exists() and not a.force:raise RuntimeError('Test already evaluated; refusing to overwrite. Use a new untouched holdout after tuning.')
    torch.set_num_threads(3);cv2.setNumThreads(1)
    checkpoint=torch.load(a.training/'best.pt',map_location='cpu',weights_only=False)
    selection=json.loads((a.training/'validation-selection.json').read_text());threshold=selection['selectedThreshold']
    model=create_model();model.load_state_dict(checkpoint['model']);model.eval()
    for m in model.modules():
        if hasattr(m,'conv') and hasattr(m,'bn') and hasattr(m,'fuseforward'):
            m.conv=fuse_conv_bn_eval(m.conv,m.bn);delattr(m,'bn');m.forward=m.fuseforward
    sample=torch.zeros((1,3,320,320),dtype=torch.float32)
    path=models/'banana-yolox-nano.onnx'
    torch.onnx.export(model,sample,str(path),input_names=['images'],output_names=['predictions'],opset_version=17,dynamo=False,external_data=False,do_constant_folding=True)
    graph=onnx.load(str(path));onnx.checker.check_model(graph)
    for k,v in {'architecture':'YOLOX-Nano','classes':json.dumps(CLASSES,ensure_ascii=False),'preprocessing':'BGR float32 0..255 CHW; top-left letterbox 320; right/bottom padding 114','training_dataset':'10.17632/56c75hvm7r.1'}.items():
        prop=graph.metadata_props.add();prop.key=k;prop.value=v
    onnx.save(graph,str(path))
    options=ort.SessionOptions();options.intra_op_num_threads=3;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(20261109);arr=rng.uniform(0,255,(1,3,320,320)).astype(np.float32)
    with torch.inference_mode():pt=model(torch.from_numpy(arr)).numpy()
    ox=session.run(None,{'images':arr})[0]
    max_error=float(np.max(np.abs(pt-ox)));np.testing.assert_allclose(ox,pt,atol=.01,rtol=.005)
    # Threshold selection is repeated on validation using the exact exported
    # artifact and browser-equivalent clipping/NMS, before touching test images.
    val=BananaDataset(a.data,'val');vpreds=[];vgts=[]
    for i,(image,target) in enumerate(val):
        output=session.run(None,{'images':image.unsqueeze(0).numpy()})[0][0]
        im=val.images[i];vpreds.append(postprocess(output,image_shape=(im.shape[1],im.shape[0])));vgts.append(gt_from_target(target))
    candidates=[metrics(vpreds,vgts,t) for t in [.15,.2,.25,.3,.35,.4,.5]]
    selected=max(candidates,key=lambda m:(m['f1'],m['precision'],m['threshold']))
    selection['metrics']=selected;selection['selectedThreshold']=selected['threshold'];selection['onnxBrowserEquivalentCandidates']=candidates
    threshold=selected['threshold']
    (evaluation/'validation-selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    # The test inference below uses the exact deployed ONNX artifact.
    test=BananaDataset(a.data,'test');preds=[];gts=[];times=[]
    for i in range(5):session.run(None,{'images':arr})
    for image,target in test:
        start=time.perf_counter();output=session.run(None,{'images':image.unsqueeze(0).numpy()})[0][0];times.append((time.perf_counter()-start)*1000)
        im=test.images[len(preds)];preds.append(postprocess(output,image_shape=(im.shape[1],im.shape[0])));gts.append(gt_from_target(target))
    result=metrics(preds,gts,threshold)
    audit=json.loads((a.data/'audit.json').read_text());provenance=json.loads((ROOT/'provenance.json').read_text())
    now=datetime.now(timezone.utc).isoformat();digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    limits=[
        'Erste CPU-Transfer-Lernrunde auf einem kleinen öffentlichen Datensatz. Keine unabhängige Prüfung in Supermarktumgebungen.',
        'Die öffentlichen manuellen Annotationen sind teilweise ungenau; keine vollständige Neuannotation. EXIF-Drehungen wurden vor Training und Prüfung normalisiert.',
        'Nahe Bildduplikate per pHash und Namensfamilien gruppiert. Unbekannte Aufnahmeserien und dieselben physischen Früchte können trotz Audit gruppenübergreifend vorkommen.',
        'Die Originalannotation kann ein Büschel als ein Objekt zusammenfassen. Sichere Einzelbananenerkennung bei Büscheln und Überlappungen ist nicht belegt.',
        'Der Datensatz enthält keine annotierten Negativbilder. Falschalarme auf beliebigen Szenen ohne Bananen sind nicht quantifiziert.',
        'Gemischte Reifestufen im selben realen Bild, wechselndes Licht, starke Bewegung, Überlappungen und Kameradrehungen benötigen separate Praxistests.',
        'Android- und iPhone-Kameratests sowie Browser- und Energieverbrauchsmessungen auf echten Geräten stehen aus.',
        'Die Konfidenz ist ein Modellausgabewert und keine kalibrierte Wahrscheinlichkeit; unsicher ab Detektionswert <0,60 oder Abstand der zwei höchsten Klassenwerte <0,15.'
    ]
    speed={'environment':'Cloud-Linux, ONNX Runtime CPU, 3 Threads; keine Handy-Messung','medianMs':float(np.median(times)),'p95Ms':float(np.percentile(times,95)),'evaluationsPerSecond':1000/float(np.median(times)),'sampleCount':len(times),'includes':'Nur ONNX-Inferenz, ohne Kamera, Vor-/Nachverarbeitung oder Rendering','cpuCount':os.cpu_count(),'platform':platform.platform(),'onnxruntime':ort.__version__}
    report={'title':'Bananenblick – Modellprüfung','createdAt':now,'status':'prototype-device-validation-pending','modelSha256':digest,'dataset':provenance['dataset'],'splits':audit['splits'],'audit':{'sourceImages':audit['source_images'],'uniqueImages':audit['unique_decoded_images'],'exactDuplicatesRemoved':len(audit['exact_duplicates_removed']),'nearDuplicatePairsGrouped':len(audit['near_duplicate_pairs']),'groups':audit['group_count'],'detectedCrossSplitDuplicates':len(audit['cross_split_near_duplicate_pairs']),'boundaryBoxesClipped':len(audit['boundary_clipping']),'exifOrientations':audit['exif_orientations'],'orientationHandling':audit['orientation_handling'],'limitations':audit['limitations']},'training':{k:checkpoint[k] for k in ['epoch','epochsRequested','seed','trainingSeconds','pretrainedSha256']},'selection':{'checkpointRule':'Höchstes mAP50 im Validierungssatz','thresholdRule':'Höchster F1-Wert im Validierungssatz; bei Gleichstand höhere Präzision, dann höherer Schwellwert','threshold':threshold,'validation':selection['metrics']},'test':result,'speed':speed,'export':{'input':[1,3,320,320],'output':[1,2100,8],'opset':17,'maxAbsoluteDifferenceTorchOnnx':max_error,'bytes':path.stat().st_size},'limitations':limits,'deviceTests':{'android':'nicht durchgeführt','iphone':'nicht durchgeführt'},'scenarioTests':{'singleFruit':'öffentlicher Holdout ausgewertet','mixedRipenessRealImage':'nicht separat geprüft','bunchesIndividualFruit':'nicht separat geprüft','occlusion':'nicht separat geprüft','changingLight':'nicht separat geprüft','noBananas':'kein annotierter Negativtest im Datensatz','rotationAndMotion':'echte Geräteprüfung ausstehend'},'privacyVerification':{'sourceCode':'Keine Bildübertragungsfunktion in der Modellpipeline; App-Prüfung durch Integrationsverantwortlichen','realDeviceNetworkCapture':'ausstehend'}}
    (evaluation/'test-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (evaluation/'split-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    (evaluation/'validation-history.json').write_text((a.training/'history.json').read_text())
    (evaluation/'test-predictions.json').write_text(json.dumps([{'id':r['id'],'source':r['source'],'groundTruth':g,'predictions':ps} for r,g,ps in zip(test.records,gts,preds)],ensure_ascii=False,indent=2))
    card={'name':'Bananenblick YOLOX-Nano','architecture':'YOLOX-Nano','version':'0.1.0','path':'/models/banana-yolox-nano.onnx','inputSize':320,'inputName':'images','outputName':'predictions','classes':CLASSES,'trainedAt':now,'dataset':{'title':provenance['dataset']['title'],'url':provenance['dataset']['url'],'license':'CC BY 4.0'},'preprocessing':{'channelOrder':'BGR','range':[0,255],'layout':'NCHW','padding':114,'letterbox':'top-left'},'output':{'shape':[1,2100,8],'columns':['cx','cy','width','height','objectness','unreif','reif','überreif'],'decoded':True},'thresholds':{'detection':threshold,'classification':.6,'margin':.15,'nmsIou':.45},'sha256':digest,'bytes':path.stat().st_size,'metrics':{'mAP50':result['mAP50'],'precision':result['precision'],'recall':result['recall'],'visibleRecall':result['visibleRecall'],'uncertainCount':result['uncertainCount'],'testImages':len(test),'perClass':result['perClass'],'cpuReference':speed,'realDeviceTestsComplete':False},'evaluationUrl':'/evaluation/test-report.json','status':'prototype','limitations':limits}
    artifacts=ROOT/'artifacts';artifacts.mkdir(parents=True,exist_ok=True)
    checkpoint_path=artifacts/'banana-yolox-nano.pt';shutil.copyfile(a.training/'best.pt',checkpoint_path)
    card['checkpointSha256']=hashlib.file_digest(checkpoint_path.open('rb'),'sha256').hexdigest()
    (models/'model-card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2))
    # Select three examples from the untouched test split by ground truth class,
    # without filtering by prediction success. Thus examples do not inflate QA.
    chosen=[]
    for c,title,desc in [(0,'Grüne Banane','Unreife Schale im öffentlichen Testdatensatz.'),(1,'Gelbe Banane','Reife Schale im öffentlichen Testdatensatz.'),(2,'Braune Schale','Überreife Schale im öffentlichen Testdatensatz.')]:
        candidates=[r for r in test.records if any(int(b[0])==c for b in r['boxes'])]
        if candidates:
            # Fixed image ID order makes the choice independent of model output.
            r=sorted(candidates,key=lambda r:r['id'])[0];name=f'{["unreif","reif","ueberreif"][c]}.jpg';shutil.copyfile(a.data/r['file'],examples/name)
            chosen.append({'id':['unreif','reif','ueberreif'][c],'src':'/examples/'+name,'title':title,'description':desc+' CC BY 4.0 · Rohit et al. (2026).','width':r['width'],'height':r['height'],'sourceImage':r['source'],'sourceUrl':provenance['dataset']['url']})
    (examples/'index.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2))
    if len(chosen)>1:shutil.copyfile(examples/Path(chosen[1]['src']).name,examples/'standby.jpg')
    print(json.dumps({'model':str(path),'sha256':digest,'bytes':path.stat().st_size,'test':result,'speed':speed,'examples':chosen},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
