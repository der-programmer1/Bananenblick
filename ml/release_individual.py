"""Export once, tune NMS on validation only, then evaluate the frozen benchmark once."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, subprocess, time
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn.utils import fuse_conv_bn_eval
from model import create_model
from train import BananaDataset, metrics, postprocess
from postprocess import decode

ROOT = Path(__file__).resolve().parent
SITE = ROOT.parent
DATA = ROOT/'work/individual'
TRAINING = ROOT/'work/individual-training'
PUBLIC = SITE/'public'

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def ground_truth(record):
    ratio = min(320/record['width'], 320/record['height'])
    return [{'class': c, 'box': [(x-w/2)*record['width']*ratio, (y-h/2)*record['height']*ratio,
                               (x+w/2)*record['width']*ratio, (y+h/2)*record['height']*ratio]}
            for c,x,y,w,h in record['boxes']]

def ripe_metrics(predictions, records):
    from postprocess import overlap
    tp = fp = support = 0
    for ds,r in zip(predictions,records):
        truth = [{'cls':c, 'box':{'x':x-w/2,'y':y-h/2,'width':w,'height':h}} for c,x,y,w,h in r['boxes']]
        support += sum(g['cls']==1 for g in truth); used = set()
        for d in ds:
            if not d['ripeEvidence']: continue
            matches = [(overlap(d['box'],g['box']),j) for j,g in enumerate(truth) if j not in used]
            best = max(matches,default=(0,-1))
            if best[0]>=.5:
                used.add(best[1])
                if truth[best[1]]['cls']==1:tp+=1;continue
            fp+=1
    precision = tp/max(tp+fp,1);recall = tp/max(support,1)
    return {'precision':precision,'recall':recall,'truePositive':tp,'falsePositive':fp,'support':support,
            'f1':2*precision*recall/max(precision+recall,1e-12)}

def main():
    evaluation = PUBLIC/'evaluation'; output_report = evaluation/'individual-report.json'
    if output_report.exists():raise RuntimeError('This benchmark has already been evaluated. Do not tune on its results; freeze a new holdout.')
    benchmark = json.loads((ROOT/'individual-benchmark.json').read_text())
    assert sha(ROOT/'individual_annotations.json') == benchmark['annotationSha256']
    checkpoint = torch.load(TRAINING/'best.pt', map_location='cpu', weights_only=False)
    assert checkpoint['benchmarkSha256'] == sha(ROOT/'individual-benchmark.json')
    for records in benchmark['splits'].values():
        for r in records: assert sha(DATA/r['file'])==r['sha256_file']
    torch.set_num_threads(3);cv2.setNumThreads(1)
    model = create_model();model.load_state_dict(checkpoint['model']);model.eval()
    for m in model.modules():
        if hasattr(m,'conv') and hasattr(m,'bn') and hasattr(m,'fuseforward'):
            m.conv=fuse_conv_bn_eval(m.conv,m.bn);delattr(m,'bn');m.forward=m.fuseforward
    path = TRAINING/'banana-yolox-nano.onnx'
    sample=torch.zeros((1,3,320,320),dtype=torch.float32)
    torch.onnx.export(model,sample,str(path),input_names=['images'],output_names=['predictions'],opset_version=17,dynamo=False,external_data=False,do_constant_folding=True)
    graph=onnx.load(str(path));onnx.checker.check_model(graph)
    for k,v in {'architecture':'YOLOX-Nano','classes':'["unreif","reif","überreif"]',
                'preprocessing':'BGR float32 0..255 NCHW, top-left letterbox 320, padding 114',
                'benchmarkSha256':sha(ROOT/'individual-benchmark.json')}.items():
        prop=graph.metadata_props.add();prop.key=k;prop.value=v
    onnx.save(graph,str(path))
    options=ort.SessionOptions();options.intra_op_num_threads=3;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(20261110);arr=rng.uniform(0,255,(1,3,320,320)).astype(np.float32)
    with torch.inference_mode(): pt=model(torch.from_numpy(arr)).numpy()
    ox=session.run(None,{'images':arr})[0];np.testing.assert_allclose(ox,pt,atol=.01,rtol=.005)
    validation=BananaDataset(DATA,'val');raw_val=[]
    for image,_ in validation:raw_val.append(session.run(None,{'images':image.unsqueeze(0).numpy()})[0][0])
    candidates=[]
    for nms in [.4,.45,.5,.55,.6,.65,.7,.75]:
        for detection in [.25,.35,.5]:
            for classification in [.6,.7,.8]:
                thresholds={'detection':detection,'classification':classification,'margin':.15,'nmsIou':nms}
                ds=[decode(out.reshape(-1).tolist(),320,r['width'],r['height'],thresholds=thresholds) for out,r in zip(raw_val,validation.records)]
                score=ripe_metrics(ds,validation.records)
                candidates.append({'thresholds':thresholds,**score})
    eligible=[c for c in candidates if c['precision']>=.95]
    if eligible:
        selected=max(eligible,key=lambda c:(c['recall'],c['precision'],-c['thresholds']['nmsIou'],c['thresholds']['detection']))
    else:selected=max(candidates,key=lambda c:(c['f1'],c['precision'],c['recall'],-c['thresholds']['nmsIou']))
    thresholds=selected['thresholds']
    selection={'selected':selected,'candidates':candidates,'rule':'Validation-only ripe recommendation precision >= .95, then highest recall; if none qualifies, highest validation F1. Ties prefer precision, lower NMS IoU, higher detection threshold. Static single-pass evidence for threshold selection; final benchmark runs the browser tracker for three confirmations.',
               'checkpointEpoch':checkpoint['epoch'],'validationImages':len(validation),'benchmarkUntouched':True}
    (evaluation/'individual-validation-selection.json').write_text(json.dumps(selection,indent=2))
    print('VALIDATION_SELECTED',json.dumps(selected),flush=True)
    # Persist the exact shared browser/Python defaults before the held-out run.
    (SITE/'lib/postprocess-config.json').write_text(json.dumps(thresholds,indent=2)+'\n')
    cases=[];times=[]
    acceptance=BananaDataset(DATA,'acceptance')
    for r,(image,_) in zip(acceptance.records,acceptance):
        start=time.perf_counter();out=session.run(None,{'images':image.unsqueeze(0).numpy()})[0][0];times.append((time.perf_counter()-start)*1000)
        detections=decode(out.reshape(-1).tolist(),320,r['width'],r['height'],thresholds=thresholds)
        cases.append({'id':r['id'],'scene':r['scene'],'boxes':r['boxes'],
                      'detections':[{k:v for k,v in d.items() if not k.startswith('_')} for d in detections]})
    predictions=evaluation/'individual-predictions.json';predictions.write_text(json.dumps(cases,ensure_ascii=False,indent=2))
    subprocess.run(['node','--import','tsx','scripts/evaluate-recommendations.ts',str(predictions),str(output_report)],cwd=SITE,check=True)
    report=json.loads(output_report.read_text());report.update({
        'realBunchScenes':None,'realBunchSceneCountVerified':False,
        'generatedAt':datetime.now(timezone.utc).isoformat(),'modelSha256':sha(path),'benchmarkSha256':sha(ROOT/'individual-benchmark.json'),
        'frozenAt':benchmark['frozenAt'],'thresholds':thresholds,'checkpointEpoch':checkpoint['epoch'],
        'limitations':benchmark['limitations'],'inferenceMs':{'median':float(np.median(times)),'p95':float(np.percentile(times,95)),
        'context':'ONNX Runtime CPU, Linux, three threads; inference only, not smartphone throughput'}})
    output_report.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    # Separately re-measure the legacy single-fruit test set with the new artifact.
    legacy=BananaDataset(DATA,'test');preds=[]
    for r,(image,_) in zip(legacy.records,legacy):
        out=session.run(None,{'images':image.unsqueeze(0).numpy()})[0][0]
        preds.append(postprocess(out,threshold=.01,nms=thresholds['nmsIou'],image_shape=(r['width'],r['height'])))
    legacy_metrics=metrics(preds,[ground_truth(r) for r in legacy.records],thresholds['detection'],thresholds['classification'],thresholds['margin'])
    legacy_metrics['nmsIou']=thresholds['nmsIou']
    previous=json.loads((evaluation/'test-report.json').read_text())
    (evaluation/'legacy-v0.1-test-report.json').write_text(json.dumps(previous,ensure_ascii=False,indent=2))
    (evaluation/'test-report.json').write_text(json.dumps({'generatedAt':report['generatedAt'],'modelSha256':sha(path),
        'scope':'Legacy 101-image single-fruit test, re-evaluated after fixed validation selection; not the individual-bunch acceptance benchmark.',
        'test':legacy_metrics,'individualBenchmark':'individual-report.json'},ensure_ascii=False,indent=2))
    (evaluation/'test-predictions.json').write_text(json.dumps({'modelSha256':sha(path),'scope':'Legacy 101-image test; input-pixel xyxy coordinates',
        'images':[{'id':r['id'],'source':r['source'],'groundTruth':ground_truth(r),'predictions':ps} for r,ps in zip(legacy.records,preds)]},ensure_ascii=False,indent=2))
    card=json.loads((PUBLIC/'models/model-card.json').read_text())
    card.update({'version':'0.2.0','trainedAt':report['generatedAt'],'sha256':sha(path),'bytes':path.stat().st_size,
                 'thresholds':thresholds,'metrics':{**legacy_metrics,'testImages':len(legacy)},
                 'recommendationPolicy':{'requiredConsecutiveConfirmations':3,'resetOnUncertainOrMissing':True},
                 'individualBenchmark':{k:v for k,v in report.items() if k!='scenes'},
                 'limitations':benchmark['limitations']+['The individual-fruit target is '+('met on this static benchmark.' if report['passed'] else 'NOT met on this static benchmark.'),
                    'No physical Android/Chrome or iPhone/Safari checks in this environment.'],
                 'training':{'epochs':8,'selectedEpoch':checkpoint['epoch'],'seconds':checkpoint['trainingSeconds'],
                    'seed':checkpoint['seed'],'initialCheckpointSha256':checkpoint['pretrainedSha256'],
                    'data':{s:len(json.loads((DATA/f'{s}.json').read_text())) for s in ['train','val','test','acceptance']},
                    'benchmarkFrozenCommit':'28b35eb','validationSelection':'/evaluation/individual-validation-selection.json'}})
    shutil.copyfile(path,PUBLIC/'models/banana-yolox-nano.onnx')
    shutil.copyfile(TRAINING/'best.pt',ROOT/'artifacts/banana-yolox-nano.pt')
    card['checkpointSha256']=sha(ROOT/'artifacts/banana-yolox-nano.pt')
    (PUBLIC/'models/model-card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2))
    shutil.copyfile(ROOT/'individual-benchmark.json',evaluation/'individual-benchmark.json')
    shutil.copyfile(TRAINING/'history.json',evaluation/'individual-training-history.json')
    print('RELEASE_EVALUATED',json.dumps({k:v for k,v in report.items() if k!='scenes'}),flush=True)

if __name__ == '__main__':main()
