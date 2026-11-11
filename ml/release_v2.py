"""Export and calibrate the full-backbone model, then test the NEW frozen set once."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,time
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn.utils import fuse_conv_bn_eval
from model import create_model
from train import BananaDataset,metrics
from postprocess import decode,DEFAULTS

ROOT=Path(__file__).resolve().parent;SITE=ROOT.parent;DATA=ROOT/'work/individual-v3';TRAINING=ROOT/'work/training-v3';PUBLIC=SITE/'public';EVAL=PUBLIC/'evaluation'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def truth(r,size):
    ratio=min(size/r['width'],size/r['height'])
    return [{'class':c,'box':[(x-w/2)*r['width']*ratio,(y-h/2)*r['height']*ratio,(x+w/2)*r['width']*ratio,(y+h/2)*r['height']*ratio]} for c,x,y,w,h in r['boxes']]
def raw_predictions(ds,r,size):
    w,h=r['width'],r['height'];ratio=min(size/w,size/h)
    return [{'class':d['predictedClass'],'score':d['confidence'],'classMargin':d['_classMargin'],'classProbability':d['_classProbability'],
        'objectness':d['_objectness'],'box':[d['box']['x']*w*ratio,d['box']['y']*h*ratio,(d['box']['x']+d['box']['width'])*w*ratio,(d['box']['y']+d['box']['height'])*h*ratio]} for d in ds]

def main():
    target=EVAL/'individual-v2-report.json'
    if target.exists():raise RuntimeError('V2 test already inspected. No retuning or report overwrite; use a new frozen holdout.')
    frozen=ROOT/'individual-benchmark-v2.json';manifest=json.loads(frozen.read_text());checkpoint=torch.load(TRAINING/'best.pt',map_location='cpu',weights_only=False)
    plan_path=ROOT/'individual-training-v3.json';plan=json.loads(plan_path.read_text())
    assert checkpoint['benchmarkSha256']==sha(frozen)
    assert checkpoint['trainingPlanSha256']==sha(plan_path)
    assert plan['benchmarkSha256']==sha(frozen)
    for split,digest in plan['splitSha256'].items():assert sha(DATA/f'{split}.json')==digest
    assert manifest['annotationSha256']==sha(ROOT/'individual_annotations_v2.json')
    for rs in manifest['splits'].values():
        for r in rs:assert sha(DATA/r['file'])==r['sha256_file']
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'audit_individual_data.py')],cwd=SITE,check=True)
    assert json.loads((EVAL/'individual-split-audit.json').read_text())['passedAutomaticChecks']
    size=checkpoint['inputSize'];torch.set_num_threads(2);cv2.setNumThreads(1)
    model=create_model();model.load_state_dict(checkpoint['model']);model.eval()
    for m in model.modules():
        if hasattr(m,'conv') and hasattr(m,'bn') and hasattr(m,'fuseforward'):
            m.conv=fuse_conv_bn_eval(m.conv,m.bn);delattr(m,'bn');m.forward=m.fuseforward
    model_path=TRAINING/'banana-yolox-nano.onnx';sample=torch.zeros((1,3,size,size),dtype=torch.float32)
    torch.onnx.export(model,sample,str(model_path),input_names=['images'],output_names=['predictions'],opset_version=17,dynamo=False,external_data=False,do_constant_folding=True)
    graph=onnx.load(str(model_path));onnx.checker.check_model(graph)
    for k,v in {'architecture':'YOLOX-Nano','classes':'["unreif","reif","überreif"]','preprocessing':f'BGR float32 0..255 CHW; top-left letterbox {size}, padding 114','benchmarkSha256':sha(frozen)}.items():
        p=graph.metadata_props.add();p.key=k;p.value=v
    onnx.save(graph,str(model_path))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(20261112);arr=rng.uniform(0,255,(1,3,size,size)).astype(np.float32)
    with torch.inference_mode():pytorch=model(torch.from_numpy(arr)).numpy()
    exported=session.run(None,{'images':arr})[0];np.testing.assert_allclose(exported,pytorch,atol=.015,rtol=.005)
    val=BananaDataset(DATA,'val',size=size);raw=[]
    for image,_ in val:raw.append(session.run(None,{'images':image.unsqueeze(0).numpy()})[0][0])
    cache={'split':'validation','byNms':{}}
    for nms in [.45,.55,.65,.75,.85,.9]:
        cases=[]
        for output,r in zip(raw,val.records):
            ds=decode(output.reshape(-1).tolist(),size,r['width'],r['height'],thresholds={'detection':.1,'classification':.6,'margin':.1,'nmsIou':nms})
            cases.append({'id':str(r['id']),'scene':r.get('scene','single-fruit'),'boxes':r['boxes'],'detections':ds})
        cache['byNms'][str(nms)]=cases
    (TRAINING/'validation-cache.json').write_text(json.dumps(cache,ensure_ascii=False))
    subprocess.run(['node','--import','tsx','scripts/select-validation-thresholds.ts',str(TRAINING/'validation-cache.json'),str(TRAINING/'threshold-selection.json')],cwd=SITE,check=True)
    selection=json.loads((TRAINING/'threshold-selection.json').read_text());thresholds=selection['selected']['thresholds']
    print('V2_VALIDATION_SELECTION',json.dumps(selection['selected']),flush=True)
    # The selected validation operating point is the minimum recommendation policy.
    DEFAULTS.update(thresholds)
    (SITE/'lib/postprocess-config.json').write_text(json.dumps(thresholds,indent=2)+'\n')
    # The v2 holdout is first inferred only AFTER checkpoint and thresholds are fixed.
    cases=[];times=[];test=BananaDataset(DATA,'acceptance',size=size)
    for image,r in zip(test,test.records):
        image=image[0];start=time.perf_counter();out=session.run(None,{'images':image.unsqueeze(0).numpy()})[0][0];times.append((time.perf_counter()-start)*1000)
        ds=decode(out.reshape(-1).tolist(),size,r['width'],r['height'],thresholds=thresholds)
        cases.append({'id':str(r['id']),'scene':r['scene'],'boxes':r['boxes'],'detections':[{k:v for k,v in d.items() if not k.startswith('_')} for d in ds]})
    predictions=EVAL/'individual-v2-predictions.json';predictions.write_text(json.dumps(cases,ensure_ascii=False,indent=2))
    subprocess.run(['node','--import','tsx','scripts/evaluate-recommendations.ts',str(predictions),str(target)],cwd=SITE,check=True)
    report=json.loads(target.read_text());report.update({'generatedAt':datetime.now(timezone.utc).isoformat(),'modelSha256':sha(model_path),
        'benchmarkSha256':sha(frozen),'frozenAt':manifest['frozenAt'],'realBunchScenes':30,'realBunchSceneCountVerified':True,
        'thresholds':thresholds,'inputSize':size,'selectedEpoch':checkpoint['epoch'],'trainingPlanSha256':sha(plan_path),'limitations':[v for v in manifest['limits'] if not v.startswith('Previous failed v1 test is now validation')]+plan['limits'],
        'inferenceMs':{'median':float(np.median(times)),'p95':float(np.percentile(times,95)),'context':'ONNX Runtime CPU, Linux, two threads; inference only'}})
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    archive=EVAL/'experiment-v1';archive.mkdir(exist_ok=True)
    for name in ['individual-report.json','individual-predictions.json','individual-validation-selection.json','individual-training-history.json','individual-benchmark.json','test-report.json','test-predictions.json','validation-selection.json','validation-history.json','stress-report.json','web-runtime.json','model-report.md','acceptance.md','device-validation.json']:
        if (EVAL/name).exists():shutil.copyfile(EVAL/name,archive/name)
    shutil.copyfile(PUBLIC/'models/model-card.json',archive/'model-card.json')
    shutil.copyfile(target,EVAL/'individual-report.json');shutil.copyfile(predictions,EVAL/'individual-predictions.json')
    shutil.copyfile(TRAINING/'threshold-selection.json',EVAL/'individual-validation-selection.json')
    shutil.copyfile(TRAINING/'history.json',EVAL/'individual-training-history.json');shutil.copyfile(frozen,EVAL/'individual-benchmark.json')
    shutil.copyfile(plan_path,EVAL/'individual-training-plan.json')
    # Keep old experiments accessible under their explicit archive, not as current reports.
    for name in ['validation-selection.json','validation-history.json','stress-report.json']:
        if (EVAL/name).exists():(EVAL/name).unlink()
    # Independent legacy Mendeley test is reported separately from the bunch benchmark.
    legacy=BananaDataset(DATA,'test',size=size);ps=[];rows=[]
    for image,r in zip(legacy,legacy.records):
        out=session.run(None,{'images':image[0].unsqueeze(0).numpy()})[0][0]
        ds=decode(out.reshape(-1).tolist(),size,r['width'],r['height'],thresholds={**thresholds,'detection':.01})
        predictions_raw=raw_predictions(ds,r,size);ps.append(predictions_raw);rows.append({'id':r['id'],'source':r['source'],'groundTruth':truth(r,size),'predictions':predictions_raw})
    legacy_metrics=metrics(ps,[truth(r,size) for r in legacy.records],thresholds['detection'],thresholds['classification'],thresholds['margin'],thresholds['nmsIou'])
    (EVAL/'test-report.json').write_text(json.dumps({'modelSha256':sha(model_path),'scope':'Legacy Mendeley 101-image test; separate from the new 30-bunch/30-negative benchmark','test':legacy_metrics},ensure_ascii=False,indent=2))
    (EVAL/'test-predictions.json').write_text(json.dumps({'modelSha256':sha(model_path),'images':rows},ensure_ascii=False,indent=2))
    card=json.loads((PUBLIC/'models/model-card.json').read_text());history=json.loads((TRAINING/'history.json').read_text())
    card.update({'version':'0.3.0','inputSize':size,'trainedAt':report['generatedAt'],'sha256':sha(model_path),'bytes':model_path.stat().st_size,
        'thresholds':thresholds,'metrics':{**legacy_metrics,'testImages':len(legacy)},'individualBenchmark':{k:v for k,v in report.items() if k!='scenes'},
        'training':{'epochs':len(history),'selectedEpoch':checkpoint['epoch'],'totalRunSeconds':sum(r['seconds'] for r in history),'seed':checkpoint['seed'],
            'initialCheckpointSha256':checkpoint['initialCheckpointSha256'],'data':plan['counts'],'pilot':plan['pilot'],
            'syntheticTrainingScenesPerEpoch':600,'realPositiveRepeat':6,'fullBackboneFromEpoch':1,'benchmarkFrozenCommit':'59efe9e','continuationFrozenCommit':'18308d4','trainingPlanSha256':sha(plan_path)},
        'limitations':report['limitations']+['Individual-fruit acceptance '+('passed on this static benchmark.' if report['passed'] else 'NOT met on this static benchmark.'),'No physical Android/Chrome or iPhone/Safari checks in this environment.']})
    card['output']['shape']=[1,sum((size//s)**2 for s in [8,16,32]),8]
    card['recommendationPolicy']={'requiredConsecutiveConfirmations':3,'resetOnUncertainOrMissing':True,'minimumJointConfidence':thresholds['classification'],'meaning':'visible peel ripeness only'}
    card['additionalDatasets']=[{'name':'COCO 2017','url':'https://cocodataset.org/','sourcesAndLicenses':'/evaluation/individual-training-plan.json'},
        {'name':'Wikimedia Commons photographs','url':'https://commons.wikimedia.org/','sourcesAndLicenses':'/evaluation/individual-benchmark.json'}]
    shutil.copyfile(model_path,PUBLIC/'models/banana-yolox-nano.onnx');shutil.copyfile(TRAINING/'best.pt',ROOT/'artifacts/banana-yolox-nano.pt')
    card['checkpointSha256']=sha(ROOT/'artifacts/banana-yolox-nano.pt')
    (PUBLIC/'models/model-card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2)+'\n')
    provenance_path=ROOT/'provenance.json';provenance=json.loads(provenance_path.read_text())
    provenance['training_environment']['num_threads']=2
    provenance['individual_fruit_v3']={'modelSha256':card['sha256'],'training':card['training'],
        'benchmark':'individual-benchmark-v2.json','trainingPlan':'individual-training-v3.json',
        'acceptedStaticBenchmark':report['passed'],'physicalDeviceTestsPerformed':False,
        'annotationReview':'Coding assistant visible-instance review; independent human adjudication pending.'}
    provenance_path.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    (SITE/'lib/postprocess-config.json').write_text(json.dumps(thresholds,indent=2)+'\n')
    print('V2_TEST_RESULT',json.dumps({k:v for k,v in report.items() if k!='scenes'}),flush=True)

if __name__=='__main__':main()
