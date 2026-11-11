"""Calibrate on validation and evaluate the frozen holdout exactly once.

Failed candidates stay in an explicit experiment archive. Only an accepted
static candidate replaces the app's current weights; phone evidence remains a
separate publication requirement.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import shutil
import subprocess
import time
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn.utils import fuse_conv_bn_eval
from model import create_model
from train import BananaDataset, metrics
from postprocess import decode, DEFAULTS
from release_v2 import truth, raw_predictions

ROOT=Path(__file__).resolve().parent
SITE=ROOT.parent
DATA=ROOT/'work/individual-v4'
TRAINING=ROOT/'work/training-v4'
EVALUATION=SITE/'public/evaluation'
ARCHIVE=EVALUATION/'experiment-v4'


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()


def main():
    target=EVALUATION/'individual-v4-report.json'
    if target.exists():raise RuntimeError('V4 holdout already inspected; preserve its result and freeze a new holdout before further test-driven tuning.')
    frozen=ROOT/'individual-benchmark-v4.json'
    plan=json.loads(frozen.read_text())
    checkpoint=torch.load(TRAINING/'best.pt',map_location='cpu',weights_only=False)
    history=json.loads((TRAINING/'history.json').read_text())
    protocol=plan['protocol']
    if (ROOT/'individual-refinement-v4.json').exists():
        refinement=json.loads((TRAINING/'refinement-summary.json').read_text())
        assert refinement['planSha256']==sha(ROOT/'individual-refinement-v4.json')
        assert refinement['selectedCheckpointSha256']==sha(TRAINING/'best.pt')
        assert refinement['totalCompletedEpochs']==len(history)
    assert len(history)==protocol['maxEpochs'] or (
        len(history)>=protocol['minimumEpochs'] and
        len(history)-checkpoint['epoch']>=protocol['earlyStoppingPatience']
    ), 'Wait for the complete, predeclared training run before selecting and testing a model.'
    assert checkpoint['benchmarkSha256']==sha(frozen)
    assert plan['annotationSha256']==sha(ROOT/'individual_annotations_v4.json')
    assert plan['trainingAnnotationSha256']==sha(ROOT/'individual_training_v4.json')
    for split,digest in plan['splitSha256'].items():assert sha(DATA/f'{split}.json')==digest
    for rows in plan['supplemental'].values():
        for record in rows:assert sha(DATA/record['file'])==record['sha256_file']
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'audit_individual_data.py'),'--data',str(DATA),
        '--output',str(ROOT/'work/v4-split-audit.json')],cwd=SITE,check=True,stdout=subprocess.DEVNULL)
    assert json.loads((ROOT/'work/v4-split-audit.json').read_text())['passedAutomaticChecks']
    ARCHIVE.mkdir(exist_ok=True)
    size=checkpoint['inputSize'];torch.set_num_threads(2);cv2.setNumThreads(1)
    model=create_model();model.load_state_dict(checkpoint['model']);model.eval()
    for layer in model.modules():
        if hasattr(layer,'conv') and hasattr(layer,'bn') and hasattr(layer,'fuseforward'):
            layer.conv=fuse_conv_bn_eval(layer.conv,layer.bn);delattr(layer,'bn');layer.forward=layer.fuseforward
    model_path=TRAINING/'banana-yolox-nano.onnx'
    sample=torch.zeros((1,3,size,size),dtype=torch.float32)
    torch.onnx.export(model,sample,str(model_path),input_names=['images'],output_names=['predictions'],
        opset_version=17,dynamo=False,external_data=False,do_constant_folding=True)
    graph=onnx.load(str(model_path));onnx.checker.check_model(graph)
    for key,value in {'architecture':'YOLOX-Nano','classes':'["unreif","reif","überreif"]',
        'preprocessing':f'BGR float32 0..255 CHW; top-left letterbox {size}, padding 114',
        'benchmarkSha256':sha(frozen)}.items():
        prop=graph.metadata_props.add();prop.key=key;prop.value=value
    onnx.save(graph,str(model_path))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
    array=np.random.default_rng(20261114).uniform(0,255,(1,3,size,size)).astype(np.float32)
    with torch.inference_mode():expected=model(torch.from_numpy(array)).numpy()
    actual=session.run(None,{'images':array})[0]
    np.testing.assert_allclose(actual,expected,atol=.015,rtol=.005)
    validation=BananaDataset(DATA,'val',size=size)
    raw=[session.run(None,{'images':image[0].unsqueeze(0).numpy()})[0][0] for image in validation]
    cache={'split':'validation','byNms':{}}
    for nms in [.45,.55,.65,.75,.85,.9]:
        cases=[]
        for output,record in zip(raw,validation.records):
            detections=decode(output.reshape(-1).tolist(),size,record['width'],record['height'],
                thresholds={'detection':.1,'classification':.6,'margin':.1,'nmsIou':nms})
            cases.append({'id':str(record['id']),'scene':record.get('scene','single-fruit'),
                'boxes':record['boxes'],'detections':detections})
        cache['byNms'][str(nms)]=cases
    cache_path=TRAINING/'validation-cache.json';cache_path.write_text(json.dumps(cache,ensure_ascii=False))
    selection_path=TRAINING/'threshold-selection.json'
    subprocess.run(['node','--import','tsx','scripts/select-validation-thresholds.ts',str(cache_path),str(selection_path)],cwd=SITE,check=True)
    selection=json.loads(selection_path.read_text());thresholds=selection['selected']['thresholds']
    DEFAULTS.update(thresholds)
    print('V4_VALIDATION_SELECTION',json.dumps(selection['selected']),flush=True)
    # No checkpoint, threshold, NMS or annotation change follows this boundary.
    holdout=BananaDataset(DATA,'acceptance',size=size);cases=[];times=[]
    for item,record in zip(holdout,holdout.records):
        start=time.perf_counter()
        output=session.run(None,{'images':item[0].unsqueeze(0).numpy()})[0][0]
        times.append((time.perf_counter()-start)*1000)
        detections=decode(output.reshape(-1).tolist(),size,record['width'],record['height'],thresholds=thresholds)
        cases.append({'id':str(record['id']),'scene':record['scene'],'boxes':record['boxes'],
            'detections':[{k:v for k,v in d.items() if not k.startswith('_')} for d in detections]})
    predictions=ARCHIVE/'individual-predictions.json'
    predictions.write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    subprocess.run(['node','--import','tsx','scripts/evaluate-recommendations.ts',str(predictions),str(target)],cwd=SITE,check=True)
    frozen_commit=subprocess.check_output(['git','log','-1','--format=%H','--','ml/individual-benchmark-v4.json'],cwd=SITE,text=True).strip()
    report=json.loads(target.read_text())
    report.update({'generatedAt':datetime.now(timezone.utc).isoformat(),'modelSha256':sha(model_path),
        'benchmarkSha256':sha(frozen),'trainingPlanSha256':sha(frozen),'frozenAt':plan['frozenAt'],
        'frozenCommit':frozen_commit,'realBunchScenes':30,'realBunchSceneCountVerified':True,
        'thresholds':thresholds,'inputSize':size,'selectedEpoch':checkpoint['epoch'],
        'limitations':plan['limits'],'physicalDeviceTestsPerformed':False,
        **({'refinement':checkpoint['refinement']} if 'refinement' in checkpoint else {}),
        'baselineValidation':{k:v for k,v in json.loads((TRAINING/'baseline-validation.json').read_text()).items() if k!='scenes'},
        'selectedEpochValidation':checkpoint['validation'],
        'baseDatasetProvenance':[{'file':str(path.relative_to(SITE)),'sha256':sha(path)} for path in
            [ROOT/'individual-benchmark.json',ROOT/'individual-benchmark-v2.json',ROOT/'individual-training-v3.json']],
        'inferenceMs':{'median':float(np.median(times)),'p95':float(np.percentile(times,95)),
            'context':'ONNX Runtime CPU, Linux, two threads; inference only, not phone throughput'}})
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    shutil.copyfile(target,ARCHIVE/'individual-report.json')
    shutil.copyfile(selection_path,ARCHIVE/'validation-selection.json')
    shutil.copyfile(TRAINING/'history.json',ARCHIVE/'training-history.json')
    shutil.copyfile(TRAINING/'baseline-validation.json',ARCHIVE/'baseline-validation.json')
    shutil.copyfile(ROOT/'work/v4-split-audit.json',ARCHIVE/'split-audit.json')
    shutil.copyfile(frozen,ARCHIVE/'benchmark.json')
    if (ROOT/'individual-refinement-v4.json').exists():
        shutil.copyfile(ROOT/'individual-refinement-v4.json',ARCHIVE/'refinement-plan.json')
        shutil.copyfile(TRAINING/'refinement-summary.json',ARCHIVE/'refinement-summary.json')
    candidate_checkpoint=ROOT/'artifacts/banana-individual-v4-candidate.pt'
    shutil.copyfile(TRAINING/'best.pt',candidate_checkpoint)
    legacy=BananaDataset(DATA,'test',size=size);legacy_predictions=[];legacy_rows=[]
    for item,record in zip(legacy,legacy.records):
        output=session.run(None,{'images':item[0].unsqueeze(0).numpy()})[0][0]
        detections=decode(output.reshape(-1).tolist(),size,record['width'],record['height'],thresholds={**thresholds,'detection':.01})
        values=raw_predictions(detections,record,size);legacy_predictions.append(values)
        legacy_rows.append({'id':record['id'],'source':record['source'],'groundTruth':truth(record,size),'predictions':values})
    legacy_metrics=metrics(legacy_predictions,[truth(r,size) for r in legacy.records],thresholds['detection'],
        thresholds['classification'],thresholds['margin'],thresholds['nmsIou'])
    (ARCHIVE/'test-report.json').write_text(json.dumps({'modelSha256':sha(model_path),
        'scope':'Legacy Mendeley 101-image test; separate from the new cropped-photo benchmark',
        'test':legacy_metrics},ensure_ascii=False,indent=2)+'\n')
    (ARCHIVE/'test-predictions.json').write_text(json.dumps({'modelSha256':sha(model_path),'images':legacy_rows},ensure_ascii=False,indent=2)+'\n')
    card=copy.deepcopy(json.loads((SITE/'public/models/model-card.json').read_text()))
    card.update({'version':'0.4.0','inputSize':size,'trainedAt':report['generatedAt'],'sha256':sha(model_path),
        'bytes':model_path.stat().st_size,'checkpointSha256':sha(candidate_checkpoint),'thresholds':thresholds,
        'metrics':{**legacy_metrics,'testImages':len(legacy)},
        'individualBenchmark':{k:v for k,v in report.items() if k!='scenes'},
        'training':{'epochs':len(history),'selectedEpoch':checkpoint['epoch'],'totalRunSeconds':sum(r['seconds'] for r in history),
            'seed':checkpoint['seed'],'initialCheckpointSha256':checkpoint['initialCheckpointSha256'],
            'data':plan['counts'],'syntheticTrainingScenesPerEpoch':150,'realPositiveRepeat':8,
            'realBunchCropProbability':.5,'fullBackboneFromEpoch':1,'benchmarkFrozenCommit':frozen_commit,
            'trainingPlanSha256':sha(frozen),
            **({'refinement':checkpoint['refinement']} if 'refinement' in checkpoint else {})},
        'limitations':plan['limits']+['Static individual-fruit acceptance '+('passed.' if report['passed'] else 'NOT met.')],
        'additionalDatasets':[{'name':'COCO 2017','url':'https://cocodataset.org/',
            'sourcesAndLicenses':f'https://github.com/der-programmer1/Bananenblick/blob/{frozen_commit}/ml/individual-benchmark-v4.json'},
            {'name':'Wikimedia Commons photographs','url':'https://commons.wikimedia.org/',
            'sourcesAndLicenses':f'https://github.com/der-programmer1/Bananenblick/blob/{frozen_commit}/ml/individual-benchmark-v4.json'}],
        'recommendationPolicy':{'requiredConsecutiveConfirmations':3,'resetOnUncertainOrMissing':True,
            'minimumJointConfidence':thresholds['classification'],'meaning':'visible peel ripeness only'}})
    (ARCHIVE/'model-card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2)+'\n')
    (ARCHIVE/'postprocess-config.json').write_text(json.dumps(thresholds,indent=2)+'\n')
    if report['passed']:
        previous=EVALUATION/'experiment-v3';previous.mkdir(exist_ok=True)
        for name in ['individual-report.json','individual-predictions.json','individual-validation-selection.json',
            'individual-training-history.json','individual-benchmark.json','individual-training-plan.json',
            'test-report.json','test-predictions.json','model-report.md','acceptance.md','web-runtime.json','device-validation.json']:
            if (EVALUATION/name).exists():shutil.copyfile(EVALUATION/name,previous/name)
        shutil.copyfile(SITE/'public/models/model-card.json',previous/'model-card.json')
        for source,destination in [(target,EVALUATION/'individual-report.json'),(predictions,EVALUATION/'individual-predictions.json'),
            (selection_path,EVALUATION/'individual-validation-selection.json'),(TRAINING/'history.json',EVALUATION/'individual-training-history.json'),
            (frozen,EVALUATION/'individual-benchmark.json'),(frozen,EVALUATION/'individual-training-plan.json'),
            (ARCHIVE/'test-report.json',EVALUATION/'test-report.json'),(ARCHIVE/'test-predictions.json',EVALUATION/'test-predictions.json'),
            (model_path,SITE/'public/models/banana-yolox-nano.onnx'),(candidate_checkpoint,ROOT/'artifacts/banana-yolox-nano.pt'),
            (ARCHIVE/'model-card.json',SITE/'public/models/model-card.json'),(ARCHIVE/'postprocess-config.json',SITE/'lib/postprocess-config.json')]:
            shutil.copyfile(source,destination)
    print('V4_TEST_RESULT',json.dumps({k:v for k,v in report.items() if k!='scenes'}),flush=True)
    print('CURRENT_APP_WEIGHTS_UPDATED',report['passed'],flush=True)


if __name__=='__main__':
    main()
