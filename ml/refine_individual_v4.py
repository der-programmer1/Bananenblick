"""Finish a predeclared localization refinement without viewing the new holdout."""
from pathlib import Path
import copy
import hashlib
import json
import math
import random
import subprocess
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import create_model
from train import BananaDataset
from scene_augmentation import SceneDataset
import train_individual_v2 as validation

ROOT=Path(__file__).resolve().parent
SITE=ROOT.parent
DATA=ROOT/'work/individual-v4'
OUT=ROOT/'work/training-v4'
PLAN=ROOT/'individual-refinement-v4.json'


def sha(path):
    with path.open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def main():
    plan=json.loads(PLAN.read_text())
    benchmark=ROOT/'individual-benchmark-v4.json'
    frozen=json.loads(benchmark.read_text());protocol=frozen['protocol']
    OUT.mkdir(exist_ok=True)
    assert plan['benchmarkSha256']==sha(benchmark)
    assert not (SITE/'public/evaluation/individual-v4-report.json').exists()
    assert not (OUT/'refinement-summary.json').exists()
    subprocess.run(['git','ls-files','--error-unmatch','ml/individual-refinement-v4.json'],cwd=SITE,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--exit-code','HEAD','--','ml/individual-refinement-v4.json','ml/refine_individual_v4.py'],cwd=SITE,check=True,stdout=subprocess.DEVNULL)
    for split,digest in frozen['splitSha256'].items():assert sha(DATA/f'{split}.json')==digest
    initial=ROOT/plan['initialCheckpoint']
    assert sha(initial)==plan['initialCheckpointSha256']
    history_path=ROOT/plan['stage1HistorySource']
    history=json.loads(history_path.read_text())
    assert len(history)==plan['stage1CompletedEpochs']
    assert sha(history_path)==plan['stage1HistorySha256']
    selected=torch.load(initial,map_location='cpu',weights_only=False)
    assert selected['benchmarkSha256']==sha(benchmark)
    seed=plan['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
    size=protocol['inputSize'];validation.OUT=OUT;validation.SIZE=size
    model=create_model(initial);ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    dataset=SceneDataset(DATA,ROOT/'work/sprites',synthetic_count=protocol['syntheticScenesPerEpoch'],
        real_repeat=protocol['realPositiveSamplingRepeat'],size=size,
        plain_background_probability=protocol['plainBackgroundProbability'],
        real_crop_probability=protocol['realBunchCropProbability'])
    val=BananaDataset(DATA,'val',size=size)
    loader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=0,drop_last=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learningRate'],weight_decay=.0001)
    best=selected['validation']['f1'];best_epoch=selected['epoch'];steps=0;started=time.time()
    print('V4_REFINEMENT_STARTED',json.dumps({'planSha256':sha(PLAN),'initialCheckpointSha256':sha(initial),
        'stage1CompletedEpochs':len(history),'eligibleInitialEpoch':best_epoch,'epochs':plan['epochs'],
        'useL1':True,'holdoutInferred':False}),flush=True)
    for phase_epoch in range(plan['epochs']):
        model.train();model.head.use_l1=True;sums=[];start=time.time()
        for batch,(images,targets) in enumerate(loader):
            fraction=(phase_epoch+batch/len(loader))/plan['epochs']
            lr=plan['learningRate']*(.25+.75*.5*(1+math.cos(math.pi*fraction)))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);loss=model(images,targets)['total_loss']
            if not torch.isfinite(loss):raise RuntimeError('Non-finite refinement loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();steps+=1
            with torch.no_grad():
                current=model.state_dict()
                for key,value in ema.state_dict().items():
                    if value.is_floating_point():value.mul_(.99).add_(current[key],alpha=.01)
                    else:value.copy_(current[key])
            sums.append(float(loss.detach()))
            if batch%40==0:print(json.dumps({'refinementEpoch':phase_epoch+1,'batch':batch,'batches':len(loader),
                'loss':sums[-1],'elapsedSeconds':round(time.time()-started,1)}),flush=True)
        result=validation.validate(ema,val)
        epoch=plan['stage1CompletedEpochs']+phase_epoch+1
        row={'epoch':epoch,'phase':'l1-refinement','phaseEpoch':phase_epoch+1,'seed':seed,
            'loss':float(np.mean(sums)),'seconds':time.time()-start,
            'validation':{k:v for k,v in result.items() if k!='scenes'}}
        history.append(row)
        payload={'model':ema.state_dict(),'epoch':epoch,'seed':seed,'inputSize':size,
            'epochsRequested':plan['stage1CompletedEpochs']+plan['epochs'],'validation':row['validation'],
            'benchmarkSha256':sha(benchmark),'initialCheckpointSha256':selected['initialCheckpointSha256'],
            'architecture':selected['architecture'],'trainingProtocol':protocol,'refinementPlanSha256':sha(PLAN)}
        if result['f1']>best:
            best=result['f1'];best_epoch=epoch;selected=copy.deepcopy(payload)
        torch.save(payload,OUT/'refinement-last.pt')
        (OUT/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2)+'\n')
        print('V4_REFINEMENT_VALIDATION',json.dumps({**row,'selectedEpoch':best_epoch,'selectedF1':best}),flush=True)
    selected['refinement']={'planSha256':sha(PLAN),'initialCheckpointSha256':sha(initial),
        'stage1CompletedEpochs':plan['stage1CompletedEpochs'],'epochs':plan['epochs'],
        'selectedFromRefinement':best_epoch>plan['stage1CompletedEpochs']}
    torch.save(selected,OUT/'best.pt')
    summary={**selected['refinement'],'selectedEpoch':best_epoch,'selectedF1':best,
        'totalCompletedEpochs':len(history),'seconds':time.time()-started,'holdoutInferred':False,
        'selectedCheckpointSha256':sha(OUT/'best.pt')}
    (OUT/'refinement-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print('V4_REFINEMENT_FINISHED',json.dumps(summary),flush=True)


if __name__=='__main__':main()
