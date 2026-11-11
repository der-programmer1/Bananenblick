"""Train with real scene crops; keep the newly frozen holdout unseen."""
from pathlib import Path
import copy
import hashlib
import json
import math
import random
import shutil
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


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()


def main():
    frozen=ROOT/'individual-benchmark-v4.json'
    plan=json.loads(frozen.read_text())
    protocol=plan['protocol']
    assert plan['annotationSha256']==sha(ROOT/'individual_annotations_v4.json')
    assert plan['trainingAnnotationSha256']==sha(ROOT/'individual_training_v4.json')
    for split,digest in plan['splitSha256'].items():
        assert sha(DATA/f'{split}.json')==digest
    assert not (SITE/'public/evaluation/individual-v4-report.json').exists(), 'The new holdout has already been inspected.'
    subprocess.run(['git','ls-files','--error-unmatch','ml/individual-benchmark-v4.json'],cwd=SITE,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--exit-code','HEAD','--','ml/individual-benchmark-v4.json','ml/individual_annotations_v4.json','ml/individual_training_v4.json','ml/train_individual_v4.py','ml/train.py','ml/scene_augmentation.py'],cwd=SITE,check=True,stdout=subprocess.DEVNULL)
    audit_path=ROOT/'work/v4-split-audit.json'
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'audit_individual_data.py'),'--data',str(DATA),'--output',str(audit_path)],cwd=SITE,check=True,stdout=subprocess.DEVNULL)
    assert json.loads(audit_path.read_text())['passedAutomaticChecks']
    seed=protocol['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'history.json').exists(), 'Preserve each training attempt rather than silently overwriting it.'
    initial=OUT/'initial.pt'
    shutil.copyfile(ROOT/protocol['initialCheckpoint'],initial)
    assert sha(initial)==protocol['initialCheckpointSha256']
    size=protocol['inputSize'];validation.OUT=OUT;validation.SIZE=size
    model=create_model(initial)
    ema=copy.deepcopy(model).eval()
    for parameter in ema.parameters():parameter.requires_grad_(False)
    dataset=SceneDataset(DATA,ROOT/'work/sprites',synthetic_count=protocol['syntheticScenesPerEpoch'],
        real_repeat=protocol['realPositiveSamplingRepeat'],size=size,
        plain_background_probability=protocol['plainBackgroundProbability'],
        real_crop_probability=protocol['realBunchCropProbability'])
    val=BananaDataset(DATA,'val',size=size)
    loader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=0,drop_last=True)
    epochs=protocol['maxEpochs'];base_lr=protocol['learningRate']
    optimizer=torch.optim.AdamW(model.parameters(),lr=base_lr,weight_decay=.0001)
    started=time.time();steps=0;best=-1;best_epoch=0;history=[]
    print('TRAIN_V4_STARTED',json.dumps({'benchmarkSha256':sha(frozen),'inputSize':size,
        'trainingImages':len(dataset.base.records),'samplesPerEpoch':len(dataset),'validationImages':len(val),
        'maxEpochs':epochs,'holdoutInferred':False}),flush=True)
    baseline=validation.validate(ema,val)
    (OUT/'baseline-validation.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2)+'\n')
    for epoch in range(epochs):
        model.train();model.head.use_l1=epoch>=epochs-4
        sums=[];start=time.time()
        for batch,(images,targets) in enumerate(loader):
            fraction=(epoch+batch/len(loader))/epochs
            warmup=min(1,(steps+1)/len(loader))
            lr=(base_lr*.1+base_lr*.9*.5*(1+math.cos(math.pi*fraction)))*warmup
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            loss=model(images,targets)['total_loss']
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step();steps+=1
            decay=min(.995,1-1/(steps+1))
            with torch.no_grad():
                current=model.state_dict()
                for key,value in ema.state_dict().items():
                    if value.is_floating_point():value.mul_(decay).add_(current[key],alpha=1-decay)
                    else:value.copy_(current[key])
            sums.append(float(loss.detach()))
            if batch%40==0:print(json.dumps({'epoch':epoch+1,'batch':batch,'batches':len(loader),
                'loss':sums[-1],'elapsedSeconds':round(time.time()-started,1)}),flush=True)
        result=validation.validate(ema,val)
        row={'epoch':epoch+1,'loss':float(np.mean(sums)),'seconds':time.time()-start,
             'validation':{k:v for k,v in result.items() if k!='scenes'}}
        history.append(row)
        payload={'model':ema.state_dict(),'epoch':epoch+1,'seed':seed,'inputSize':size,
            'epochsRequested':epochs,'validation':row['validation'],'trainingSeconds':time.time()-started,
            'benchmarkSha256':sha(frozen),'initialCheckpointSha256':sha(initial),
            'architecture':'YOLOX-Nano .33/.25 depthwise','trainingProtocol':protocol}
        if result['f1']>best:
            best=result['f1'];best_epoch=epoch+1;torch.save(payload,OUT/'best.pt')
        torch.save(payload,OUT/'last.pt')
        (OUT/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2)+'\n')
        print('VALIDATION_V4',json.dumps({**row,'bestEpoch':best_epoch,'bestF1':best}),flush=True)
        if epoch+1>=protocol['minimumEpochs'] and epoch+1-best_epoch>=protocol['earlyStoppingPatience']:
            break
    print('TRAIN_V4_FINISHED',json.dumps({'epochs':len(history),'selectedEpoch':best_epoch,
        'validationF1':best,'seconds':time.time()-started,'holdoutInferred':False}),flush=True)


if __name__=='__main__':main()
