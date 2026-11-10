"""Continue individual-fruit learning without exposing the new frozen test."""
from pathlib import Path
import copy,hashlib,json,math,random,subprocess,time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import create_model
from train import BananaDataset
from scene_augmentation import SceneDataset
import train_individual_v2 as pilot

ROOT=Path(__file__).resolve().parent;SITE=ROOT.parent;DATA=ROOT/'work/individual-v3';OUT=ROOT/'work/training-v3'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    frozen=ROOT/'individual-benchmark-v2.json';plan_path=ROOT/'individual-training-v3.json';plan=json.loads(plan_path.read_text());protocol=plan['protocol']
    assert plan['benchmarkSha256']==sha(frozen) and plan['newHoldoutUnchanged']
    assert plan['annotationSha256']==sha(ROOT/'validation_annotations_v3.json')
    for split,digest in plan['splitSha256'].items():assert sha(DATA/f'{split}.json')==digest
    assert not (SITE/'public/evaluation/individual-v2-report.json').exists(),'The new test must still be unseen.'
    seed=20261113;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True);OUT.mkdir(parents=True,exist_ok=True)
    initial=OUT/'initial.pt';assert initial.exists(),'Copy the selected completed pilot checkpoint before continuation.'
    size=protocol['inputSize'];pilot.OUT=OUT;pilot.SIZE=size
    model=create_model(initial);ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    dataset=SceneDataset(DATA,ROOT/'work/sprites',synthetic_count=protocol['syntheticScenesPerEpoch'],real_repeat=protocol['realPositiveSamplingRepeat'],size=size,
        plain_background_probability=protocol['plainBackgroundProbability'])
    val=BananaDataset(DATA,'val',size=size);loader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=0,drop_last=True)
    epochs=protocol['maxEpochs'];base_lr=protocol['learningRate'];optimizer=torch.optim.AdamW(model.parameters(),lr=base_lr,weight_decay=.0001)
    started=time.time();steps=0;best=-1;best_epoch=0;history=[]
    print('TRAIN_V3_STARTED',json.dumps({'benchmarkSha256':sha(frozen),'trainingPlanSha256':sha(plan_path),'initialCheckpointSha256':sha(initial),
        'inputSize':size,'uniqueImages':len(dataset.base.records),'samplesPerEpoch':len(dataset),'validationImages':len(val),'epochsMax':epochs}),flush=True)
    for epoch in range(epochs):
        model.train();model.head.use_l1=epoch>=epochs-4;sums=[];start=time.time()
        for batch,(images,targets) in enumerate(loader):
            fraction=(epoch+batch/len(loader))/epochs;warmup=min(1,(steps+1)/(len(loader)*.5))
            lr=(base_lr*.1+base_lr*.9*.5*(1+math.cos(math.pi*fraction)))*warmup
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);loss=model(images,targets)['total_loss']
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();steps+=1
            decay=min(.99,1-1/(steps+1))
            with torch.no_grad():
                current=model.state_dict()
                for k,v in ema.state_dict().items():
                    if v.is_floating_point():v.mul_(decay).add_(current[k],alpha=1-decay)
                    else:v.copy_(current[k])
            sums.append(float(loss.detach()))
            if batch%40==0:print(json.dumps({'epoch':epoch+1,'batch':batch,'batches':len(loader),'loss':sums[-1],'elapsedSeconds':round(time.time()-started,1)}),flush=True)
        result=pilot.validate(ema,val);row={'epoch':epoch+1,'loss':float(np.mean(sums)),'seconds':time.time()-start,'validation':{k:v for k,v in result.items() if k!='scenes'}};history.append(row)
        payload={'model':ema.state_dict(),'epoch':epoch+1,'seed':seed,'inputSize':size,'epochsRequested':epochs,'validation':row['validation'],
            'trainingSeconds':time.time()-started,'benchmarkSha256':sha(frozen),'trainingPlanSha256':sha(plan_path),'initialCheckpointSha256':sha(initial),
            'architecture':'YOLOX-Nano .33/.25 depthwise','trainingProtocol':protocol}
        if result['f1']>best:best=result['f1'];best_epoch=epoch+1;torch.save(payload,OUT/'best.pt')
        torch.save(payload,OUT/'last.pt');(OUT/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2)+'\n')
        print('VALIDATION_V3',json.dumps({**row,'bestEpoch':best_epoch,'bestF1':best}),flush=True)
        if epoch+1>=8 and epoch+1-best_epoch>=protocol['earlyStoppingPatience']:break
    print('TRAIN_V3_FINISHED',json.dumps({'epochs':len(history),'selectedEpoch':best_epoch,'validationF1':best,'seconds':time.time()-started}),flush=True)

if __name__=='__main__':main()
