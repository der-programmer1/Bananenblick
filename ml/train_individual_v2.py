"""Full-backbone training for individual bananas; selects on validation only."""
from pathlib import Path
import argparse,copy,hashlib,json,math,random,subprocess,time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import create_model
from train import BananaDataset
from scene_augmentation import SceneDataset
from postprocess import decode,DEFAULTS

ROOT=Path(__file__).resolve().parent;SITE=ROOT.parent;DATA=ROOT/'work/individual-v2';OUT=ROOT/'work/training-v2';SIZE=448
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

@torch.inference_mode()
def validate(model,dataset):
    # Reproducing training must not depend on a later release's calibrated floor.
    DEFAULTS.update({'detection':.25,'classification':.6,'margin':.15,'nmsIou':.65})
    model.eval();cases=[];loader=DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0);index=0
    for images,_ in loader:
        outputs=model(images).numpy()
        for output in outputs:
            r=dataset.records[index];index+=1
            detections=decode(output.reshape(-1).tolist(),SIZE,r['width'],r['height'],thresholds={'detection':.25,'classification':.6,'margin':.15,'nmsIou':.65})
            cases.append({'id':str(r['id']),'scene':r.get('scene','single-fruit'),'boxes':r['boxes'],
                'detections':[{k:v for k,v in d.items() if not k.startswith('_')} for d in detections]})
    (OUT/'validation-predictions.json').write_text(json.dumps(cases,ensure_ascii=False))
    subprocess.run(['node','--import','tsx','scripts/evaluate-recommendations.ts',str(OUT/'validation-predictions.json'),str(OUT/'validation-report.json')],cwd=SITE,check=True,stdout=subprocess.DEVNULL)
    report=json.loads((OUT/'validation-report.json').read_text());p,r=report['precision'],report['recall'];report['f1']=2*p*r/max(p+r,1e-12)
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stop-after',type=int);args=parser.parse_args()
    frozen=ROOT/'individual-benchmark-v2.json';manifest=json.loads(frozen.read_text());assert manifest['realBunchScenes']==30 and manifest['negativeScenes']==30
    assert manifest['annotationSha256']==sha(ROOT/'individual_annotations_v2.json')
    assert manifest['trainingAnnotationSha256']==sha(ROOT/'individual_training_v2.json')
    for rs in manifest['splits'].values():
        for r in rs:assert sha(DATA/r['file'])==r['sha256_file']
    seed=20261112;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True)
    initial=ROOT/'work/base-v0.1.pt'
    if not initial.exists():
        with initial.open('wb') as f:subprocess.run(['git','show','6b682c4:ml/artifacts/banana-yolox-nano.pt'],cwd=SITE,stdout=f,check=True)
    model=create_model(initial);ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    dataset=SceneDataset(DATA,ROOT/'work/sprites',synthetic_count=600,real_repeat=6,size=SIZE)
    validation=BananaDataset(DATA,'val',size=SIZE)
    loader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=0,drop_last=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.00035,weight_decay=.0001)
    started=time.time();steps=0;best=-1;history=[];best_epoch=0;epochs=manifest['protocol']['maxEpochs'];patience=manifest['protocol']['earlyStoppingPatience']
    print('TRAIN_V2_STARTED',json.dumps({'benchmarkSha256':sha(frozen),'inputSize':SIZE,'uniqueTrainingImages':len(dataset.base.records),
        'samplesPerEpoch':len(dataset),'validationImages':len(validation),'syntheticTrainingOnly':600,'fullBackboneFromEpoch':1,'epochsMax':epochs}),flush=True)
    for epoch in range(min(epochs,args.stop_after or epochs)):
        model.train();model.head.use_l1=epoch>=epochs-4;sums=[];start=time.time()
        for batch,(images,targets) in enumerate(loader):
            fraction=(epoch+batch/len(loader))/epochs;warmup=min(1,(steps+1)/len(loader))
            lr=(.000035+.000315*.5*(1+math.cos(math.pi*fraction)))*warmup
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);loss=model(images,targets)['total_loss']
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();steps+=1
            decay=min(.995,1-1/(steps+1))
            with torch.no_grad():
                current=model.state_dict()
                for k,v in ema.state_dict().items():
                    if v.is_floating_point():v.mul_(decay).add_(current[k],alpha=1-decay)
                    else:v.copy_(current[k])
            sums.append(float(loss.detach()))
            if batch%30==0:print(json.dumps({'epoch':epoch+1,'batch':batch,'batches':len(loader),'loss':sums[-1],'elapsedSeconds':round(time.time()-started,1)}),flush=True)
        metrics=validate(ema,validation);row={'epoch':epoch+1,'loss':float(np.mean(sums)),'seconds':time.time()-start,'validation':{k:v for k,v in metrics.items() if k!='scenes'}};history.append(row)
        payload={'model':ema.state_dict(),'epoch':epoch+1,'seed':seed,'inputSize':SIZE,'epochsRequested':epochs,'validation':row['validation'],
            'trainingSeconds':time.time()-started,'benchmarkSha256':sha(frozen),'initialCheckpointSha256':sha(initial),
            'architecture':'YOLOX-Nano .33/.25 depthwise','trainingProtocol':manifest['protocol']}
        if metrics['f1']>best:
            best=metrics['f1'];best_epoch=epoch+1;torch.save(payload,OUT/'best.pt')
        torch.save(payload,OUT/'last.pt');(OUT/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2))
        print('VALIDATION_V2',json.dumps({**row,'bestEpoch':best_epoch,'bestF1':best}),flush=True)
        if epoch+1>=8 and epoch+1-best_epoch>=patience:
            print('EARLY_STOP_VALIDATION_PLATEAU',flush=True);break
    print('TRAIN_V2_FINISHED',json.dumps({'epochs':len(history),'selectedEpoch':best_epoch,'validationF1':best,'seconds':time.time()-started}),flush=True)

if __name__=='__main__':main()
