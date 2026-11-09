#!/usr/bin/env python3
"""Reproducible CPU training and validation-only checkpoint selection."""
from pathlib import Path
import argparse,copy,json,math,os,random,time
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from model import create_model
ROOT=Path(__file__).resolve().parent
SIZE=320

def prepare(im,boxes,augment=False,size=SIZE):
    h,w=im.shape[:2];ratio=min(size/h,size/w)
    if augment:ratio*=random.uniform(.78,1.)
    nh,nw=int(h*ratio),int(w*ratio)
    oy=random.randint(0,size-nh) if augment else 0
    ox=random.randint(0,size-nw) if augment else 0
    canvas=np.full((size,size,3),114,dtype=np.uint8)
    canvas[oy:oy+nh,ox:ox+nw]=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
    b=np.asarray(boxes,dtype=np.float32).reshape(-1,5).copy()
    if len(b):
        b[:,1]=b[:,1]*w*ratio+ox;b[:,2]=b[:,2]*h*ratio+oy
        b[:,3]*=w*ratio;b[:,4]*=h*ratio
    return canvas,b

class BananaDataset(Dataset):
    def __init__(self,data,split,augment=False,mosaic=.25):
        self.data=Path(data);self.records=json.loads((self.data/f'{split}.json').read_text());self.augment=augment;self.mosaic=mosaic
        self.images=[cv2.imread(str(self.data/r['file'])) for r in self.records]
        if any(im is None for im in self.images):raise ValueError('Cannot decode a prepared image')
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        if self.augment and random.random()<self.mosaic:
            im=np.full((SIZE,SIZE,3),114,dtype=np.uint8);bs=[]
            for k,j in enumerate([i]+random.choices(range(len(self)),k=3)):
                tile,b=prepare(self.images[j],self.records[j]['boxes'],True,SIZE//2)
                ox=(k%2)*(SIZE//2);oy=(k//2)*(SIZE//2)
                im[oy:oy+SIZE//2,ox:ox+SIZE//2]=tile
                if len(b):b[:,1]+=ox;b[:,2]+=oy;bs.append(b)
            b=np.concatenate(bs) if bs else np.zeros((0,5),dtype=np.float32)
        else:im,b=prepare(self.images[i],self.records[i]['boxes'],self.augment)
        if self.augment:
            if random.random()<.5:
                im=im[:,::-1].copy()
                if len(b):b[:,1]=SIZE-b[:,1]
            # Hue is kept within +/-2 degrees to preserve the ripeness target.
            hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,0]=(hsv[:,:,0]+random.uniform(-1,1))%180
            hsv[:,:,1]*=random.uniform(.9,1.1);hsv[:,:,2]*=random.uniform(.8,1.2)
            im=cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8),cv2.COLOR_HSV2BGR)
        target=np.zeros((100,5),dtype=np.float32);target[:min(len(b),100)]=b[:100]
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))).float(),torch.from_numpy(target)

def iou(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    tl=np.maximum(a[:,None,:2],b[None,:,:2]);br=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(br-tl,0).prod(2);area_a=np.maximum(a[:,2:]-a[:,:2],0).prod(1);area_b=np.maximum(b[:,2:]-b[:,:2],0).prod(1)
    return inter/(area_a[:,None]+area_b[None,:]-inter+1e-9)

def postprocess(output,threshold=.01,nms=.45,image_shape=None):
    a=np.asarray(output,dtype=np.float32)
    cls=np.argmax(a[:,5:],axis=1);prob=a[np.arange(len(a)),cls+5];score=a[:,4]*prob
    boxes=np.concatenate([a[:,:2]-a[:,2:4]/2,a[:,:2]+a[:,2:4]/2],axis=1)
    valid=np.isfinite(score)&np.isfinite(boxes).all(1)
    if image_shape:
        width,height=image_shape;ratio=min(SIZE/width,SIZE/height)
        boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,width*ratio)
        boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,height*ratio)
        valid&=(boxes[:,2]-boxes[:,0]>=3*ratio)&(boxes[:,3]-boxes[:,1]>=3*ratio)
    keep=np.flatnonzero((score>=threshold)&valid);keep=keep[np.argsort(score[keep])[::-1]]
    selected=[]
    while len(keep):
        k=keep[0];selected.append(k)
        keep=keep[1:][iou(boxes[k:k+1],boxes[keep[1:]])[0]<=nms]
        if len(selected)>=40:break
    return [{'box':boxes[k].tolist(),'class':int(cls[k]),'score':float(score[k]),'classProbability':float(prob[k]),'classMargin':float(np.sort(a[k,5:])[-1]-np.sort(a[k,5:])[-2]),'objectness':float(a[k,4])} for k in selected]

def gt_from_target(target):
    a=target.numpy();a=a[a[:,3]>0]
    return [{'class':int(v[0]),'box':[float(v[1]-v[3]/2),float(v[2]-v[4]/2),float(v[1]+v[3]/2),float(v[2]+v[4]/2)]} for v in a]

@torch.inference_mode()
def predict(model,loader):
    model.eval();preds=[];gts=[]
    for images,targets in loader:
        outputs=model(images).numpy()
        for out,target in zip(outputs,targets):
            image=loader.dataset.images[len(preds)]
            preds.append(postprocess(out,image_shape=(image.shape[1],image.shape[0])));gts.append(gt_from_target(target))
    return preds,gts

def metrics(preds,gts,threshold=.25):
    # Rows: actual class; columns: predicted 0,1,2, missed. Classification
    # confusion uses greedy, class-agnostic IoU>=.5 matching.
    confusion=np.zeros((3,4),int);visible=np.zeros((3,5),int);fp=np.zeros(3,int);tp=np.zeros(3,int);fn=np.zeros(3,int);loc_matched=np.zeros(3,int);unknown=np.zeros(3,int)
    for predictions,truth in zip(preds,gts):
        ps=sorted([p for p in predictions if p['score']>=threshold],key=lambda p:p['score'],reverse=True)
        overlaps=iou([p['box'] for p in ps],[g['box'] for g in truth]);used=set()
        for pi,p in enumerate(ps):
            candidates=[j for j in range(len(truth)) if j not in used and overlaps[pi,j]>=.5]
            if not candidates:fp[p['class']]+=1;continue
            j=max(candidates,key=lambda j:overlaps[pi,j]);used.add(j);actual=truth[j]['class'];guess=p['class'];confusion[actual,guess]+=1;loc_matched[actual]+=1
            uncertain=p['score']<.60 or p['classMargin']<.15
            visible[actual,3 if uncertain else guess]+=1
            if uncertain:unknown[actual]+=1
            if actual==guess:tp[actual]+=1
            else:fp[guess]+=1;fn[actual]+=1
        for j,g in enumerate(truth):
            if j not in used:fn[g['class']]+=1;confusion[g['class'],3]+=1;visible[g['class'],4]+=1
    classes=[];aps=[]
    for c in range(3):
        total=sum(g['class']==c for gs in gts for g in gs)
        ranked=sorted([(p['score'],i,p) for i,ps in enumerate(preds) for p in ps if p['class']==c],key=lambda x:x[0],reverse=True)
        used=set();hits=[]
        for _,i,p in ranked:
            choices=[(j,g) for j,g in enumerate(gts[i]) if g['class']==c and (i,j) not in used]
            if choices:
                ov=iou([p['box']],[g['box'] for _,g in choices])[0];k=int(np.argmax(ov))
                if ov[k]>=.5:used.add((i,choices[k][0]));hits.append(1);continue
            hits.append(0)
        hits=np.array(hits);cum=np.cumsum(hits);rec=cum/max(total,1);prec=cum/np.maximum(np.arange(len(hits))+1,1)
        ap=float(np.mean([float(prec[rec>=r].max()) if np.any(rec>=r) else 0. for r in np.linspace(0,1,101)]));aps.append(ap)
        precision=float(tp[c]/max(tp[c]+fp[c],1));recall=float(tp[c]/max(tp[c]+fn[c],1))
        classes.append({'class':['unreif','reif','überreif'][c],'support':total,'truePositive':int(tp[c]),'falsePositive':int(fp[c]),'falseNegative':int(fn[c]),'precision':precision,'recall':recall,'localizationRecall':float(loc_matched[c]/max(total,1)),'wrongRipeness':int(confusion[c,:3].sum()-confusion[c,c]),'missed':int(confusion[c,3]),'uncertain':int(unknown[c]),'visibleTruePositive':int(visible[c,c]),'visibleRecall':float(visible[c,c]/max(total,1)),'ap50':ap})
    micro_p=float(tp.sum()/max(tp.sum()+fp.sum(),1));micro_r=float(tp.sum()/max(tp.sum()+fn.sum(),1))
    return {'threshold':threshold,'iouThreshold':.5,'nmsIou':.45,'apCandidateThreshold':.01,'maxDetectionsPerImage':40,'mAP50':float(np.mean(aps)),'precision':micro_p,'recall':micro_r,'f1':2*micro_p*micro_r/max(micro_p+micro_r,1e-9),'visibleRecall':float(sum(visible[c,c] for c in range(3))/max(sum(len(gs) for gs in gts),1)),'uncertainCount':int(unknown.sum()),'perClass':classes,'confusion':{'rows':['unreif','reif','überreif'],'columns':['unreif','reif','überreif','nicht erkannt'],'counts':confusion.tolist()},'visibleConfusion':{'rows':['unreif','reif','überreif'],'columns':['unreif','reif','überreif','unsicher','nicht erkannt'],'counts':visible.tolist()}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'work/data');p.add_argument('--output',type=Path,default=ROOT/'work/training');p.add_argument('--pretrained',type=Path,required=True);p.add_argument('--epochs',type=int,default=16);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--threads',type=int,default=3);p.add_argument('--seed',type=int,default=20261109);a=p.parse_args()
    torch.set_num_threads(a.threads);cv2.setNumThreads(1);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True,exist_ok=True)
    train=BananaDataset(a.data,'train',True);val=BananaDataset(a.data,'val')
    loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0,drop_last=True);vloader=DataLoader(val,batch_size=a.batch_size,num_workers=0)
    model=create_model(a.pretrained);ema=copy.deepcopy(model).eval()
    for q in ema.parameters():q.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.0001)
    best=-1;history=[];started=time.time();steps=0
    for epoch in range(a.epochs):
        model.train();train.mosaic=.25 if epoch<a.epochs-3 else 0
        model.head.use_l1=epoch>=a.epochs-3
        # First two epochs update the head while retaining pretrained features.
        for q in model.backbone.parameters():q.requires_grad_(epoch>=2)
        if epoch<2:model.backbone.eval()
        sums=[];t=time.time()
        for batch,(images,targets) in enumerate(loader):
            fraction=(epoch+batch/len(loader))/a.epochs
            warmup=min(1.,(epoch+batch/len(loader)+.1)/1.)
            lr=(.00007+.00063*.5*(1+math.cos(math.pi*fraction)))*warmup
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);out=model(images,targets);loss=out['total_loss']
            if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss at epoch {epoch} batch {batch}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();steps+=1
            # EMA of parameters and BN statistics; validation uses EMA only.
            decay=min(.99,1-1/(steps+1))
            with torch.no_grad():
                current_state=model.state_dict()
                for k,v in ema.state_dict().items():
                    other=current_state[k]
                    if v.is_floating_point():v.mul_(decay).add_(other,alpha=1-decay)
                    else:v.copy_(other)
            sums.append(float(loss.detach()))
            if batch%20==0:print(json.dumps({'epoch':epoch+1,'batch':batch,'batches':len(loader),'loss':sums[-1],'elapsed':round(time.time()-started,1)}),flush=True)
        preds,gts=predict(ema,vloader);vm=metrics(preds,gts,.25)
        row={'epoch':epoch+1,'loss':float(np.mean(sums)),'validation':vm,'seconds':time.time()-t};history.append(row)
        payload={'model':ema.state_dict(),'epoch':epoch+1,'seed':a.seed,'epochsRequested':a.epochs,'inputSize':SIZE,'validation':vm,'trainingSeconds':time.time()-started,'architecture':'YOLOX-Nano .33/.25 depthwise','pretrainedSha256':__import__('hashlib').file_digest(a.pretrained.open('rb'),'sha256').hexdigest()}
        if vm['mAP50']>best:
            best=vm['mAP50'];torch.save(payload,a.output/'best.pt')
        torch.save(payload,a.output/'last.pt')
        (a.output/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2))
        print(json.dumps({'epoch':epoch+1,'loss':row['loss'],'valAP50':vm['mAP50'],'valRecall':vm['recall'],'valPrecision':vm['precision'],'seconds':row['seconds'],'best':best}),flush=True)
    # Tune the display detection threshold on validation only after checkpoint selection.
    payload=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False);ema.load_state_dict(payload['model']);preds,gts=predict(ema,vloader)
    candidates=[metrics(preds,gts,t) for t in [.15,.2,.25,.3,.35,.4,.5]]
    selected=max(candidates,key=lambda m:(m['f1'],m['precision'],m['threshold']))
    (a.output/'validation-selection.json').write_text(json.dumps({'selectedThreshold':selected['threshold'],'metrics':selected,'candidates':candidates,'checkpointEpoch':payload['epoch'],'rule':'highest validation mAP50 chooses checkpoint, then highest validation F1 chooses threshold; test split untouched'},ensure_ascii=False,indent=2))
    print('TRAINING_FINISHED',json.dumps({'seconds':time.time()-started,'checkpointEpoch':payload['epoch'],'threshold':selected['threshold']}),flush=True)

if __name__=='__main__':main()
