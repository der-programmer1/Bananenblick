"""Training-only occluded banana scenes with exact visible-instance boxes."""
from pathlib import Path
import json,math,random
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from train import BananaDataset

def visible_targets(masks,classes,original_areas):
    targets=[]
    for mask,cls,area in zip(masks,classes,original_areas):
        ys,xs=np.where(mask)
        if len(xs)<max(48,area*.18):continue
        x1,x2=xs.min(),xs.max()+1;y1,y2=ys.min(),ys.max()+1
        if min(x2-x1,y2-y1)<6:continue
        targets.append([cls,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1])
    return np.asarray(targets,np.float32).reshape(-1,5)

def compose_scene(sprites,background,rng=random,size=320):
    canvas=cv2.resize(background,(size,size)).copy();factor=size/320
    center=np.array([rng.uniform(110,210),rng.uniform(110,210)])*factor
    count=rng.randint(3,9);masks=[];classes=[];areas=[]
    base_angle=rng.choice([0,0,0,90,-90]);illumination=rng.uniform(.75,1.25)
    for sprite in rng.sample(sprites,min(count,len(sprites))):
        rgba=sprite['pixels'];height,width=rgba.shape[:2]
        scale=rng.uniform(55,235)*factor/max(width,height);angle=base_angle+rng.uniform(-35,35)
        matrix=cv2.getRotationMatrix2D((width/2,height/2),angle,scale)
        c,s=abs(matrix[0,0]),abs(matrix[0,1]);nw=max(1,math.ceil(height*s+width*c));nh=max(1,math.ceil(height*c+width*s))
        matrix[0,2]+=nw/2-width/2;matrix[1,2]+=nh/2-height/2
        transformed=cv2.warpAffine(rgba,matrix,(nw,nh),flags=cv2.INTER_LINEAR,borderValue=(0,0,0,0))
        x=round(center[0]+rng.gauss(0,48)*factor-nw/2);y=round(center[1]+rng.gauss(0,44)*factor-nh/2)
        xa,ya=max(0,x),max(0,y);xb,yb=min(size,x+nw),min(size,y+nh)
        if xb<=xa or yb<=ya:continue
        patch=transformed[ya-y:yb-y,xa-x:xb-x];alpha=patch[:,:,3:4].astype(np.float32)/255
        foreground=np.clip(patch[:,:,:3].astype(np.float32)*illumination,0,255)
        canvas[ya:yb,xa:xb]=(foreground*alpha+canvas[ya:yb,xa:xb]*(1-alpha)).astype(np.uint8)
        occupied=np.zeros((size,size),bool);occupied[ya:yb,xa:xb]=patch[:,:,3]>96
        for previous in masks:previous[occupied]=False
        masks.append(occupied);classes.append(sprite['class']);areas.append(int(occupied.sum()))
    return canvas,visible_targets(masks,classes,areas)

class SceneDataset(Dataset):
    def __init__(self,data,sprites,synthetic_count=600,real_repeat=6,size=448,plain_background_probability=0,real_crop_probability=0):
        self.size=size
        self.plain_background_probability=plain_background_probability
        self.base=BananaDataset(data,'train',True,mosaic=0,size=size,crop_probability=real_crop_probability)
        self.indices=[]
        for i,r in enumerate(self.base.records):
            self.indices.extend([i]*(real_repeat if isinstance(r['id'],str) and r['boxes'] else 1))
        self.synthetic_count=synthetic_count;root=Path(sprites)
        self.sprites=[dict(r,pixels=cv2.imread(str(root/r['file']),cv2.IMREAD_UNCHANGED)) for r in json.loads((root/'index.json').read_text())]
        if not self.sprites or any(r['pixels'] is None for r in self.sprites):raise ValueError('Prepare valid training-only sprites first')
        self.backgrounds=[im for im,r in zip(self.base.images,self.base.records) if not r['boxes']]
        if not self.backgrounds:raise ValueError('Real training negatives are required')
    def __len__(self):return len(self.indices)+self.synthetic_count
    def __getitem__(self,index):
        if index<len(self.indices):return self.base[self.indices[index]]
        background=random.choice(self.backgrounds)
        if self.plain_background_probability>0 and random.random()<self.plain_background_probability:
            # Single-fruit training mostly has plain backgrounds. Show multiple
            # fruits there as well so the background does not imply one object.
            shade=random.randint(100,255)
            background=np.full((self.size,self.size,3),shade,np.uint8)
        image,boxes=compose_scene(self.sprites,background,size=self.size)
        target=np.zeros((100,5),np.float32);target[:len(boxes)]=boxes
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float(),torch.from_numpy(target)
