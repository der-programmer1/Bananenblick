"""Training-only foreground cutouts; never creates or scores real benchmark scenes."""
from pathlib import Path
import json
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    cv2.setNumThreads(1);cv2.setRNGSeed(20261110)
    data=ROOT/'work/data';out=ROOT/'work/sprites';out.mkdir(parents=True,exist_ok=True);records=[]
    for r in json.loads((data/'train.json').read_text()):
        if len({tuple(b) for b in r['boxes']})!=1:continue
        cls,cx,cy,bw,bh=r['boxes'][0];im=cv2.imread(str(data/r['file']));h,w=im.shape[:2]
        scale=min(1,320/max(w,h));im=cv2.resize(im,(round(w*scale),round(h*scale)));h,w=im.shape[:2]
        x1=max(1,int((cx-bw/2)*w)-3);y1=max(1,int((cy-bh/2)*h)-3)
        x2=min(w-1,int((cx+bw/2)*w)+3);y2=min(h-1,int((cy+bh/2)*h)+3)
        if x2-x1<10 or y2-y1<10:continue
        mask=np.zeros((h,w),np.uint8)
        try:cv2.grabCut(im,mask,(x1,y1,x2-x1,y2-y1),np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),3,cv2.GC_INIT_WITH_RECT)
        except cv2.error:continue
        fg=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)).astype(np.uint8)
        n,labels,stats,_=cv2.connectedComponentsWithStats(fg,8)
        if n<2:continue
        k=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));fg=(labels==k).astype(np.uint8)*255
        ys,xs=np.where(fg>0)
        if len(xs)<300:continue
        xa,xb=max(0,xs.min()-1),min(w,xs.max()+2);ya,yb=max(0,ys.min()-1),min(h,ys.max()+2)
        alpha=cv2.GaussianBlur(fg,(3,3),.5);rgba=np.dstack([im,alpha])[ya:yb,xa:xb]
        name=f"{r['id']:04d}.png";cv2.imwrite(str(out/name),rgba)
        records.append({'id':r['id'],'file':name,'class':cls,'source':r['source'],'sourceGroup':r['group'],'width':rgba.shape[1],'height':rgba.shape[0]})
        if len(records)%50==0:print('sprites',len(records),flush=True)
    (out/'index.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print('TRAIN_ONLY_SPRITES',len(records),flush=True)

if __name__=='__main__':main()
