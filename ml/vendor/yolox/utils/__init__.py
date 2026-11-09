# Minimal subset from official YOLOX utils (Apache-2.0).
import torch

def meshgrid(*tensors):
    return torch.meshgrid(*tensors, indexing="ij")

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if xyxy:
        tl = torch.max(bboxes_a[:,None,:2], bboxes_b[:,:2])
        br = torch.min(bboxes_a[:,None,2:], bboxes_b[:,2:])
        area_a = torch.prod(bboxes_a[:,2:]-bboxes_a[:,:2], 1)
        area_b = torch.prod(bboxes_b[:,2:]-bboxes_b[:,:2], 1)
    else:
        tl = torch.max(bboxes_a[:,None,:2]-bboxes_a[:,None,2:]/2, bboxes_b[:,:2]-bboxes_b[:,2:]/2)
        br = torch.min(bboxes_a[:,None,:2]+bboxes_a[:,None,2:]/2, bboxes_b[:,:2]+bboxes_b[:,2:]/2)
        area_a = torch.prod(bboxes_a[:,2:], 1)
        area_b = torch.prod(bboxes_b[:,2:], 1)
    area_i = torch.prod(br-tl,2)*(tl<br).type(tl.dtype).prod(2)
    return area_i/(area_a[:,None]+area_b-area_i+1e-12)

def cxcywh2xyxy(bboxes):
    bboxes[:,0:2] -= bboxes[:,2:4]*.5
    bboxes[:,2:4] += bboxes[:,0:2]
    return bboxes

def visualize_assign(*args,**kwargs):
    raise NotImplementedError("Assignment visualization is not part of this CPU training subset.")
