"""CPU-capable official YOLOX-Nano, with a three-class head."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
import torch
from torch import nn
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

def create_model(pretrained=None):
    model = YOLOX(YOLOPAFPN(.33,.25,depthwise=True),YOLOXHead(3,.25,depthwise=True))
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eps = 1e-3
            layer.momentum = .03
    model.head.initialize_biases(.01)
    if pretrained:
        payload = torch.load(pretrained,map_location='cpu',weights_only=False)
        weights = payload.get('model',payload)
        state = model.state_dict()
        valid = {k:v for k,v in weights.items() if k in state and state[k].shape==v.shape}
        # Transfer the pretrained banana class (COCO category index 46) to each
        # ripeness logit. Fine tuning learns the three separate ripeness classes.
        for scale in range(3):
            for kind in ['weight','bias']:
                key=f'head.cls_preds.{scale}.{kind}'
                if key in weights and weights[key].shape[0]==80:
                    valid[key]=weights[key][46:47].repeat((3,)+(1,)*(weights[key].ndim-1))
        model.load_state_dict(valid,strict=False)
    return model
