"""Small shared side regressor using only two-step warmup information."""

import torch
import math
from torch import nn
import torch.nn.functional as F


def side_features(clean, reference, text_pool, bbox, source_wh, grid_hw=(64,64)):
    h, w = grid_hw
    x0,y0,x1,y1 = torch.tensor(bbox, device=clean.device, dtype=torch.float32) / torch.tensor(
        [source_wh[0],source_wh[1],source_wh[0],source_wh[1]], device=clean.device)
    energy = (clean.float()-reference.float()).square().mean(-1).sqrt().reshape(h,w)
    scale = energy.mean().clamp_min(1e-5)
    energy = (energy/scale).clamp(0,20)
    yy,xx = torch.meshgrid((torch.arange(h,device=clean.device)+0.5)/h,
                          (torch.arange(w,device=clean.device)+0.5)/w,indexing="ij")
    core=(xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1)
    def stats(mask):
        a=energy[mask]
        return torch.stack([a.mean(),a.max()]) if a.numel() else torch.zeros(2,device=clean.device)
    text=F.adaptive_avg_pool1d(text_pool.float().reshape(1,1,-1),16).flatten()
    text=text/(text.square().mean().sqrt().clamp_min(1e-5))
    available=torch.stack([x0,y0,1-x1,1-y1])
    distances=[x0-xx,y0-yy,xx-x1,yy-y1]
    spans=[(yy>=y0-0.04)&(yy<=y1+0.04),(xx>=x0-0.04)&(xx<=x1+0.04)]*2
    rows=[]
    for side in range(4):
        bands=[stats((distances[side]>=a)&(distances[side]<b)&spans[side])
               for a,b in zip([0,.04,.08,.16],[.04,.08,.16,.32])]
        geo=torch.stack([x0,y0,x1,y1,x1-x0,y1-y0,available[side]])
        orientation=F.one_hot(torch.tensor(side,device=clean.device),4).float()
        rows.append(torch.cat(bands+[stats(core),geo,orientation,text]))
    return torch.stack(rows),available


class SideExpansionHead(nn.Module):
    def __init__(self, feature_dim=37):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(feature_dim,16),nn.SiLU(),nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias,-3.2)

    def forward(self,x,available):
        return torch.minimum(F.softplus(self.net(x).squeeze(-1)),available)


def margins_to_box(margins,bbox,source_wh):
    w,h=source_wh
    d=margins.detach().float().cpu().tolist()
    return [max(0,bbox[0]-d[0]*w),max(0,bbox[1]-d[1]*h),
            min(w,bbox[2]+d[2]*w),min(h,bbox[3]+d[3]*h)]


class ExpansionPredictor:
    def __init__(self,checkpoint,device="cuda"):
        self.bundle=torch.load(checkpoint,map_location=device,weights_only=True)
        self.model=SideExpansionHead(self.bundle["feature_dim"]).to(device).eval()
        self.model.load_state_dict(self.bundle["state_dict"])
        self.device=device
        self.last=None

    @torch.no_grad()
    def __call__(self,clean,reference,text_pool,case,grid_hw=(64,64)):
        x,available=side_features(clean,reference,text_pool,case["bbox"],case["source_wh"],grid_hw)
        z=(x-self.bundle["mean"])/self.bundle["std"]
        raw=self.model(z,available)
        margin=torch.minimum(raw+self.bundle["safety_pad"],available)
        box=margins_to_box(margin,case["bbox"],case["source_wh"])
        fraction=(box[2]-box[0])*(box[3]-box[1])/(case["source_wh"][0]*case["source_wh"][1])
        fallback=fraction>=0.8
        self.last=dict(raw_margins=raw.cpu().tolist(),calibrated_margins=margin.cpu().tolist(),
                       envelope=box,area_fraction=fraction,fallback_full=fallback,
                       uncertainty="empirical_calibration_max_positive_residual_not_probability")
        return box,fallback


class CenteredBudgetPredictor:
    def __init__(self,predictor):
        self.predictor=predictor
        self.last=None

    def __call__(self,clean,reference,text_pool,case,grid_hw=(64,64)):
        box,fallback=self.predictor(clean,reference,text_pool,case,grid_hw)
        h,w=grid_hw;sw,sh=case["source_wh"];old=case["bbox"]
        left,top=math.floor(box[0]*w/sw),math.floor(box[1]*h/sh)
        right,bottom=math.ceil(box[2]*w/sw),math.ceil(box[3]*h/sh)
        width,height=right-left,bottom-top
        oldleft,oldtop=math.floor(old[0]*w/sw),math.floor(old[1]*h/sh)
        oldright,oldbottom=math.ceil(old[2]*w/sw),math.ceil(old[3]*h/sh)
        l=max(0,min(w-width,round((oldleft+oldright-width)/2)))
        t=max(0,min(h-height,round((oldtop+oldbottom-height)/2)))
        result=[l*sw/w,t*sh/h,(l+width)*sw/w,(t+height)*sh/h]
        self.last=dict(learned_prediction=self.predictor.last,centered_envelope=result,
                       matched_token_rectangle=[width,height])
        return result,fallback
