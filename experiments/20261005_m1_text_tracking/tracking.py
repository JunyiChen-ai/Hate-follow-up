"""CPU pixel correspondence only; source words never enter tracking decisions."""
import json
from pathlib import Path
import cv2
import numpy as np

SPEC=json.loads((Path(__file__).parent/'spec.json').read_text())
cv2.setNumThreads(1)
cv2.setRNGSeed(SPEC['seed'])


def tracking_image(rgb):
    h,w=rgb.shape[:2];scale=min(1.,SPEC['tracking_long_side']/max(h,w))
    width=max(1,round(w*scale));height=max(1,round(h*scale))
    small=cv2.resize(rgb,(width,height),interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small,cv2.COLOR_RGB2GRAY),[width/w,height/h]


def crop(rgb,box):
    x0,y0,x1,y1=box;h,w=rgb.shape[:2]
    assert all(isinstance(x,int) for x in box) and 0<=x0<x1<=w and 0<=y0<y1<=h
    return rgb[y0:y1,x0:x1].copy()


def points(gray,box,scales):
    x0,y0,x1,y1=box;sx,sy=scales;h,w=gray.shape
    bounds=[max(0,int(np.floor(x0*sx))),max(0,int(np.floor(y0*sy))),
            min(w,int(np.ceil(x1*sx))),min(h,int(np.ceil(y1*sy)))]
    mask=np.zeros_like(gray);a,b,c,d=bounds;mask[b:d,a:c]=255
    return cv2.goodFeaturesToTrack(gray,maxCorners=SPEC['corner_max'],
        qualityLevel=SPEC['corner_quality'],minDistance=SPEC['corner_distance'],
        mask=mask,blockSize=SPEC['corner_block'],useHarrisDetector=False)


def cut(previous,current):
    if previous.shape!=current.shape:return True,dict(reason='shape_change')
    size=tuple(SPEC['cut_shape'])
    a=cv2.resize(previous,size,interpolation=cv2.INTER_AREA).astype(np.float32)
    b=cv2.resize(current,size,interpolation=cv2.INTER_AREA).astype(np.float32)
    difference=float(np.abs(a-b).mean()/255)
    return difference>SPEC['cut_mae_threshold'],dict(thumbnail_mae=difference)


def appearance(reference,current):
    size=tuple(SPEC['appearance_shape'])
    a=cv2.resize(reference,size,interpolation=cv2.INTER_AREA).astype(np.float32)
    b=cv2.resize(current,size,interpolation=cv2.INTER_AREA).astype(np.float32)
    mae=float(np.abs(a-b).mean()/255)
    aa=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float64).reshape(-1)
    bb=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY).astype(np.float64).reshape(-1)
    aa-=aa.mean();bb-=bb.mean();va=float(np.mean(aa*aa));vb=float(np.mean(bb*bb))
    valid=va>SPEC['variance_epsilon'] and vb>SPEC['variance_epsilon']
    ncc=float(np.dot(aa,bb)/np.sqrt(np.dot(aa,aa)*np.dot(bb,bb))) if valid else None
    accepted=valid and ncc>=SPEC['ncc_minimum'] and mae<=SPEC['appearance_mae_max']
    return bool(accepted),dict(ncc=ncc,mae=mae,reference_variance=va,current_variance=vb)


def advance(previous,current,box,reference):
    """Fresh corners each step; retain old support only through accepted pixels."""
    if previous.shape!=current.shape:return None,dict(reason='shape_change')
    pg,scales=tracking_image(previous);cg,current_scales=tracking_image(current)
    assert scales==current_scales
    pp=points(pg,box,scales)
    if pp is None or len(pp)<SPEC['minimum_valid_points']:
        return None,dict(reason='insufficient_corners',initial_points=0 if pp is None else len(pp))
    kwargs=dict(winSize=tuple(SPEC['lk_window']),maxLevel=SPEC['lk_max_level'],
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,SPEC['lk_iterations'],SPEC['lk_epsilon']),
        flags=0,minEigThreshold=SPEC['lk_min_eigen'])
    qq,forward,_=cv2.calcOpticalFlowPyrLK(pg,cg,pp,None,**kwargs)
    if qq is None:return None,dict(reason='forward_unavailable',initial_points=len(pp))
    back,backward,_=cv2.calcOpticalFlowPyrLK(cg,pg,qq,None,**kwargs)
    if back is None:return None,dict(reason='backward_unavailable',initial_points=len(pp))
    fb=np.linalg.norm(back.reshape(-1,2)-pp.reshape(-1,2),axis=1)
    q=qq.reshape(-1,2);h,w=cg.shape
    valid=(forward[:,0]!=0)&(backward[:,0]!=0)&np.isfinite(q).all(1)&np.isfinite(fb)
    valid&=(fb<=SPEC['fb_max_pixels'])&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    count=int(valid.sum());e=dict(initial_points=len(pp),valid_points=count,
        forward_backward_pixels=[float(v) if np.isfinite(v) else None for v in fb],
        valid_mask=valid.tolist())
    if count<SPEC['minimum_valid_points'] or count/len(pp)<SPEC['minimum_valid_fraction']:
        return None,{**e,'reason':'forward_backward_rejected'}
    delta=np.median(q[valid]-pp.reshape(-1,2)[valid],axis=0)
    dx=int(round(float(delta[0])/scales[0]));dy=int(round(float(delta[1])/scales[1]))
    next_box=[box[0]+dx,box[1]+dy,box[2]+dx,box[3]+dy]
    hh,ww=current.shape[:2];e.update(translation_tracking_pixels=delta.tolist(),translation_original_pixels=[dx,dy],candidate_box=next_box)
    if not (0<=next_box[0]<next_box[2]<=ww and 0<=next_box[1]<next_box[3]<=hh):
        return None,{**e,'reason':'out_of_bounds'}
    accepted,visual=appearance(reference,crop(current,next_box));e['appearance']=visual
    return (next_box,{**e,'reason':'accepted'}) if accepted else (None,{**e,'reason':'appearance_rejected'})


def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=x*y
    return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)
