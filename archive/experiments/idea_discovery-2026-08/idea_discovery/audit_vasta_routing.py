#!/usr/bin/env python3
"""Audit conditional transcript value in VASTA's visual-disagreement state."""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scripts.label_free_adapt.evaluate import binary_intervals,temporal_iou
TAUS=(-20,-16,-12,-8,-4,0,4,8,12,16,20)
def load(p):return {(r['dataset'],r['video_id']):r for r in map(json.loads,p.open())}
def split(k):return 'dev' if int(hashlib.sha256(f'method-hash-v1\0{k[0]}\0{k[1]}'.encode()).hexdigest(),16)%2==0 else 'holdout'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--extent',type=Path,required=True);ap.add_argument('--a08',type=Path,required=True);ap.add_argument('--a12',type=Path,required=True);ap.add_argument('--text',type=Path,required=True);ap.add_argument('--gt-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();extent=load(a.extent);x8=load(a.a08);x12=load(a.a12);xt=load(a.text);keys=sorted(set(extent)&set(x8)&set(x12)&set(xt));gt={}
 for d in sorted({k[0] for k in keys}):
  z=np.load(a.gt_dir/f'{d}.npz',allow_pickle=True)
  for i,v in enumerate(z['video_ids']):
   if (d,str(v)) in keys:gt[(d,str(v))]=binary_intervals(np.asarray(z['y4'][i],np.int8))
 rows=[]
 for k in keys:
  e8=bool(x8[k].get('intervals'));e12=bool(x12[k].get('intervals'));state='both_positive' if e8 and e12 else 'both_negative' if not e8 and not e12 else 'disagree';p=extent[k]['intervals'][0];iou=max((temporal_iou(tuple(p[:2]),g) for g in gt[k]),default=0.)
  rows.append({'dataset':k[0],'video_id':k[1],'split':split(k),'state':state,'text_log_odds':float(xt[k]['log_odds']),'has_gt':bool(gt[k]),'iou':iou})
 out={'n':len(rows),'states':{},'thresholds':{}}
 for s in ('all','dev','holdout'):
  rs=rows if s=='all' else [x for x in rows if x['split']==s];out['states'][s]={}
  for d in ('ALL',*sorted({x['dataset'] for x in rs})):
   z=rs if d=='ALL' else [x for x in rs if x['dataset']==d];out['states'][s][d]={'n':len(z),'counts':dict(Counter(x['state'] for x in z)),'disagreement_text_pass@-12':sum(x['state']=='disagree' and x['text_log_odds']>=-12 for x in z),'disagreement_correct_veto@-12':sum(x['state']=='disagree' and x['text_log_odds']< -12 and x['iou']<.5 for x in z),'disagreement_wrong_veto@-12':sum(x['state']=='disagree' and x['text_log_odds']< -12 and x['iou']>=.5 for x in z)}
 for tau in TAUS:
  out['thresholds'][str(tau)]={}
  for s in ('dev','holdout'):
   z=[x for x in rows if x['split']==s and x['state']=='disagree'];veto=[x for x in z if x['text_log_odds']<tau];out['thresholds'][str(tau)][s]={'disagreement_n':len(z),'pass':len(z)-len(veto),'veto':len(veto),'correct_veto_iou@.5':sum(x['iou']<.5 for x in veto),'wrong_veto_iou@.5':sum(x['iou']>=.5 for x in veto),'net_veto':sum(x['iou']<.5 for x in veto)-sum(x['iou']>=.5 for x in veto)}
 a.out.write_text(json.dumps({'summary':out,'detail':rows},indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
