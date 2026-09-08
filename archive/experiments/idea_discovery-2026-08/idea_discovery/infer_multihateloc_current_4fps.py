#!/usr/bin/env python3
"""Run frozen local MultiHateLoc checkpoints on the current 4 fps cohort."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/reproduction_baselines/multihateloc'))
import data as mdata
from model import MultiHateLoc
from train import predict
from scripts.label_free_adapt.evaluate import binary_intervals

MAP={'HateMM':'hatemm','HateClipSeg':'hateclipseg','MHC':'mhclip_en','MHC_zh':'mhclip_zh'}
def resize(x,n):
 x=np.asarray(x,float);return x.copy() if len(x)==n else np.interp(np.linspace(0,1,n),np.linspace(0,1,len(x)),x)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint-root',type=Path,required=True);ap.add_argument('--gt-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='cuda');a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 audit={}
 with a.out.open('w') as h:
  for dataset,corpus in MAP.items():
   gt=np.load(a.gt_dir/f'{dataset}.npz',allow_pickle=True);lengths={str(v):len(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
   ids=[v for v in sorted(lengths) if all(Path(mdata.feature_path(mod,corpus,v)).exists() for mod in mdata.MODALITIES)]
   ds=mdata.MultiModalDataset(corpus,ids,labels={v:0 for v in ids});loader=DataLoader(ds,batch_size=32,shuffle=False,collate_fn=mdata.collate,num_workers=2)
   log=json.loads((a.checkpoint_root/corpus/'train_log.json').read_text());cfg=log['args'];model=MultiHateLoc({m:mdata.FEATURE_DIMS[m] for m in mdata.MODALITIES},hidden=cfg['hidden'],embed=cfg['embed'],dropout=cfg['dropout'],k_proportion=cfg['k_proportion'],temperature=cfg['temperature']).to(a.device)
   model.load_state_dict(torch.load(a.checkpoint_root/corpus/'model.pt',map_location=a.device,weights_only=True));frames,_,_=predict(model,loader,a.device)
   for vid in ids:
    n=lengths[vid];union=resize(frames[vid]['score_union'],n)>=.5;intervals=[list(x)+[1.] for x in binary_intervals(union,4.)]
    for branch in ('fused','dms'):
     scores=resize(frames[vid][f'score_{branch}'],n);row={'schema_version':'1.0','method':f'multihateloc_frozen_{branch}_current4fps','dataset':dataset,'video_id':vid,'duration':n/4,'native_rate':4.,'score_curve':scores.tolist(),'intervals':intervals,'modality_evidence':{},'raw':{'gt_access':False,'checkpoint':str((a.checkpoint_root/corpus/'model.pt').resolve()),'training_supervision':'video_level_hate_labels','evaluation_manifest_access':'ids_and_timeline_length_only','source_grid':'1fps','target_grid':'4fps_linear'},'calls':0,'seed':cfg['seed'],'error':None};h.write(json.dumps(row,separators=(',',':'))+'\n')
   audit[dataset]={'target':len(lengths),'predicted':len(ids),'missing':sorted(set(lengths)-set(ids))}
 print(json.dumps(audit,indent=2,sort_keys=True))
if __name__=='__main__':main()
