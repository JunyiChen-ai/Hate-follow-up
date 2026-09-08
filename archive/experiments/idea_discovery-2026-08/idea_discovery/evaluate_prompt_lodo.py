#!/usr/bin/env python3
"""Leave-one-dataset-out prompt selection for exploratory OCQ-TTA prompts."""
from pathlib import Path
import json,sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).parent));from prequential_write_gate_pilot import ecdf,interp
ROOT=Path(__file__).resolve().parents[2];GT=Path('/home/jehc223/Retrieval-hate/data/gt/frame_gt_4fps')
PROMPTS={'baseprompt':ROOT/'results/idea_discovery/cpo_tta/chunk50_all','mainq':ROOT/'results/idea_discovery/prompt_query/mainq','hostile':ROOT/'results/idea_discovery/prompt_query/hostile','visual':ROOT/'results/idea_discovery/prompt_query/visual','abuse':ROOT/'results/idea_discovery/prompt_query/abuse'};DS=['HateMM','HateClipSeg','MHC']
scores={p:{} for p in PROMPTS}
for d in DS:
 z=np.load(GT/(d+'.npz'),allow_pickle=True);gt={str(v):np.asarray(y,int) for v,s,y in zip(z['video_ids'],z['split'],z['y4']) if str(s)=='test' and len(np.unique(y))==2};common=set(gt)
 for root in PROMPTS.values():common&={x.stem for x in (root/d).glob('*.npy')}
 gt={v:gt[v] for v in common}
 for p,root in PROMPTS.items():scores[p][d]=float(np.mean([roc_auc_score(y,interp(ecdf(np.load(root/d/(v+'.npy'))),len(y))) for v,y in gt.items()]))
folds={}
for held in DS:
 train=[d for d in DS if d!=held];chosen=max(PROMPTS,key=lambda p:np.mean([scores[p][d] for d in train]));folds[held]={'selected_prompt':chosen,'train_macro':float(np.mean([scores[chosen][d] for d in train])),'held_within_roc':scores[chosen][held]}
result={'protocol':'LODO over datasets with mixed-label test videos; exploratory because candidate prompt set was conceived during test-informed mechanism debugging','scores':scores,'folds':folds,'lodo_macro':float(np.mean([x['held_within_roc'] for x in folds.values()]))};out=ROOT/'results/idea_discovery/prompt_query/lodo.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
