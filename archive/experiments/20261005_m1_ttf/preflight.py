#!/usr/bin/env python3
"""Actual full333 native image/grid/source validation, CPU only and no labels."""
import json
import os
import socket
from measure import ROOT,selected_rows
from src.video_inputs import frame_paths,load_asr
from src.mllm_renderer import cpu_renderer
from analyze import position_renderer
from src.stance_cache import positions
import torch

def main():
    torch.set_num_threads(4);j=position_renderer();asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};result=[]
    for i,row in enumerate(selected_rows(),1):
        frames=frame_paths(row['dataset'],row['video_id'],20);segments=asr[row['dataset']].get(row['video_id'],[])
        msgs,paths=j.prefix_messages(frames,segments);text,enc=j.encode_prefix(msgs,paths);gg=enc['image_grid_thw'].tolist()
        assert 18<=len(frames)<=20 and len(gg)==len(frames) and all(g==gg[0] for g in gg)
        assert all(g[0]==1 and g[1]%2==0 and g[2]%2==0 for g in gg)
        assert int((enc['input_ids']==j.image_token_id).sum())==sum(t*h*w//4 for t,h,w in gg)
        p,delta=positions(j,enc['input_ids'],enc['image_grid_thw'])
        assert p.shape==(3,1,enc['input_ids'].shape[1])
        result.append(dict(dataset=row['dataset'],video_id=row['video_id'],frames=len(frames),grid=gg[0],prefix_tokens=enc['input_ids'].shape[1],logical_offset=int(p.max())+1))
        if i%25==0:print(i,'/333',flush=True)
    out=ROOT/'runs/20261005_m1_ttf/full_input_preflight';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(host=socket.gethostname(),GT_read=False,coverage=len(result),pass_all=True,
        largest=sorted(result,key=lambda r:r['prefix_tokens'],reverse=True)[:10],rows=result),indent=2)+'\n')
    print('PREFLIGHT_PASS coverage=333',flush=True)

if __name__=='__main__':main()
