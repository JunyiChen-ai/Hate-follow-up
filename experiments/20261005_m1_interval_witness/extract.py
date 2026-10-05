#!/usr/bin/env python3
"""One complete source-tree/reconciliation acquisition per original video."""
import argparse
import json
import logging
import socket
import sys
import time
import torch
from inputs import ROOT,CACHE,DATASETS,selected_rows,windows_for,overview_for,local_content,parent_content,parent_record,validate
from interface import SPEC,VERSION,CONSTANTS,catalog,coverage,write_record,compile_record,partition,repair_plan,final_tree
from src.actual_video_frames import acquire_window_frames
from src.structured_source_generation import generate
from src.source_generation import clock
from src.mllm_judge import Judge,MODEL
from src.qwen3_mlp_memory import chunked_mlp
from src.video_inputs import load_asr


@torch.no_grad()
def acquire(j,row,segments,folder):
    torch.cuda.reset_peak_memory_stats();start=clock(j);before=j.forward_calls;before_vision=j.vision_calls
    source=acquire_window_frames(row,folder/'frames');windows=windows_for(row,segments,source,folder);overview=overview_for(row)
    leaves=[];records={};parents={}
    # Leaves precede parent calls. Complete source catalog is always current data.
    for w in windows:
        sources=catalog(w);content,paths=local_content(overview,w)
        g=generate(j,ROOT,SPEC['leaf_system'],content,paths,512,lambda s:write_record(s,sources));p=compile_record(g,sources,coverage([w]))
        leaves.append(dict(generation=g,record=p))
    for node in partition(len(windows)):
        lo,hi=node['lo'],node['hi']
        if hi-lo==1:records[node['id']]=dict(**node,start=windows[lo]['start'],end=windows[lo]['end'],**leaves[lo]['record']);continue
        children=[records[c] for c in node['children']];sources={h:r for w in windows[lo:hi] for h,r in catalog(w).items()}
        content=parent_content(node,children,windows)
        # Parent source catalogs can exceed 50k tokens. Only tokenwise MLP row
        # batches change; attention, complete input, KV and greedy grammar stay.
        with chunked_mlp(j.model.model,4096):
            g=generate(j,ROOT,SPEC['parent_system'],content,[],512,lambda s:write_record(s,sources))
        p=compile_record(g,sources,coverage(windows[lo:hi]))
        parents[node['id']]=dict(generation=g,record=p);records[node['id']]=parent_record(node,p,children,windows)
    original=[l['record'] for l in leaves];plan=repair_plan(windows,original,parents);repaired=list(original);repairs=[]
    for target in plan:
        w=windows[target['leaf']];sources=catalog(w);content,paths=local_content(overview,w,target['context'])
        g=generate(j,ROOT,SPEC['repair_system'],content,paths,512,lambda s:write_record(s,sources));p=compile_record(g,sources,coverage([w]))
        repairs.append(dict(target=target,generation=g,record=p));repaired[w['i']]=p
    generations=[l['generation'] for l in leaves+list(parents.values())+repairs]
    return dict(version=VERSION,constants=CONSTANTS,spec=SPEC,model=MODEL,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),
        GT_read=False,host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),segments=[list(s) for s in segments],source=source,windows=windows,overview=overview,
        leaves=leaves,parents=parents,repair_plan=plan,repairs=repairs,final_leaves=repaired,final_tree=final_tree(windows,repaired,parents),
        generation_seconds=sum(g['seconds'] for g in generations),standalone_seconds=clock(j)-start,
        actual_forwards=j.forward_calls-before,actual_vision_forwards=j.vision_calls-before_vision,peak_GiB=torch.cuda.max_memory_allocated()/2**30)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261005_m1_interval_witness'/('r1_extract_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,version=VERSION,constants=CONSTANTS,spec=SPEC,GT_read=False,smoke=a.smoke,
        torch=torch.__version__,transformers=transformers.__version__,parent_mlp_chunk_tokens=4096,code='experiments/20261005_m1_interval_witness/{interface,inputs,extract}.py + stable src; sources2026-10-05',command='python -u '+' '.join(sys.argv))
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=j.vision_calls=0
    hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    asr={ds:load_asr(ds) for ds in DATASETS};rows=selected_rows(a.smoke)
    for number,row in enumerate(rows,1):
        folder=CACHE/row['dataset']/row['video_id'];folder.mkdir(parents=True,exist_ok=True);path=folder/'metadata.json';segments=asr[row['dataset']].get(row['video_id'],[])
        if path.exists():m=json.loads(path.read_text());validate(m,row,segments,j);logging.info('%d/%d source reuse %s/%s',number,len(rows),row['dataset'],row['video_id']);continue
        m=acquire(j,row,segments,folder);validate(m,row,segments,j);temporary=folder/'metadata.partial';temporary.write_text(json.dumps(m)+'\n');temporary.replace(path)
        logging.info('%d/%d %s/%s %.2fs parents=%d repairs=%d',number,len(rows),row['dataset'],row['video_id'],m['standalone_seconds'],len(m['parents']),len(m['repairs']))
    for hook in hooks:hook.remove()
    (CACHE/'PROVENANCE.md').write_text('# Interval-owned source witnesses\n\nGenerated by experiments/20261005_m1_interval_witness/extract.py + shared actual_video_frames/structured_source_generation, sources2026-10-05.\n'
        'Frozen Qwen/Qwen3-VL-8B-Instruct, no labels. Raw video paths/decoded PTS/PNG sources, ASR proportional crops, overview nominal times, all prompts/IDs/grids/costs per metadata.json. Structural source validity is not semantic proof.\nCommand '+cfg['command']+' in allocated Slurm on '+cfg['host']+', '+cfg['date']+'.\n')
    logging.info('EXTRACTION_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
