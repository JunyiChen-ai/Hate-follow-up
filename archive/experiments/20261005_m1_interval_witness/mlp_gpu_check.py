"""Actual small source-parent greedy replay before resuming long source prefill."""
import json
import logging
import os
import socket
import torch
from inputs import ROOT,CACHE,DATASETS,selected_rows,validate,parent_content,parent_record
from interface import SPEC,catalog,coverage,write_record,compile_record,partition
from src.structured_source_generation import generate
from src.qwen3_mlp_memory import chunked_mlp
from src.mllm_judge import Judge,MODEL


def main():
    out=ROOT/'runs/20261005_m1_interval_witness/prefill_mlp_fix/gpu';out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=j.vision_calls=0
    hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    picks={ds:[] for ds in DATASETS}
    for row in selected_rows(True):
        m=json.loads((CACHE/row['dataset']/row['video_id']/'metadata.json').read_text());validate(m,row,[tuple(s) for s in m['segments']])
        records={}
        for node in partition(len(m['windows'])):
            lo,hi=node['lo'],node['hi']
            if hi-lo==1:records[node['id']]=dict(**node,start=m['windows'][lo]['start'],end=m['windows'][lo]['end'],**m['leaves'][lo]['record']);continue
            saved=m['parents'][node['id']];children=[records[c] for c in node['children']]
            content=parent_content(node,children,m['windows']);n=len(saved['generation']['input_tokens'])
            if 4096<n<=16000:picks[row['dataset']].append((n,row,node,content,saved,m))
            records[node['id']]=parent_record(node,saved['record'],children,m['windows'])
    results=[]
    for ds in DATASETS:
        assert picks[ds],(ds,'no actual parent crossing4096 available')
        n,row,node,content,saved,m=max(picks[ds],key=lambda t:t[0]);lo,hi=node['lo'],node['hi']
        sources={h:r for w in m['windows'][lo:hi] for h,r in catalog(w).items()}
        writer=lambda stream:write_record(stream,sources)
        direct=generate(j,ROOT,SPEC['parent_system'],content,[],512,writer)
        assert direct['tokens']==saved['generation']['tokens'] and direct['events']==saved['generation']['events']
        with chunked_mlp(j.model.model,4096):chunk=generate(j,ROOT,SPEC['parent_system'],content,[],512,writer)
        assert chunk['tokens']==direct['tokens'] and chunk['events']==direct['events']
        assert compile_record(chunk,sources,coverage(m['windows'][lo:hi]))==saved['record']
        results.append(dict(dataset=ds,video_id=row['video_id'],parent=node['id'],input_tokens=n,
            native_source_tokens_exact=True,chunk_source_tokens_exact=True,direct=direct,chunk=chunk))
        logging.info('%s %s prefix%d tokens%d exact',ds,row['video_id'],n,len(chunk['tokens']))
    for hook in hooks:hook.remove()
    (out/'summary.json').write_text(json.dumps(dict(host=socket.gethostname(),GT_read=False,pass_all=True,checks=results),indent=2)+'\n')
    logging.info('MLP_GPU_PASS')


if __name__=='__main__':main()
