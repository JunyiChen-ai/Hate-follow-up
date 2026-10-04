#!/usr/bin/env python3
"""Acquire actual window witnesses and a whole video entity/discourse graph."""
import argparse
import json
import logging
import socket
import sys
import time
import torch
from inputs import (ROOT,CACHE,DATASETS,selected_rows,windows_for,native_frames,
    ledger_content,link_content,validate)
from graph import VERSION,CONSTANTS,LEDGER_SYSTEM,LINK_SYSTEM,parse_ledger,parse_links,compile_graph,retrieve
from src.actual_video_frames import acquire_window_frames
from src.source_generation import generate,clock
from src.mllm_judge import Judge,MODEL
from src.video_inputs import load_asr


@torch.no_grad()
def acquire(j,row,segments,folder):
    torch.cuda.reset_peak_memory_stats();start=clock(j);before=j.forward_calls;before_vision=j.vision_calls
    source=acquire_window_frames(row,folder/'frames');windows=windows_for(row,segments,source,folder);overview=native_frames(row)
    ledgers=[]
    for window in windows:
        content,paths=ledger_content(overview,window)
        g=generate(j,ROOT,LEDGER_SYSTEM,content,paths,512)
        ledgers.append(dict(generation=g,parsed=parse_ledger(g,window)))
    local=compile_graph([l['parsed'] for l in ledgers],[]);links=[]
    for offset in range(0,len(windows),8):
        anchors=list(range(offset,min(offset+8,len(windows))));content=link_content(local,anchors)
        # Actual generated table, source-only size observation BEFORE GPU link
        # prefill. Save it; never truncate or modify the scientific table to fit.
        rendered=j.render([j.turn('system',LINK_SYSTEM),dict(role='user',content=content)],True)
        actual_size=len(j.tok.encode(rendered,add_special_tokens=False))
        g=generate(j,ROOT,LINK_SYSTEM,content,[],2048)
        assert len(g['input_tokens'])==actual_size
        links.append(dict(anchors=anchors,input_size_before_prefill=actual_size,generation=g,
            parsed=parse_links(g,local['nodes'],anchors)))
    graph=compile_graph([l['parsed'] for l in ledgers],[l['parsed'] for l in links])
    generations=[l['generation'] for l in ledgers+links]
    return dict(version=VERSION,constants=CONSTANTS,model=MODEL,dataset=row['dataset'],video_id=row['video_id'],
        duration=float(row['duration']),GT_read=False,host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),
        segments=[list(s) for s in segments],source=source,windows=windows,overview=overview,
        ledgers=ledgers,links=links,graph=graph,packets=[retrieve(graph,i,windows) for i in range(len(windows))],
        generation_seconds=sum(g['seconds'] for g in generations),standalone_seconds=clock(j)-start,
        actual_forwards=j.forward_calls-before,actual_vision_forwards=j.vision_calls-before_vision,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261005_m1_provenance'/('r1_extract_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,version=VERSION,constants=CONSTANTS,
        GT_read=False,smoke=a.smoke,torch=torch.__version__,transformers=transformers.__version__,
        code='experiments/20261005_m1_provenance/{graph,inputs,extract}.py + src/{actual_video_frames,source_generation}.py; sources2026-10-05',
        command='python -u '+' '.join(sys.argv))
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=j.vision_calls=0
    hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),
        j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    asr={ds:load_asr(ds) for ds in DATASETS};rows=selected_rows(a.smoke)
    for number,row in enumerate(rows,1):
        folder=CACHE/row['dataset']/row['video_id'];folder.mkdir(parents=True,exist_ok=True);path=folder/'metadata.json'
        segments=asr[row['dataset']].get(row['video_id'],[])
        if path.exists():m=json.loads(path.read_text());validate(m,row,segments,j);logging.info('%d/%d reuse %s/%s',number,len(rows),row['dataset'],row['video_id']);continue
        m=acquire(j,row,segments,folder);validate(m,row,segments,j)
        temporary=folder/'metadata.partial';temporary.write_text(json.dumps(m)+'\n');temporary.replace(path)
        logging.info('%d/%d %s/%s %.2fs nodes=%d edges=%d remote_windows=%d',number,len(rows),row['dataset'],row['video_id'],m['standalone_seconds'],
            len(m['graph']['nodes']),len(m['graph']['edges']),sum(bool(p['selected_windows']) for p in m['packets']))
    for hook in hooks:hook.remove()
    (CACHE/'PROVENANCE.md').write_text('# Temporal entity and discourse graph\n\n'
        'Generated source code experiments/20261005_m1_provenance/{graph,inputs,extract}.py and src/{actual_video_frames,source_generation}.py, sources2026-10-05.\n'
        'Frozen Qwen/Qwen3-VL-8B-Instruct, no labels/GT. Actual raw source paths/decoded PTS/witness PNGs and exact source ASR bodies, prompts/token grids/costs in per-video metadata.json.\n'
        'Native overview times nominal; ASR crops proportional, not verified word times. Structural validation is not semantic accuracy.\n'
        'Command '+cfg['command']+' in allocated lab Slurm on '+cfg['host']+', '+cfg['date']+'.\n')
    logging.info('EXTRACTION_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
