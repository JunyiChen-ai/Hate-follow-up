#!/usr/bin/env python3
"""Explicit interface B acquisition: constrained source programs, original executor."""
import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.mllm_judge import Judge,MODEL
from src.mllm_renderer import cpu_renderer
from src.video_inputs import load_asr
from handle_inputs import DATASETS,CACHE,selected_rows,sources,validate,plans_for
from handle_interface import (CACHE_VERSION,CONSTANTS,PLANNER_SYSTEM,MODULE_SYSTEM,
    planner_content,module_content,write_plans,write_module)
from handle_decoder import generate,validate_generation
from program import execute


def tick():torch.cuda.synchronize();return time.perf_counter()


def validate_generations(j,m):
    generations=[]
    for c in m['chunks']:
        requested=c['requested_windows'];g=c['generation']
        writer=lambda s,requested=requested:write_plans(s,m['source'],requested)
        generations.append(g)
        validate_generation(j,ROOT,g,PLANNER_SYSTEM,planner_content(m['source'],requested),m['frame_paths'],CONSTANTS['planner_tokens'],writer)
    for w in m['windows']:
        for g in w['module_generations']:
            kind,packet=g['kind'],g['packet']
            paths=[m['frame_paths'][packet['sources'][0]['value']['ref']['frame']]] if kind=='action' else []
            writer=lambda s,kind=kind,packet=packet:write_module(s,kind,packet)
            generations.append(g)
            validate_generation(j,ROOT,g,MODULE_SYSTEM,module_content(kind,packet),paths,CONSTANTS['module_tokens'],writer)
    assert m['actual_forwards']==sum(g['actual_forwards'] for g in generations)


@torch.no_grad()
def acquire(j,row,segments):
    source,windows,paths=sources(row,segments);torch.cuda.reset_peak_memory_stats();start=tick();before=j.forward_calls
    chunks=[];plans={}
    for offset in range(0,len(windows),CONSTANTS['planner_windows']):
        requested=windows[offset:offset+CONSTANTS['planner_windows']]
        writer=lambda s:write_plans(s,source,requested)
        g=generate(j,ROOT,PLANNER_SYSTEM,planner_content(source,requested),paths,CONSTANTS['planner_tokens'],writer)
        parsed=plans_for(source,requested,g);plans.update(parsed)
        chunks.append(dict(requested_windows=requested,generation=g,plans={str(k):v for k,v in parsed.items()}))
    executed=[]
    for w in windows:
        generations=[]
        def perceive(kind,packet):
            input_paths=[paths[packet['sources'][0]['value']['ref']['frame']]] if kind=='action' else []
            g=generate(j,ROOT,MODULE_SYSTEM,module_content(kind,packet),input_paths,CONSTANTS['module_tokens'],
                lambda s:write_module(s,kind,packet))
            g.update(kind=kind,packet=packet);generations.append(g);return g['text']
        result=execute(source,w,plans[w['id']],perceive)
        executed.append(dict(window=w,execution=result,module_generations=generations))
    return dict(cache_version=CACHE_VERSION,constants=CONSTANTS,model=MODEL,dataset=row['dataset'],video_id=row['video_id'],
        duration=float(row['duration']),manifest_video_path=row['video_path'],GT_read=False,host=socket.gethostname(),
        date=time.strftime('%Y-%m-%d'),source=source,frame_paths=paths,chunks=chunks,windows=executed,
        standalone_seconds=tick()-start,actual_forwards=j.forward_calls-before,peak_GiB=torch.cuda.max_memory_allocated()/2**30)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--capacity-check',action='store_true');a=ap.parse_args()
    assert not (a.smoke and a.capacity_check)
    out=ROOT/'runs/20261005_m1_program'/('r1_handles_extract_'+('capacity' if a.capacity_check else 'smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    cfg=dict(date=time.strftime('%Y-%m-%d'),host=socket.gethostname(),model=MODEL,cache_version=CACHE_VERSION,constants=CONSTANTS,
        torch=torch.__version__,transformers=transformers.__version__,GT_read=False,smoke=a.smoke,capacity_check=a.capacity_check,
        code='experiments/20261005_m1_program/handle_{interface,inputs,decoder,extract}.py + original program.py; sources2026-10-05',command='python -u '+' '.join(sys.argv))
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=0
    hook=j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1))
    asr={ds:load_asr(ds) for ds in DATASETS};rows=selected_rows(a.smoke)
    if a.capacity_check:
        rows=[r for r in rows if (r['dataset'],r['video_id'])==('HateMM','non_hate_video_134')]
        assert len(rows)==1
    for i,row in enumerate(rows,1):
        path=CACHE/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        segments=asr[row['dataset']].get(row['video_id'],[])
        if path.exists():
            m=json.loads(path.read_text());validate(m,row,segments);validate_generations(j,m)
            logging.info('%d/%d reuse %s/%s',i,len(rows),row['dataset'],row['video_id']);continue
        m=acquire(j,row,segments);validate(m,row,segments);validate_generations(j,m)
        tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(m)+'\n');tmp.replace(path)
        logging.info('%d/%d %s/%s %.2fs programs=%d calls=%d',i,len(rows),row['dataset'],row['video_id'],m['standalone_seconds'],
            sum(w['execution']['status']=='valid' for w in m['windows']),sum(len(w['execution']['calls']) for w in m['windows']))
    hook.remove();(CACHE/'PROVENANCE.md').write_text('# Source-handle temporal evidence programs\n\n'
        'Interface B sources2026-10-05, experiments/20261005_m1_program/handle_*.py plus original program.py. '
        'Frozen Qwen/Qwen3-VL-8B-Instruct, no labels/GT. Native cached frames and ASR segments; frame times nominal, character times proportional. '
        'Exact source/choices/compiler/decoder tokens/grammar/cost/hosts in each JSON. Generated by '+cfg['command']+
        ' in allocated lab Slurm on '+cfg['host']+', '+cfg['date']+'.\n')
    logging.info('EXTRACTION_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
