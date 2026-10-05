#!/usr/bin/env python3
"""Acquire one source-executed program per native window with a single Qwen."""
import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
import torch
from PIL import Image
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.mllm_judge import Judge,MODEL
from src.mllm_renderer import cpu_renderer
from src.video_inputs import load_asr
from inputs import DATASETS,CACHE,selected_rows,sources,validate
from program import CACHE_VERSION,CONSTANTS,canonical,PLANNER_SYSTEM,API,PLANNER_END,MODULE_SYSTEM,SCOPE_QUESTION,ACTION_QUESTION,parse_chunk,execute


def tick():torch.cuda.synchronize();return time.perf_counter()


@torch.no_grad()
def generate(j,system,content,paths,max_tokens):
    """Fresh cache; FP32 greedy readout, same numerical procedure for all calls."""
    before=j.forward_calls;start=tick();msgs=[j.turn('system',system),dict(role='user',content=content)]
    rendered=j.render(msgs,True);images=[Image.open(ROOT/p).convert('RGB') for p in paths]
    try:enc=j.encode(rendered,images)
    finally:
        for im in images:im.close()
    j.model.model.rope_deltas=None
    out=j.model.model(**j.model_inputs(enc),use_cache=True)
    cache=out.past_key_values;h=out.last_hidden_state[0,-1];del out
    if not hasattr(j,'generation_W32'):j.generation_W32=j.model.get_output_embeddings().weight.float()
    gen=[];stopped=False
    for _ in range(max_tokens):
        logits=h.float()@j.generation_W32.T
        if j.softcap:logits=torch.tanh(logits/j.softcap)*j.softcap
        nxt=int(logits.argmax())
        if nxt in j.eos_ids:stopped=True;break
        gen.append(nxt);h=j._step(cache,[nxt])
    del cache
    return dict(text=j.tok.decode(gen,skip_special_tokens=True).strip(),tokens=gen,truncated=not stopped,
        seconds=tick()-start,actual_forwards=j.forward_calls-before,prompt=rendered,system=system,
        max_tokens=max_tokens,image_paths=paths,input_tokens=enc['input_ids'][0].tolist(),
        image_grid=enc['image_grid_thw'].tolist() if 'image_grid_thw' in enc else [])


def planner_content(source,requested):
    content=[]
    for f in source['frames']:
        content.extend([dict(type='text',text=f'frame {f["id"]}, nominal t={f["time"]}s\n'),dict(type='image')])
    table={**source,'requested_windows':requested}
    content.append(dict(type='text',text=API+'\n'+canonical(table)+'\n'+PLANNER_END))
    return content


def module_content(kind,packet):
    content=[dict(type='text',text=canonical(packet)+'\n'+(SCOPE_QUESTION if kind=='scope' else ACTION_QUESTION))]
    if kind=='action':content.insert(0,dict(type='image'))
    return content


def validate_generations(j,metadata):
    """Bind saved generations to literal prompts, original pixels and tokenizer IDs."""
    allgens=[]
    for chunk in metadata['chunks']:
        g=chunk['generation'];content=planner_content(metadata['source'],chunk['requested_windows'])
        allgens.append((g,PLANNER_SYSTEM,content,metadata['frame_paths'],CONSTANTS['planner_tokens']))
    for w in metadata['windows']:
        for g in w['module_generations']:
            paths=[metadata['frame_paths'][g['packet']['sources'][0]['value']['ref']['frame']]] if g['kind']=='action' else []
            allgens.append((g,MODULE_SYSTEM,module_content(g['kind'],g['packet']),paths,CONSTANTS['module_tokens']))
    for g,system,content,paths,limit in allgens:
        rendered=j.render([j.turn('system',system),dict(role='user',content=content)],True)
        assert g['system']==system and g['prompt']==rendered and g['image_paths']==paths and g['max_tokens']==limit
        images=[Image.open(ROOT/p).convert('RGB') for p in paths]
        try:encoded=j.encode(rendered,images)
        finally:
            for image in images:image.close()
        assert encoded['input_ids'][0].tolist()==g['input_tokens']
        assert (encoded['image_grid_thw'].tolist() if 'image_grid_thw' in encoded else [])==g['image_grid']
        assert j.tok.decode(g['tokens'],skip_special_tokens=True).strip()==g['text'] and len(g['tokens'])<=limit
        assert g['actual_forwards']==1+len(g['tokens'])
    assert metadata['actual_forwards']==sum(g['actual_forwards'] for g,*_ in allgens)


@torch.no_grad()
def acquire(j,row,segments):
    source,windows,paths=sources(row,segments);torch.cuda.reset_peak_memory_stats();start=tick();before=j.forward_calls
    chunks=[];plans={}
    for offset in range(0,len(windows),CONSTANTS['planner_windows']):
        requested=windows[offset:offset+CONSTANTS['planner_windows']]
        g=generate(j,PLANNER_SYSTEM,planner_content(source,requested),paths,CONSTANTS['planner_tokens'])
        parsed,rejected=parse_chunk(g['text'],requested,g['truncated']);plans.update(parsed)
        chunks.append(dict(requested_windows=requested,generation=g,plans={str(k):v for k,v in parsed.items()},rejected=rejected))
    executed=[]
    for w in windows:
        generations=[]
        def perceive(kind,packet):
            input_paths=[paths[packet['sources'][0]['value']['ref']['frame']]] if kind=='action' else []
            g=generate(j,MODULE_SYSTEM,module_content(kind,packet),input_paths,CONSTANTS['module_tokens'])
            g.update(kind=kind,packet=packet);generations.append(g);return g['text']
        result=execute(source,w,plans[w['id']],perceive)
        executed.append(dict(window=w,execution=result,module_generations=generations))
    return dict(cache_version=CACHE_VERSION,constants=CONSTANTS,model=MODEL,dataset=row['dataset'],video_id=row['video_id'],
        duration=float(row['duration']),manifest_video_path=row['video_path'],GT_read=False,host=socket.gethostname(),
        date=time.strftime('%Y-%m-%d'),source=source,frame_paths=paths,chunks=chunks,windows=executed,
        standalone_seconds=tick()-start,actual_forwards=j.forward_calls-before,peak_GiB=torch.cuda.max_memory_allocated()/2**30)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261005_m1_program'/('r1_extract_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    cfg=dict(date=time.strftime('%Y-%m-%d'),host=socket.gethostname(),model=MODEL,cache_version=CACHE_VERSION,constants=CONSTANTS,
        torch=torch.__version__,transformers=transformers.__version__,GT_read=False,smoke=a.smoke,
        code='experiments/20261005_m1_program/{program,inputs,extract}.py + src/{video_inputs,mllm_judge}.py; sources2026-10-05',command='python -u '+' '.join(sys.argv))
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=0
    hook=j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1))
    asr={ds:load_asr(ds) for ds in DATASETS};rows=selected_rows(a.smoke)
    for i,row in enumerate(rows,1):
        path=CACHE/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        segments=asr[row['dataset']].get(row['video_id'],[])
        if path.exists():m=json.loads(path.read_text());validate(m,row,segments);validate_generations(j,m);logging.info('%d/%d reuse %s/%s',i,len(rows),row['dataset'],row['video_id']);continue
        m=acquire(j,row,segments);validate(m,row,segments);validate_generations(j,m)
        tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(m)+'\n');tmp.replace(path)
        logging.info('%d/%d %s/%s %.2fs programs=%d calls=%d',i,len(rows),row['dataset'],row['video_id'],m['standalone_seconds'],
            sum(w['execution']['status']=='valid' for w in m['windows']),sum(len(w['execution']['calls']) for w in m['windows']))
    hook.remove()
    (CACHE/'PROVENANCE.md').write_text('# Temporal evidence programs\n\n'
        'Sources2026-10-05: experiments/20261005_m1_program/{program,inputs,extract}.py. Frozen Qwen/Qwen3-VL-8B-Instruct.\n'
        'Native frames_k20 and timestamped Whisper segments; filenames provide nominal frame times, not verified decoded PTS.\n'
        'No labels/GT. Actual source paths/text/coordinates, literal prompts, outputs/tokens and generation hosts/dates in each JSON.\n'
        'Generated via '+cfg['command']+' in allocated lab Slurm; current host '+cfg['host']+', date '+cfg['date']+'.\n')
    logging.info('EXTRACTION_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
