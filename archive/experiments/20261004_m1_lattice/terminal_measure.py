#!/usr/bin/env python3
"""R3 paired native and end-state path reading; actual inputs only, no GT."""
import argparse
import copy
import json
import logging
import math
import os
import socket
import sys
import time
import numpy as np
import torch
from extract import ROOT,CACHE,DATASETS,selected_rows,validate
from lattice import CACHE_VERSION,CONSTANTS
from terminal_graph import VERSION,positions
from terminal_reader import compile_branch,structural,unit_reference
from measure import record,tick
from control_measure import current_context
from src.mllm_judge import Judge,MODEL,yesno_question
from src.mllm_renderer import cpu_renderer
from src.stance_cache import build,margin
from src.video_inputs import frame_paths,load_asr


def validate_bundle(row,bundle,metadata,segments,renderer,smoke):
    validate(metadata,row,segments)
    assert bundle['version']==VERSION and bundle['GT_read'] is False and bundle['segments']==[list(s) for s in segments]
    base,new=bundle['base'],bundle['optimized'];ww=metadata['windows']
    ctx,current=current_context(renderer,row,segments,base['extra']['stance']);assert current==bundle['native_input']
    n=len(current['token_ids']);start=base['extra']['stance_cache_logical_start']
    assert n==base['extra']['stance_cache_tokens'] and current['prefix_tokens']==base['extra']['prefix_tokens']
    count=diagnostics=0
    for name,r in [('base',base),('optimized',new)]:
        assert (r['dataset'],r['video_id'])==(row['dataset'],row['video_id']) and r['error'] is None
        assert r['duration']==float(row['duration']) and r['native_rate']==4
        assert r['extra']['stance']==('Yes' if r['extra']['z_video']>0 else 'No')
        for k in ('z_video','stance','prefix_tokens','stance_cache_tokens','stance_cache_logical_start'):
            assert r['extra'][k]==base['extra'][k]
        wins=r['extra']['windows'];assert len(wins)==len(ww)
        for w,m in zip(wins,ww):
            assert (w['i'],w['start'],w['end'])==(m['i'],m['start'],m['end'])
            assert ('z_speech' in w)==(bool(m['native_body'].strip()) if name=='base' else m['available'])
            assert w['z_visual']==base['extra']['windows'][m['i']]['z_visual']
            assert w['z']==max(w['z_visual'],w.get('z_speech',float('-inf')))
        idx=np.clip(((np.arange(math.ceil(r['duration']*4))+.5)/4//8).astype(int),0,len(wins)-1)
        assert np.array_equal(r['score_curve'],np.asarray([w['z'] for w in wins])[idx])
        assert np.isfinite(r['score_curve']).all() and np.isfinite(r['extra']['z_video'])
        assert r['calls']==3+len(wins)+sum('z_speech' in w for w in wins)
    assert len(bundle['traces'])==len(ww)
    for w,m,t in zip(new['extra']['windows'],ww,bundle['traces']):
        if not m['available']:assert t==dict(available=False,reason=m['reason']);continue
        count+=1;ids,expected=compile_branch(renderer,ctx,m,len(ww))
        assert t['ids']==ids and all(t[k]==v for k,v in expected.items())
        assert t['logical_positions']==positions(expected['graph'],len(expected['head_tokens']),len(expected['tail_tokens']),start)
        assert t['margin']==w['z_speech'] and t['available'] and t['mask_forward'] and t['terminal_key_mask_forward']
        assert t['prefix_tokens']==n and t['prefix_logical_start']==start and t['attention_dtype']=='torch.bfloat16'
        if 'smoke_checks' in t:
            c=t['smoke_checks'];assert c['cloned_cache_exact'] is True and c['diagnostic_forwards']==3
            assert c['cloned_cache_margin']==t['margin']
            unit=copy.deepcopy(m);unit['beams']=[dict(text=m['confusion']['onebest'],score=0.) for _ in range(5)]
            unit_ids,_=compile_branch(renderer,ctx,unit,len(ww));assert unit_ids==c['unit_token_ids']
            assert c['unit_difference']==c['unit_graph_margin']-c['unit_reference_margin'] and c['unit_tolerance']==.01
            assert abs(c['unit_difference'])<=.01;diagnostics+=3
    c=bundle['checks'];assert diagnostics==3*int(smoke and count>0)==c['diagnostic_forwards']
    assert c['actual_forwards']==base['calls']+count+diagnostics
    assert c['preprocessing_seconds']==metadata['standalone_seconds'] and c['input_actual_forwards']==metadata['actual_forwards']
    assert c['peak_GiB']>=metadata['peak_GiB']
    assert base['extra']['standalone_seconds']==c['prefix_seconds']+c['visual_seconds']+c['reference_speech_seconds']
    assert new['extra']['standalone_seconds']==metadata['standalone_seconds']+c['prefix_seconds']+c['visual_seconds']+c['new_speech_seconds']
    assert all(np.isfinite(v) and v>=0 for k,v in c.items() if k.endswith('seconds') or k=='peak_GiB')


@torch.no_grad()
def read_video(j,row,segments,metadata,smoke):
    first=j.forward_calls;torch.cuda.reset_peak_memory_stats();start=tick()
    cache,ctx=build(j,frame_paths(row['dataset'],row['video_id'],20),segments);prefix_seconds=tick()-start
    native_ids=[];ii=0
    for token in j.tok.encode(ctx['head'],add_special_tokens=False):
        if token==j.image_token_id:native_ids.extend([token]*ctx['image_counts'][ii]);ii+=1
        else:native_ids.append(token)
    assert ii==len(ctx['image_counts']) and len(native_ids)==cache.get_seq_length()
    native_input=dict(files=[str(p) for p in ctx['files']],token_ids=native_ids,image_counts=ctx['image_counts'],
        prefix_head=ctx['head'],prefix_tokens=ctx['prefix_tokens'])
    ww=metadata['windows'];visual=[];speech=[];new_speech=[];traces=[]
    visual_seconds=reference_seconds=new_seconds=diagnostic_seconds=0.;diagnostics=0;checked=False
    for m in ww:
        start=tick();visual.append(margin(j,cache,ctx,yesno_question(m['i'],len(ww),m['start'],m['end'],m['native_body'],'visual')))
        visual_seconds+=tick()-start
        if m['native_body'].strip():
            start=tick();speech.append(margin(j,cache,ctx,yesno_question(m['i'],len(ww),m['start'],m['end'],m['native_body'],'speech')))
            reference_seconds+=tick()-start
        else:speech.append(None)
    for m in ww:
        if not m['available']:new_speech.append(None);traces.append(dict(available=False,reason=m['reason']));continue
        start=tick();z,t=structural(j,cache,ctx,m,len(ww));new_seconds+=tick()-start;t['available']=True
        if smoke and not checked:
            start=tick();clone=copy.deepcopy(cache)
            try:replay,_=structural(j,clone,ctx,m,len(ww));assert replay==z
            finally:del clone
            unit=copy.deepcopy(m);text=m['confusion']['onebest'];unit['beams']=[dict(text=text,score=0.) for _ in range(5)]
            gz,_=structural(j,cache,ctx,unit,len(ww));ids,_=compile_branch(j,ctx,unit,len(ww))
            sz,_=unit_reference(j,cache,ctx,unit,len(ww))
            assert abs(gz-sz)<=.01
            t['smoke_checks']=dict(cloned_cache_exact=True,cloned_cache_margin=replay,unit_token_ids=ids,unit_graph_margin=gz,unit_reference_margin=sz,
                unit_difference=gz-sz,unit_tolerance=.01,diagnostic_forwards=3)
            diagnostics+=3;diagnostic_seconds+=tick()-start;checked=True
        new_speech.append(z);traces.append(t)
    base=record(row,ctx,visual,speech,prefix_seconds+visual_seconds+reference_seconds,'m1_native')
    new=record(row,ctx,visual,new_speech,metadata['standalone_seconds']+prefix_seconds+visual_seconds+new_seconds,'m1_terminal_lattice')
    for r in (base,new):r['code_path']=str(__file__).replace(str(ROOT)+'/','')
    actual=j.forward_calls-first;assert actual==base['calls']+sum(m['available'] for m in ww)+diagnostics
    checks=dict(prefix_seconds=prefix_seconds,visual_seconds=visual_seconds,reference_speech_seconds=reference_seconds,
        new_speech_seconds=new_seconds,diagnostic_seconds=diagnostic_seconds,diagnostic_forwards=diagnostics,
        preprocessing_seconds=metadata['standalone_seconds'],input_actual_forwards=metadata['actual_forwards'],actual_forwards=actual,
        peak_GiB=max(torch.cuda.max_memory_allocated()/2**30,metadata['peak_GiB']))
    del cache
    return dict(version=VERSION,GT_read=False,segments=[list(s) for s in segments],native_input=native_input,
        base=base,optimized=new,traces=traces,checks=checks)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261004_m1_lattice'/('r3_full_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    config=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,cache_version=CACHE_VERSION,
        reader_version=VERSION,constants=CONSTANTS,GT_in_reader=False,smoke=a.smoke,
        torch=torch.__version__,transformers=transformers.__version__,diagnostic_unit_tolerance=.01,
        code='experiments/20261004_m1_lattice/{terminal_graph,terminal_reader,terminal_measure}.py; unchanged R1 ASR cache; sources2026-10-05',
        command='python -u '+' '.join(sys.argv))
    cp=out/'config.json'
    if cp.exists():
        old=json.loads(cp.read_text());assert {k:v for k,v in old.items() if k!='date'}=={k:v for k,v in config.items() if k!='date'}
    else:cp.write_text(json.dumps(config,indent=2)+'\n')
    rows=selected_rows(a.smoke);asr={ds:load_asr(ds) for ds in DATASETS};renderer=cpu_renderer();done={}
    for row in rows:
        key=row['dataset'],row['video_id'];p=out/'records'/key[0]/(key[1]+'.json')
        if p.exists():
            b=json.loads(p.read_text());m=json.loads((CACHE/key[0]/(key[1]+'.json')).read_text())
            validate_bundle(row,b,m,asr[key[0]].get(key[1],[]),renderer,a.smoke);done[key]=b
    torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=0
    hook=j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1))
    for i,row in enumerate(rows,1):
        key=row['dataset'],row['video_id']
        if key in done:logging.info('%d/%d reuse %s/%s',i,len(rows),*key);continue
        segments=asr[key[0]].get(key[1],[]);m=json.loads((CACHE/key[0]/(key[1]+'.json')).read_text());validate(m,row,segments)
        b=read_video(j,row,segments,m,a.smoke);validate_bundle(row,b,m,segments,renderer,a.smoke)
        p=out/'records'/key[0]/(key[1]+'.json');p.parent.mkdir(parents=True,exist_ok=True)
        temp=p.with_suffix('.partial');temp.write_text(json.dumps(b)+'\n');temp.replace(p);done[key]=b
        logging.info('%d/%d %s/%s %.2fs',i,len(rows),*key,b['optimized']['extra']['standalone_seconds'])
    hook.remove()
    for name in ('base','optimized'):
        d=out/name;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        with (d/'predictions.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(done[row['dataset'],row['video_id']][name])+'\n')
    logging.info('DONE coverage=%d',len(rows))


if __name__=='__main__':main()
