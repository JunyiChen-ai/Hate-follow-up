"""R2: original global/ownstance/S, fresh source-path V; source cost retained."""
import argparse
import copy
import json
import logging
import math
import socket
import sys
import time
import numpy as np
import torch
from inputs import ROOT,CACHE,DATASETS,selected_rows,validate
from interface import VERSION,CONSTANTS,SPEC
from reader import read,validate_trace
from measure import native_input,prediction
from src.mllm_judge import Judge,MODEL,yesno_question
from src.stance_cache import build,margin
from src.source_generation import clock
from src.video_inputs import frame_paths,load_asr
READING_VERSION='R2 nativeglobal/ownstance/S + actual sourcepath V;2026-10-06'


@torch.no_grad()
def read_video(j,row,segments,m,smoke):
    first=j.forward_calls;vision=j.vision_calls;torch.cuda.reset_peak_memory_stats();start=clock(j)
    cache,ctx=build(j,frame_paths(row['dataset'],row['video_id'],20),segments);prefix_seconds=clock(j)-start
    start=clock(j);conversation=native_input(j,row,segments,ctx['stance']);input_binding_seconds=clock(j)-start
    assert conversation['head']==ctx['head'] and conversation['stance_cache_tokens']==ctx['stance_cache_tokens']
    ctx['native_stance_ids']=conversation['native_stance_ids'];ctx['windows']=m['windows'];native_visual=[];native_speech=[];native_times=[]
    times=dict(native_visual=0.,native_speech=0.,new_visual=0.,new_speech=0.,diagnostic=0.)
    for w in m['windows']:
        start=clock(j);v=margin(j,cache,ctx,yesno_question(w['i'],len(m['windows']),w['start'],w['end'],w['body'],'visual'));vt=clock(j)-start;times['native_visual']+=vt
        if w['body'].strip():
            start=clock(j);s=margin(j,cache,ctx,yesno_question(w['i'],len(m['windows']),w['start'],w['end'],w['body'],'speech'));st=clock(j)-start;times['native_speech']+=st
        else:s=None;st=0.
        native_visual.append(v);native_speech.append(s);native_times.append(dict(visual=vt,speech=st))
    # Fresh full multimodal V reads the literal native conversation and actual
    # owned source path. No R1 tree-global logits/stance/margins are reused.
    del cache;new_visual=[];traces=[];diagnostics=0
    for w in m['windows']:
        z,trace=read(j,ctx,m,w['i'],'visual');new_visual.append(z);times['new_visual']+=trace['seconds']
        if smoke and diagnostics==0:
            start=clock(j);replay,other=read(j,ctx,m,w['i'],'visual');assert replay==z and other['input_tokens']==trace['input_tokens'] and other['image_grid']==trace['image_grid']
            trace.update(repeat_margin=replay,repeat_exact=True,repeat_seconds=clock(j)-start);times['diagnostic']+=trace['repeat_seconds'];diagnostics+=1
        trace['available']=True;traces.append(dict(window=w['i'],branches=dict(visual=trace,speech=dict(available=bool(w['body'].strip()),source='fresh_original_native_S')),native_seconds=native_times[w['i']]))
    base=prediction(row,ctx,native_visual,native_speech,prefix_seconds+times['native_visual']+times['native_speech'],'m1_native')
    optimized=prediction(row,ctx,new_visual,native_speech,m['standalone_seconds']+prefix_seconds+times['new_visual']+times['native_speech'],'m1_interval_witness_r2')
    optimized['calls']+=m['actual_forwards'];actual=j.forward_calls-first;assert actual==base['calls']+len(m['windows'])+diagnostics
    checks=dict(actual_forwards=actual,actual_vision_forwards=j.vision_calls-vision,prefix_seconds=prefix_seconds,new_prefix_seconds=0.,input_binding_seconds=input_binding_seconds,**times,diagnostic_forwards=diagnostics,
        preprocessing_seconds=m['standalone_seconds'],input_actual_forwards=m['actual_forwards'],peak_GiB=max(torch.cuda.max_memory_allocated()/2**30,m['peak_GiB']))
    conversation['files']=[str(p.relative_to(ROOT)) for p in conversation['files']]
    return dict(version=VERSION,reading_version=READING_VERSION,GT_read=False,native_input=conversation,base=base,optimized=optimized,traces=traces,checks=checks)


def validate_bundle(row,b,m,segments,j,smoke):
    validate(m,row,segments,j);assert b['version']==VERSION and b['reading_version']==READING_VERSION and b['GT_read'] is False
    expected=native_input(j,row,segments,b['base']['extra']['stance']);expected['files']=[str(p.relative_to(ROOT)) for p in expected['files']];assert expected==b['native_input']
    ctx={**expected,'files':[ROOT/p for p in expected['files']]};base,new=b['base'],b['optimized'];repeats=0
    for k in ('z_video','stance','prefix_tokens','stance_cache_tokens','stance_cache_logical_start'):assert base['extra'][k]==new['extra'][k]
    assert len(b['traces'])==len(m['windows'])==len(base['extra']['windows'])==len(new['extra']['windows'])
    for w,t,bw,nw in zip(m['windows'],b['traces'],base['extra']['windows'],new['extra']['windows']):
        assert t['window']==w['i'];trace=t['branches']['visual'];validate_trace(j,ctx,m,w['i'],'visual',trace)
        assert trace['available'] and trace['margin']==nw['z_visual'] and bw.get('z_speech')==nw.get('z_speech')
        assert t['branches']['speech']==dict(available=bool(w['body'].strip()),source='fresh_original_native_S')
        if trace.get('repeat_exact'):assert trace['repeat_margin']==trace['margin'];repeats+=1
        for ww in (bw,nw):
            assert (ww['i'],ww['start'],ww['end'])==(w['i'],w['start'],w['end']) and ('z_speech' in ww)==bool(w['body'].strip())
            assert ww['z']==max(ww['z_visual'],ww.get('z_speech',-math.inf))
    for pred in (base,new):
        assert (pred['dataset'],pred['video_id'],pred['duration'],pred['native_rate'],pred['error'])==(row['dataset'],row['video_id'],float(row['duration']),4.,None)
        assert pred['extra']['stance']==('Yes' if pred['extra']['z_video']>0 else 'No')
        assert pred['extra']['prefix_tokens']==expected['prefix_tokens'] and pred['extra']['stance_cache_tokens']==expected['stance_cache_tokens']
        idx=np.clip(((np.arange(math.ceil(pred['duration']*4))+.5)/4//8).astype(int),0,len(m['windows'])-1)
        assert np.array_equal(pred['score_curve'],np.asarray([ww['z'] for ww in pred['extra']['windows']])[idx]) and np.isfinite(pred['score_curve']).all()
    if smoke:assert repeats==1
    c=b['checks'];assert c['actual_forwards']==base['calls']+len(m['windows'])+repeats and c['actual_vision_forwards']==1+len(m['windows'])+repeats
    assert c['diagnostic_forwards']==repeats and c['preprocessing_seconds']==m['standalone_seconds'] and c['input_actual_forwards']==m['actual_forwards']
    expected_calls=3+len(m['windows'])+sum(bool(w['body'].strip()) for w in m['windows']);assert base['calls']==expected_calls and new['calls']==expected_calls+m['actual_forwards']
    assert c['new_prefix_seconds']==c['new_speech']==0. and c['peak_GiB']>=m['peak_GiB']
    assert abs(c['new_visual']-sum(t['branches']['visual']['seconds'] for t in b['traces']))<1e-6
    for kind in ('visual','speech'):assert abs(c['native_'+kind]-sum(t['native_seconds'][kind] for t in b['traces']))<1e-6
    assert abs(c['diagnostic']-sum(t['branches']['visual'].get('repeat_seconds',0.) for t in b['traces']))<1e-6
    assert all(np.isfinite(v) and v>=0 for v in c.values())
    assert abs(base['extra']['standalone_seconds']-c['prefix_seconds']-c['native_visual']-c['native_speech'])<1e-6
    assert abs(new['extra']['standalone_seconds']-m['standalone_seconds']-c['prefix_seconds']-c['new_visual']-c['native_speech'])<1e-6


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();out=ROOT/'runs/20261005_m1_interval_witness'/('r2_full_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)]);logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,version=VERSION,reading_version=READING_VERSION,constants=CONSTANTS,GT_read=False,smoke=a.smoke,torch=torch.__version__,transformers=transformers.__version__,code='experiments/20261005_m1_interval_witness/{measure_r2,reader,inputs,interface}.py;sources2026-10-06',command='python -u '+' '.join(sys.argv))
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');rows=selected_rows(a.smoke);asr={ds:load_asr(ds) for ds in DATASETS};torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=j.vision_calls=0
    hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))];results={name:[] for name in ('base','optimized')}
    for number,row in enumerate(rows,1):
        segments=asr[row['dataset']].get(row['video_id'],[]);m=json.loads((CACHE/row['dataset']/row['video_id']/'metadata.json').read_text());validate(m,row,segments,j)
        path=out/'records'/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():b=json.loads(path.read_text())
        else:
            b=read_video(j,row,segments,m,a.smoke);validate_bundle(row,b,m,segments,j,a.smoke);temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(b)+'\n');temporary.replace(path)
        validate_bundle(row,b,m,segments,j,a.smoke)
        for name in results:results[name].append(b[name])
        logging.info('%d/%d %s/%s %.2fs',number,len(rows),row['dataset'],row['video_id'],b['optimized']['extra']['standalone_seconds'])
    for name,rr in results.items():
        d=out/name;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n');(d/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
    for hook in hooks:hook.remove()
    logging.info('DONE coverage=%d',len(rows))


if __name__=='__main__':main()
