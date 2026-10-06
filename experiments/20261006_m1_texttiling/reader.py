"""Fresh native V and source-compiled independent S on the unchanged stance."""
import copy
import json
import math
import os
import socket
import numpy as np
import torch
from inputs import ROOT,SPEC
from compiler import content,writer_for,reader_record,question
from src.structured_source_generation import generate,validate_generation
from src.stance_cache import build,margin
from src.video_inputs import frame_paths,fixed_windows,window_text
from src.mllm_judge import yesno_question
from src.source_generation import clock
from src.native_input_binding import save as save_binding

REVISION=int(os.environ.get('READER_REVISION','1'));assert REVISION in (1,2)

IMPLEMENTATION='R1 speech lexical partition/scoped grammar reader; source binding and full pipeline costs;2026-10-06'


def parse(j,words,scoped):
    base=dict(scope=scoped,generation=None,selection=None)
    if not scoped['local_ids']:return dict(**base,reason='no_local_words')
    if len(scoped['local_ids'])+len(scoped['context_ids'])>SPEC['source_word_cap']:return dict(**base,reason='whole_packet_word_cap')
    supplied=content(words,scoped);text=j.render([j.turn('system',SPEC['source_system']),dict(role='user',content=supplied)],True)
    ids=j.tok(text,add_special_tokens=False)['input_ids'];base['input_token_count']=len(ids)
    if len(ids)>SPEC['source_input_token_cap']:return dict(**base,reason='whole_packet_input_token_cap')
    generation=generate(j,ROOT,SPEC['source_system'],supplied,[],SPEC['source_generation_token_cap'],writer_for(words,scoped))
    base['generation']=generation
    if generation['truncated']:return dict(**base,reason='incomplete_generation')
    base['selection']=generation['selection']
    if generation['selection']['unknown']:return dict(**base,reason='unknown_local_act')
    return dict(**base,reason='compiled',record=reader_record(words,scoped,generation['selection']))


def validate_parse(j,words,scoped,packet):
    assert packet['scope']==scoped
    if packet['generation'] is None:
        expected=dict(scope=scoped,generation=None,selection=None)
        if not scoped['local_ids']:expected['reason']='no_local_words'
        elif len(scoped['local_ids'])+len(scoped['context_ids'])>SPEC['source_word_cap']:expected['reason']='whole_packet_word_cap'
        else:
            supplied=content(words,scoped);rendered=j.render([j.turn('system',SPEC['source_system']),dict(role='user',content=supplied)],True)
            expected['input_token_count']=len(j.tok(rendered,add_special_tokens=False)['input_ids'])
            assert expected['input_token_count']>SPEC['source_input_token_cap'],'missing required actual parser generation'
            expected['reason']='whole_packet_input_token_cap'
        assert expected==packet
        return
    supplied=content(words,scoped);g=packet['generation']
    assert len(scoped['local_ids'])+len(scoped['context_ids'])<=SPEC['source_word_cap']
    assert packet['input_token_count']==len(g['input_tokens'])<=SPEC['source_input_token_cap']
    validate_generation(j,ROOT,g,SPEC['source_system'],supplied,[],SPEC['source_generation_token_cap'],writer_for(words,scoped))
    if g['truncated']:assert packet['reason']=='incomplete_generation' and packet['selection'] is None
    else:
        assert packet['selection']==g['selection']
        if g['selection']['unknown']:assert packet['reason']=='unknown_local_act'
        else:assert packet['reason']=='compiled' and packet['record']==reader_record(words,scoped,g['selection'])


def speech_question(native_question,words,scoped,packet):
    if packet['reason']=='no_local_words' or (REVISION==2 and packet['reason']!='compiled'):return native_question
    if packet['reason']=='compiled':return question(native_question,packet['record'])
    local=[words[i] for i in scoped['local_ids']]
    record=dict(current_window=scoped['bounds'],aligned_literal_LOCAL=[dict(id=w['id'],bounds=[w['start'],w['end']],text=w['text']) for w in local],
        explaining_context=[],selection_status=packet['reason'],role_semantics='only LOCAL is direct current-window evidence; unavailable parser is not a no-hate decision')
    return question(native_question,record)


def prediction(row,ctx,visual,speech,seconds,method,additional_forwards=0):
    wins=fixed_windows(float(row['duration']),8);windows=[]
    for i,((a,b),v,s) in enumerate(zip(wins,visual,speech)):
        w=dict(i=i,start=a,end=b,z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:w['z_speech']=s
        windows.append(w)
    idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//8).astype(int),0,len(wins)-1)
    return dict(schema_version=1,method=method,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),native_rate=4,
        score_curve=np.asarray([w['z'] for w in windows])[idx].tolist(),intervals=[],error=None,seed=0,
        calls=3+len(wins)+sum(s is not None for s in speech)+additional_forwards,
        code_path='experiments/20261006_m1_texttiling/reader.py',extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],
            prefix_tokens=ctx['prefix_tokens'],stance_cache_tokens=ctx['stance_cache_tokens'],stance_cache_logical_start=ctx['stance_cache_logical_start'],
            windows=windows,standalone_seconds=seconds))


@torch.no_grad()
def read_video(j,row,segments,source,out,smoke):
    before=j.forward_calls;before_vision=j.vision_calls;start=clock(j)
    if j.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    cache,ctx=build(j,frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']),segments)
    times=dict(prefix=clock(j)-start,native_visual=0.,native_speech=0.,parser=0.,new_speech=0.,diagnostic=0.,native_binding=0.)
    proof=out/'proof'/row['dataset']/row['video_id'];proof.mkdir(parents=True,exist_ok=True)
    start=clock(j);binding=save_binding(j,row,segments,ctx,SPEC,IMPLEMENTATION,proof/'native_pixels.npy');times['native_binding']=clock(j)-start
    visual=[];native_speech=[];new_speech=[];packets=[];traces=[];clones=0;parser_forwards=0
    words=source['words'];wins=fixed_windows(float(row['duration']),8)
    # All native branches finish before source parsing touches model rope state.
    for i,(a,b) in enumerate(wins):
        body=window_text(segments,a,b);start=clock(j);v=margin(j,cache,ctx,yesno_question(i,len(wins),a,b,body,'visual'));times['native_visual']+=clock(j)-start;visual.append(v)
        if body.strip():start=clock(j);s=margin(j,cache,ctx,yesno_question(i,len(wins),a,b,body,'speech'));times['native_speech']+=clock(j)-start
        else:s=None
        native_speech.append(s)
    for i,((a,b),scoped,s) in enumerate(zip(wins,source['windows'],native_speech)):
        body=window_text(segments,a,b);original=yesno_question(i,len(wins),a,b,body,'speech')
        rope=ctx['rope'].clone();start=clock(j)
        try:packet=parse(j,words,scoped)
        finally:j.model.model.rope_deltas=rope
        times['parser']+=clock(j)-start;packets.append(packet)
        if packet['generation'] is not None:parser_forwards+=packet['generation']['actual_forwards']
        available=s is not None or (bool(scoped['local_ids']) and (REVISION==1 or packet['reason']=='compiled'))
        sq=speech_question(original,words,scoped,packet) if available else None
        if sq is not None:start=clock(j);new=margin(j,cache,ctx,sq);times['new_speech']+=clock(j)-start
        else:new=None
        if packet['reason']=='no_local_words' or (REVISION==2 and packet['reason']!='compiled'):assert new==s
        trace=dict(i=i,bounds=[a,b],body=body,original_question=original,new_question=sq,native_visual=visual[i],native_speech=s,new_speech=new,
            cache_restored=cache.get_seq_length()==ctx['stance_cache_tokens'],rope_restored=torch.equal(j.model.model.rope_deltas,ctx['rope']))
        assert trace['cache_restored'] and trace['rope_restored']
        if smoke and not clones and sq is not None:
            start=clock(j);clone=copy.deepcopy(cache);replayed=margin(j,clone,ctx,sq)
            assert replayed==new and clone.get_seq_length()==ctx['stance_cache_tokens']
            assert all(torch.equal(x.keys,y.keys) and torch.equal(x.values,y.values) for x,y in zip(cache.layers,clone.layers))
            trace['clone_exact']=True;clones=1;del clone;times['diagnostic']+=clock(j)-start
        new_speech.append(new);traces.append(trace)
    native_seconds=sum(times[k] for k in ('prefix','native_visual','native_speech'))
    seconds=source['source_seconds']+sum(times[k] for k in ('prefix','native_visual','parser','new_speech','native_binding'))
    base=prediction(row,ctx,visual,native_speech,native_seconds,'m1_native')
    new=prediction(row,ctx,visual,new_speech,seconds,'m1_texttiling' if REVISION==1 else 'm1_texttiling_r2',parser_forwards+sum(source['counts'].values()))
    checks=dict(host=socket.gethostname(),GT_read=False,times=times,source_seconds=source['source_seconds'],source_counts=source['counts'],
        parser_forwards=parser_forwards,diagnostic_forwards=clones,actual_forwards=j.forward_calls-before,actual_vision=j.vision_calls-before_vision,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30 if j.device.type=='cuda' else 0.)
    assert checks['actual_forwards']==base['calls']+parser_forwards+sum(s is not None for s in new_speech)+clones and checks['actual_vision']==1
    result=dict(version=SPEC['version'],spec=SPEC,binding=binding,segments=[list(x) for x in segments],
        native_ctx={k:v for k,v in ctx.items() if k not in ('positions','rope','files')},native_rope=ctx['rope'].tolist(),
        base=base,optimized=new,packets=packets,traces=traces,checks=checks)
    if REVISION==2:result['reader_revision']=2
    del cache;return result
