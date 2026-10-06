"""Strict reader/source/input/cost replay before any annotation evaluation."""
import math
import numpy as np
from inputs import SPEC
from reader import REVISION,IMPLEMENTATION,prediction,validate_parse,speech_question
from src.native_input_binding import validate as validate_binding
from src.video_inputs import fixed_windows,window_text
from src.mllm_judge import yesno_question


def validate_bundle(j,row,segments,source,bundle,smoke):
    assert bundle.get('reader_revision',1)==REVISION
    assert bundle['version']==SPEC['version'] and bundle['spec']==SPEC and bundle['segments']==[list(s) for s in segments]
    ctx=bundle['native_ctx'];validate_binding(j,row,segments,ctx,SPEC,IMPLEMENTATION,bundle['binding'])
    checks=bundle['checks'];assert checks['GT_read'] is False and checks['source_seconds']==source['source_seconds'] and checks['source_counts']==source['counts']
    assert checks['host']==source['host'],'whole source and reader must use the same host'
    wins=fixed_windows(float(row['duration']),8);traces=bundle['traces'];packets=bundle['packets']
    assert len(traces)==len(packets)==len(source['windows'])==len(wins)
    parser_forwards=0;visual=[];base_s=[];new_s=[]
    for i,((a,b),t,p,scoped) in enumerate(zip(wins,traces,packets,source['windows'])):
        body=window_text(segments,a,b);original=yesno_question(i,len(wins),a,b,body,'speech')
        assert (t['i'],t['bounds'],t['body'],t['original_question'])==(i,[a,b],body,original)
        validate_parse(j,source['words'],scoped,p)
        if p['generation'] is not None:parser_forwards+=p['generation']['actual_forwards']
        available=bool(body.strip()) or (bool(scoped['local_ids']) and (REVISION==1 or p['reason']=='compiled'))
        expected=speech_question(original,source['words'],scoped,p) if available else None
        assert t['new_question']==expected and t['cache_restored'] and t['rope_restored']
        assert (t['native_speech'] is not None)==bool(body.strip()) and (t['new_speech'] is not None)==(expected is not None)
        if p['reason']=='no_local_words' or (REVISION==2 and p['reason']!='compiled'):assert t['new_speech']==t['native_speech']
        visual.append(t['native_visual']);base_s.append(t['native_speech']);new_s.append(t['new_speech'])
    clones=sum(t.get('clone_exact',False) for t in traces);assert clones==int(smoke)*int(any(s is not None for s in new_s))
    assert checks['parser_forwards']==parser_forwards and checks['diagnostic_forwards']==clones
    assert checks['actual_vision']==1 and checks['actual_forwards']==3+len(wins)+sum(s is not None for s in base_s)+parser_forwards+sum(s is not None for s in new_s)+clones
    times=checks['times'];assert all(math.isfinite(t) and t>=0 for t in times.values())
    for name,speech,seconds,method,added in (
        ('base',base_s,sum(times[k] for k in ('prefix','native_visual','native_speech')),'m1_native',0),
        ('optimized',new_s,source['source_seconds']+sum(times[k] for k in ('prefix','native_visual','parser','new_speech','native_binding')),'m1_texttiling' if REVISION==1 else 'm1_texttiling_r2',parser_forwards+sum(source['counts'].values()))):
        assert bundle[name]==prediction(row,ctx,visual,speech,seconds,method,added)
        assert np.isfinite(bundle[name]['score_curve']).all()
