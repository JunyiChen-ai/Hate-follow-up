#!/usr/bin/env python3
"""Independent interval enumeration, factual packet and replay checks; no GT."""
import copy
import json
from program import *


def source(text='ab😀dEFghijklmnop'):
    return dict(frames=[dict(id=i,time=t,time_source='native_filename_nominal') for i,t in enumerate((0.,7.5,8.,15.5))],
        segments=[dict(id=0,start=0.,end=16.,text=text,time_source='native_asr_segment_proportional_chars')])


def main():
    s=source();full=span(s,0,0,16)
    assert full['text']==s['segments'][0]['text'] and len(full['text'])==16
    count=0
    # Enumerate character cells directly instead of repeating the ceil/floor formula.
    for a in (0.,.25,.5,1.,7.99,8.):
        for b in (8.,8.25,12.5,16.):
            if b<=a:continue
            w=dict(id=0,start=a,end=b);obj=local(s,full,w)
            expected=[i for i in range(16) if a<=i and i+1<=b]
            if expected:
                assert obj['text']==''.join(s['segments'][0]['text'][i] for i in expected)
                assert obj['ref']['start_char']==expected[0] and obj['ref']['end_char']==expected[-1]+1
            else:assert obj['kind']=='UNKNOWN'
            count+=1
    assert span(s,True,0,3)['kind']=='UNKNOWN'
    assert local(s,full,dict(id=0,start=.25,end=.75))['kind']=='UNKNOWN'
    windows=[dict(id=0,start=0.,end=8.),dict(id=1,start=8.,end=16.)]
    p,rejected=parse_chunk('[{"window":0,"ops":[]},{"window":0,"ops":[]},{"window":9,"ops":[]}]',windows)
    assert p[0]['reason']=='duplicate_window' and p[1]['reason']=='missing_window' and len(rejected)==2
    p,_=parse_chunk('```json\n[]\n```',windows);assert all(v['reason']=='invalid_array' for v in p.values())
    p,_=parse_chunk('[{"window":true,"ops":[]}]',windows);assert p[0]['reason']=='missing_window'
    p,_=parse_chunk('[{"window":0,"ops":[]}]',windows,True);assert all(v['reason']=='truncated_output' for v in p.values())
    # The known quotation context stays interpretation-only and never becomes a local span.
    s=source('He quoted it. I reject it now.')
    ops=[['span','context_source',0,0,12],['context','ctx','context_source'],
         ['span','utterance',0,12,len(s['segments'][0]['text'])],['local','loc','utterance'],
         ['scope','scope','loc',['ctx']],['frame','fr',3],['local','frame_local','fr'],
         ['action','action','frame_local'],['join','joint','loc','frame_local'],['emit',['scope','action','joint']]]
    packets=[]
    def perceive(kind,packet):
        packets.append((kind,copy.deepcopy(packet)))
        if kind=='scope':return canonical(dict(speaker='speaker',mode='rejected',target=packet['sources'][0]['value']['ref'],support=['loc','ctx']))
        return canonical(dict(actor='person',action='speaking',target='UNKNOWN',support=['frame_local']))
    r=execute(s,windows[1],dict(status='valid',ops=ops),perceive)
    assert len(packets)==2 and r['status']=='valid'
    assert r['emitted'][0]['value']['mode']=='rejected'
    assert packets[0][1]['sources'][1]['value']['interpretation_only'] and 'local' not in packets[0][1]['sources'][1]['value']
    assert packets[1][1]['sources'][0]['value']['ref']==dict(frame=3)
    assert all('scope' not in canonical(packet) for kind,packet in packets)
    assert r['emitted'][2]['value']['kind']=='join'
    v=branch_record(r,'visual');speech=branch_record(r,'speech')
    assert [x['value']['kind'] for x in v['evidence']]==['action','join']
    assert 'text' not in v['evidence'][1]['value'] and 'span_ref' in v['evidence'][1]['value']
    assert [x['value']['kind'] for x in speech['evidence']]==['scope']
    calls=iter(r['calls'])
    def replay(kind,packet):
        c=next(calls);assert c['kind']==kind and c['packet']==packet;return c['raw']
    assert execute(s,windows[1],dict(status='valid',ops=ops),replay)==r and next(calls,None) is None
    # The end boundary belongs to the next window, not the earlier interval.
    basic=[['span','s',0,0,len(s['segments'][0]['text'])],['local','l','s'],['frame','f',2],['local','fl','f'],['join','j','l','fl'],['emit',['j']]]
    x=execute(s,windows[0],dict(status='valid',ops=basic),lambda *_:(_ for _ in ()).throw(AssertionError('unexpected_model_call')))
    assert x['emitted'][0]['value']['kind']=='UNKNOWN' and not x['calls']
    bad=normalize_module('scope',packets[0][1],canonical(dict(speaker='speaker',mode='quoted',target='UNKNOWN',support=['invented'])))
    assert bad['reason']=='unsupported_source'
    bad=normalize_module('scope',packets[0][1],canonical(dict(speaker='speaker',mode='quoted',target=dict(segment=99,start_char=0,end_char=1),support=['loc'])))
    assert bad['target']=='UNKNOWN'
    bad=normalize_module('action',packets[1][1],canonical(dict(actor='person',action=' '.join(['x']*13),target='UNKNOWN',support=['frame_local'])))
    assert bad['action']=='UNKNOWN'
    # UNKNOWN dependencies never cause a new model call and never become absence.
    x=execute(s,windows[0],dict(status='valid',ops=[['frame','f',99],['local','l','f'],['action','a','l'],['emit',['a']]]),lambda *_:1/0)
    assert not x['calls'] and x['emitted'][0]['value']['kind']=='UNKNOWN'
    x=execute(s,windows[0],dict(status='valid',ops=[['local','x','future'],['span','future',0,0,1],['emit',['future']]]),lambda *_:1/0)
    assert x['rejected'][0]['reason']=='forward_or_missing_reference'
    # Every prompt is the frozen literal material in README, rather than a regenerated rewrite.
    from pathlib import Path
    text=(Path(__file__).parent/'README.md').read_text()
    assert PLANNER_SYSTEM in text and API in text and PLANNER_END in text
    print(json.dumps(dict(GT_read=False,character_cell_oracles=count,quotation_scope=True,point_join=True,
        support_validation=True,fresh_source_packets=True,deterministic_replay=True,UNKNOWN_preserved=True,literal_planner=True)),flush=True)
    print('CPU_CHECKS_DONE',flush=True)


if __name__=='__main__':main()
