#!/usr/bin/env python3
"""NoGT CPU checks of interface B source compiler and exact grammar replay."""
import json
import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.video_inputs import load_asr
from src.mllm_renderer import cpu_renderer
from handle_interface import (catalog,compile_selection,join_possible,target_choices,
    write_plans,write_module,CONSTANTS)
from handle_inputs import selected_rows,sources
from handle_decoder import Stream,Capped
from program import execute,canonical,span,local


class FixtureWriter:
    def __init__(self,j):self.j=j;self.tokens=[];self.events=[]
    def force(self,text):
        start=len(self.tokens);self.tokens+=self.j.tok.encode(text,add_special_tokens=False)
        self.events.append(dict(kind='forced',text=text,start=start,end=len(self.tokens)))
    def choose(self,options):
        start=len(self.tokens);value=options[-1]
        self.tokens+=self.j.tok.encode(value,add_special_tokens=False)
        self.events.append(dict(kind='choice',options=options,selected=value,start=start,end=len(self.tokens)))
        return value
    def description(self):
        start=len(self.tokens);value='UNKNOWN'
        self.tokens+=self.j.tok.encode(value,add_special_tokens=False)+self.j.tok.encode('"',add_special_tokens=False)
        self.events.append(dict(kind='description',text=value,reason='model_quote',start=start,end=len(self.tokens)))


def replay(j,writer,fn):
    stream=Stream(j,len(writer.tokens)+1,tokens=writer.tokens);value=fn(stream)
    assert stream.tokens==writer.tokens and stream.events==writer.events
    return value


def main():
    j=cpu_renderer();asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')}
    totals=dict(videos=0,windows=0,scope_calls=0,action_calls=0,joins=0,plans=0,target_candidates=0)
    for row in selected_rows(True):
        source,windows,paths=sources(row,asr[row['dataset']].get(row['video_id'],[]))
        for w in windows:
            c=catalog(source,w)
            for obj in c['speech'].values():
                r=obj['ref'];seg=source['segments'][r['segment']]
                assert obj['text']==seg['text'][r['start_char']:r['end_char']]
                assert w['start']-1e-9<=obj['start']<obj['end']<=w['end']+1e-9
            for obj in c['contexts'].values():assert obj['end']<=w['start'] or obj['start']>=w['end']
        for offset in range(0,len(windows),CONSTANTS['planner_windows']):
            requested=windows[offset:offset+CONSTANTS['planner_windows']]
            writer=FixtureWriter(j);selected=write_plans(writer,source,requested)
            assert len(writer.tokens)<=CONSTANTS['planner_tokens']
            assert json.loads(j.tok.decode(writer.tokens))==selected
            assert replay(j,writer,lambda s:write_plans(s,source,requested))==selected
            for w,selection in zip(requested,selected):
                plan=compile_selection(source,w,selection)
                def perceive(kind,packet):
                    module=FixtureWriter(j);write_module(module,kind,packet)
                    assert len(module.tokens)<=CONSTANTS['module_tokens']
                    replay(j,module,lambda s:write_module(s,kind,packet))
                    raw=j.tok.decode(module.tokens);json.loads(raw)
                    totals[kind+'_calls']+=1
                    if kind=='scope':
                        for literal in target_choices(packet)[1:]:
                            r=json.loads(literal);seg=source['segments'][r['segment']]
                            assert 0<=r['start_char']<r['end_char']<=len(seg['text']);totals['target_candidates']+=1
                    return raw
                result=execute(source,w,plan,perceive)
                assert result['status']=='valid' and not result['rejected']
                assert all(c['result']['kind']!='UNKNOWN' for c in result['calls'])
                assert len(result['calls'])<=2 and result['accepted_operations']<=12
                totals['joins']+=sum(e['value']['kind']=='join' for e in result['emitted'])
                totals['plans']+=1
        totals['videos']+=1;totals['windows']+=len(windows)
    # A boundary fixture independently checks no character may straddle a window.
    source=dict(segments=[dict(id=0,start=0.,end=3.,text='abcdef',time_source='fixture')],frames=[])
    for a,b in ((0.,1.),(.1,1.4),(1.,2.),(2.,3.)):
        w=dict(id=0,start=a,end=b);c=catalog(source,w)
        expected=[i for i in range(6) if a<=i*.5 and (i+1)*.5<=b]
        observed=[i for v in c['speech'].values() for i in range(v['ref']['start_char'],v['ref']['end_char'])]
        assert expected==observed
    # Actual source handle is mandatory when available; wrong coordinates cannot be supplied.
    w=dict(id=0,start=0.,end=3.)
    for bad in ('S99999999',None):
        try:compile_selection(source,w,dict(window=0,speech=bad,frame=None,contexts=[],join=False))
        except AssertionError:pass
        else:raise AssertionError('invalid source choice accepted')
    print(json.dumps(dict(PASS=True,GT_read=False,**totals),indent=2))


if __name__=='__main__':main()
