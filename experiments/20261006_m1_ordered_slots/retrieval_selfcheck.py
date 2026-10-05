"""Known-vector ordered assignment and actual native-token bounded grammar."""
import itertools
import json
import numpy as np
from retrieval import ROOT,SPEC,assign
from interface import caption_writer,slot_writer,compile_slots,unavailable
from src.structured_source_generation import Stream
from src.mllm_renderer import cpu_renderer


def main():
    vectors=np.asarray([[1,0],[1,0],[0,1],[-1,0],[1,0]],np.float32);mask=np.ones(5,bool)
    q={role:[1,0] for role in SPEC['slot_order']};p=assign(vectors,mask,q,2)
    assert p['bindings']['prequel']['source_id']==0 and p['bindings']['current']['source_id']==2 and p['bindings']['sequel']['source_id']==4
    assert p['objective']==2 and p['source_ids']==[0,4]
    # Independent enumeration of the feasible side sources, including NONE.
    objectives=[]
    for before,after in itertools.product([None,0,1],[None,3,4]):
        objectives.append(sum(0 if i is None else float(np.dot(vectors[i],q['prequel'])) for i in (before,after))+float(np.dot(vectors[2],q['current'])))
    assert p['objective']==max(objectives)
    empty=assign(vectors,mask,{role:None for role in q},2)
    assert empty['source_ids']==[] and empty['bindings']['current']['source_id']==2
    negative=assign(-np.abs(vectors),mask,q,2);assert negative['source_ids']==[]
    masked=assign(vectors,np.zeros(5,bool),q,2);assert masked['source_ids']==[]
    for i in range(5):
        r=assign(vectors,mask,q,i)
        assert r['bindings']['prequel']['source_id'] is None or r['bindings']['prequel']['source_id']<i
        assert r['bindings']['sequel']['source_id'] is None or r['bindings']['sequel']['source_id']>i
    j=cpu_renderer()
    class Feed(Stream):
        def __init__(self,texts,limit):super().__init__(j,limit,tokens=[]);self.texts=iter(texts)
        def force(self,text):self.saved.extend(j.tok.encode(text,add_special_tokens=False));return super().force(text)
        def description(self):
            self.saved.extend(j.tok.encode(next(self.texts),add_special_tokens=False)+j.tok.encode('"',add_special_tokens=False));return super().description()
    indices=[0,12,123];texts=['earlier cup','current cup','later cup','NONE','current sign','UNKNOWN','earlier sign','current words','later reply']
    stream=Feed(texts,768);rows=slot_writer(indices)(stream)
    assert json.loads(j.tok.decode(stream.tokens))==[dict(id=i,**dict(zip(SPEC['slot_order'],texts[k*3:k*3+3]))) for k,i in enumerate(indices)]
    replay=Stream(j,768,tokens=stream.tokens);assert slot_writer(indices)(replay)==rows and replay.events==stream.events
    assert unavailable('NONE') and unavailable('UNKNOWN details') and not unavailable('nonevent is visible')
    capped=dict(truncated=True,tokens=[0]*768,max_tokens=768,selection=rows)
    assert all(unavailable(r[role]) for r in compile_slots(capped,indices) for role in SPEC['slot_order'])
    caption=Feed(['actual cup'],96);assert caption_writer(caption)==dict(caption='actual cup')
    cap=Feed(['one two three four five six seven eight'],768);rr=slot_writer([0])
    # Reaching the declared word limit is a rejected field even when syntactically closed.
    cap.texts=iter(['one two three four five six seven eight','NONE','NONE']);value=rr(cap);assert value[0]['prequel']=='UNKNOWN'
    out=ROOT/'runs/20261006_m1_ordered_slots/retrieval_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,ordered_feasible_assignment=True,independent_enumeration=True,stable_ties=True,NONE_zero_strict=True,current_fixed=True,mask_missing=True,actual_token_json=True,exact_replay=True,whole_cap_rejected=True,field_cap_unknown=True),indent=2)+'\n')
    print('RETRIEVAL_CPU_PASS',flush=True)


if __name__=='__main__':main()
