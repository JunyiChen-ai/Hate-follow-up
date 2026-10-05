"""Known vector retrieval/source-ID grammar, not embedding or model accuracy."""
import json
import numpy as np
from retrieval import ROOT,retrieve,compile_filter,union_sources
from interface import filter_writer
from src.structured_source_generation import Stream
from src.mllm_renderer import cpu_renderer


def main():
    keys=np.zeros((8,4,2),np.float32);keys[:,:,1]=1;keys[2,0]=[1,0];keys[6,1]=[1,0]
    result=retrieve(keys,np.ones((8,4),bool),[1,0],current=0)
    assert result['anchors']==[2,6] and result['candidates']==[1,2,3,6]
    assert result['best_keys'][:2]==['event','dialogue']
    assert compile_filter(dict(status='UNKNOWN',ids=[2]),result['candidates'])==[]
    assert union_sources([1,2,3],[3,6,7])==[1,2,3,6]
    assert retrieve(keys,np.zeros((8,4),bool),[1,0],current=0)['anchors']==[]
    j=cpu_renderer()
    class Feed(Stream):
        def __init__(self):super().__init__(j,96,tokens=[]);self.plan=iter(['SUPPORTED",','1 ',',12 ',']'])
        def force(self,text):self.saved.extend(j.tok.encode(text,add_special_tokens=False));return super().force(text)
        def choose(self,options):
            value=next(self.plan);assert value in options
            self.saved.extend(j.tok.encode(value,add_special_tokens=False));return super().choose(options)
    stream=Feed();value=filter_writer([1,12,123])(stream)
    assert json.loads(j.tok.decode(stream.tokens))==value
    replay=Stream(j,96,tokens=stream.tokens);assert filter_writer([1,12,123])(replay)==value and replay.events==stream.events
    out=ROOT/'runs/20261006_m1_merit/retrieval_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    summary=dict(PASS=True,GT_read=False,scope='known vectors/actual tokenizer source-ID grammar; no model or benchmark accuracy',maxsim=True,stable_ties=True,neighborhood=True,unknown=True,cap4_union=True,source_id_json_exact=True,grammar_replay_exact=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('RETRIEVAL_CPU_PASS',flush=True)


if __name__=='__main__':main()
