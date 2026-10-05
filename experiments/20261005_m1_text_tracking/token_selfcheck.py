"""Actual tokenizer grammar fixtures; no model weights, GT, or predictions."""
import json
from interface import writer,compile_records,ROOT
from src.structured_source_generation import Stream,Capped
from src.mllm_renderer import cpu_renderer


def main():
    j=cpu_renderer();frame=dict(id='p0',shape=[200,100],time=.25,index=6,path='synthetic.png')
    class Feed(Stream):
        def __init__(self,text,limit=384):
            super().__init__(j,limit,tokens=[])
            self.answers=iter(['TEXT",','300,','300,','700,','700]']);self.literal_text=text
        def force(self,text):
            assert len(self.saved)==len(self.tokens)
            self.saved.extend(j.tok.encode(text,add_special_tokens=False));super().force(text)
            if text==',"text":"':
                escaped=json.dumps(self.literal_text,ensure_ascii=False)[1:-1]
                self.saved.extend(j.tok.encode(escaped,add_special_tokens=False)+j.tok.encode('"',add_special_tokens=False))
        def choose(self,options):
            text=next(self.answers);assert text in options
            self.saved.extend(j.tok.encode(text,add_special_tokens=False));return super().choose(options)
    checks=[]
    for text,unknown in [('Hello "A" \\ B\n中',False),('x '*31+'x',True),('UNKNOWN literal',True)]:
        s=Feed(text);value=writer([frame])(s);assert len(s.saved)==len(s.tokens)
        replay=Stream(j,384,tokens=s.tokens);assert writer([frame])(replay)==value and replay.events==s.events
        parsed=compile_records(dict(tokens=s.tokens,max_tokens=384,truncated=False,selection=value),[frame])
        assert (parsed[0]['status']=='UNKNOWN')==unknown
        assert json.loads(j.tok.decode(s.tokens))['frames'][0]['text']==text
        checks.append(dict(unknown=unknown,tokens=len(s.tokens),wire_text_exact=True,grammar_replay_exact=True))
    s=Feed('A',limit=20)
    try:writer([frame])(s);raise AssertionError('cap did not trigger')
    except Capped:assert len(s.tokens)==20
    assert compile_records(dict(tokens=s.tokens,max_tokens=20,truncated=True,selection=None),[frame])==[dict(id='p0',status='UNKNOWN')]
    summary=dict(PASS=True,GT_read=False,scope='actual native tokenizer guided grammar replay; no modelweights/GT/scores',checks=checks,whole_cap_unknown=True)
    out=ROOT/'runs/20261005_m1_text_tracking/token_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
