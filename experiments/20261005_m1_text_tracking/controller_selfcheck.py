"""Known synthetic pixels and recognition fixtures, not hate/OCR accuracy."""
import json
import numpy as np
from PIL import Image,ImageDraw
from controller import Controller,interval_lookup
from inputs import ROOT


def main():
    folder=ROOT/'runs/20261005_m1_text_tracking/controller_cpu_checks';folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(0);texture=rng.integers(0,256,(40,80,3),dtype=np.uint8)
    def frame(letter,shift=(0,0)):
        im=Image.fromarray(texture.copy());ImageDraw.Draw(im).text((8,8),letter,fill=(255,255,255))
        rgb=np.full((100,200,3),40,np.uint8);x,y=60+shift[0],30+shift[1]
        rgb[y:y+40,x:x+80]=np.asarray(im);return rgb
    calls=[]
    def observe(w,e,rgb,failure,n):calls.append(e['index']);return dict(id='p0',status='NONE'),'synthetic_repair_'+str(n)
    windows=[dict(i=0,start=0.,end=8.)];c=Controller(folder,windows,observe)
    a=dict(id='p0',status='TEXT',box=[300,300,700,700],text='A')
    b=dict(id='p0',status='TEXT',box=[315,320,715,720],text='B')
    white=np.full((100,200,3),255,np.uint8)
    frames=[frame('A'),frame('A',(3,2)),frame('B',(3,2)),white,frame('A'),white,frame('A'),white]
    anchors={0:[(a,'synthetic_A0')],2:[(b,'synthetic_B2')],4:[(a,'synthetic_A4')],6:[(a,'synthetic_A6')]}
    for i,rgb in enumerate(frames):c.step(dict(index=i,time=i/4,width=200,height=100),rgb,anchors.get(i,[]))
    r=c.finish();assert [p['index'] for p in r['events'][0]['supports']]==[0,1]
    assert r['events'][0]['ending']['reason']=='fresh_literal_change'
    assert [p['index'] for p in r['events'][1]['supports']]==[2] and r['events'][1]['ending']['reason']=='cut'
    assert r['events'][2]['id']!=r['events'][0]['id'] and r['events'][2]['initial_index']==4
    assert len(calls)==2 and any(x['reason']=='budget_unknown' for x in r['repairs'])
    lookup=interval_lookup(r['events'],windows[0]);assert len(lookup)==4
    for event in lookup:
        for key in ('first','last'):
            p=event[key];rgb=frames[p['index']];x0,y0,x1,y1=p['box']
            with Image.open(ROOT/p['path']) as im:assert np.array_equal(np.asarray(im),rgb[y0:y1,x0:x1])
    summary=dict(PASS=True,GT_read=False,scope='known synthetic pixel and external recognition fixtures; not OCR accuracy/model/performance',
        events=len(r['events']),repair_calls=len(calls),first_supports=[p['index'] for p in r['events'][0]['supports']],
        checks=['literalchange doesnotrewritepast or acceptcurrentoldword','cut endsold support','reappearance newoccurrence','two-call repairbudget','exactactual endpointcrop pixels'])
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (folder/'records.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
