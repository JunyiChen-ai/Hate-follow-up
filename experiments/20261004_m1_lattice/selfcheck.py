#!/usr/bin/env python3
"""Independent tiny-DAG path oracle and real delayed-audio timeline fixture."""
import itertools
import math
from pathlib import Path
import subprocess
import numpy as np
from lattice import align_words,confusion,tokenize_graph,graph_bias,graph_positions,validate_graph_trace
from audio import decode_audio,crop_audio,RATE
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


class Tokens:
    def encode(self,text,add_special_tokens=False):
        return [ord(c) for c in text.strip().replace(' ','')]


def graph_oracle():
    slots=[dict(alternatives=[dict(text='a',mass=.4),dict(text='bcd',mass=.3),dict(text='',mass=.3)]),
        dict(alternatives=[dict(text='ef',mass=.5),dict(text='g',mass=.5)])]
    g=tokenize_graph(dict(slots=slots),Tokens());bias=graph_bias(g,2,3,4)
    # Enumerate actual option choices; do not derive the oracle from the mask code.
    paths=[]
    for choices in itertools.product(*[range(len(s['alternatives'])) for s in slots]):
        weight=math.prod(slots[s]['alternatives'][a]['mass'] for s,a in enumerate(choices))
        members={n['physical'] for n in g['nodes'] if choices[n['slot']]==n['alternative']}
        paths.append((weight,members))
    assert np.isclose(sum(w for w,_ in paths),1)
    for i,node in enumerate(g['nodes']):
        den=sum(w for w,p in paths if i in p)
        for j,prior in enumerate(g['nodes']):
            allowed=(prior['slot']<node['slot'] or
                (prior['slot']==node['slot'] and prior['alternative']==node['alternative'] and j<=i))
            p=sum(w for w,path in paths if i in path and j in path)/den if allowed else 0.
            got=bias[2+i,4+2+j]
            if p:assert abs(got-math.log(p))<1e-6,(i,j,p,got)
            else:assert np.isneginf(got),(i,j,got)
    for j,node in enumerate(g['nodes']):
        p=sum(w for w,path in paths if j in path)
        assert abs(bias[-1,4+2+j]-math.log(p))<1e-6
    # General predecessor graph longest distances, including zero-length epsilon joins.
    distances=[];join=0
    for s,slot in enumerate(slots):
        ends=[]
        for a,alt in enumerate(slot['alternatives']):
            previous=join
            for n in [x for x in g['nodes'] if x['slot']==s and x['alternative']==a]:
                previous+=1;distances.append((n['physical'],previous-1))
            ends.append(previous)
        join=max(ends)
    assert [d for _,d in sorted(distances)]==[n['logical'] for n in g['nodes']]
    assert join==g['path_length']==5
    assert graph_positions(g,2,3,7)==[7,8,9,9,10,11,12,13,12,14,15,16]
    assert np.isneginf(bias[2,4+3])  # no first option -> other option at same slot
    assert np.all(bias[:,:4]==0)
    only=tokenize_graph(dict(slots=[dict(alternatives=[dict(text='abc',mass=1)])]),Tokens())
    flat=graph_bias(only,2,3,4,'flat');full=graph_bias(only,2,3,4)
    np.testing.assert_array_equal(full,flat)
    assert graph_positions(only,2,3,7)==graph_positions(only,2,3,7,'flat')
    zero=tokenize_graph(dict(slots=[dict(alternatives=[dict(text='ghost',mass=0.),dict(text='',mass=1.)])]),Tokens())
    assert zero['ids']==[] and zero['path_length']==0 and zero['slots'][0]['alternatives'][0]['mass']==0.
    empty=graph_bias(zero,2,3,4);assert np.all(empty[:,:4]==0)
    print('PASS independently enumerated predecessor probabilities, epsilon mass, longest paths, one-path degeneration',flush=True)


def alignment_oracle():
    beams=[dict(text=t,score=0.) for t in ('not all people','all people','not old people','not all real people','')]
    c=confusion(beams)
    for i,b in enumerate(beams):
        chosen=[next(a['text'] for a in s['alternatives'] if i in a['beams']) for s in c['slots']]
        assert ' '.join(x for x in chosen if x)==' '.join(b['text'].split())
    assert any(a['text']=='' and a['mass']>0 for s in c['slots'] for a in s['alternatives'])
    assert any(s['kind']=='insert' and any(a['text']=='real' for a in s['alternatives']) for s in c['slots'])
    assert align_words(['NOT','all'],['not','all'])==(['not','all'],['','',''])
    assert align_words([],['new','words'])==([],['new words'])
    assert align_words(['not','all'],[])==(['',''],['','',''])
    print('PASS hand lexical/deletion/insertion cases and reconstruction of every original beam',flush=True)


def resume_binding():
    import copy
    c=confusion([dict(text=t,score=0.) for t in ('not all','all','not old','not all real','')])
    g=tokenize_graph(c,Tokens());head='head';tail='tail'
    trace=dict(graph=g,arm='full',prefix_tokens=10,prefix_logical_start=7,
        head_text=head,tail_text=tail,head_tokens=Tokens().encode(head),tail_tokens=Tokens().encode(tail),
        graph_tokens=len(g['ids']),slot_count=len(g['slots']),physical_tokens=8+len(g['ids']),
        logical_positions=graph_positions(g,4,4,7))
    validate_graph_trace(trace,c,Tokens(),10,7)
    changed=copy.deepcopy(c);changed['slots'][0]['alternatives'][0]['mass']+=.01
    bad=copy.deepcopy(trace);bad['graph']['ids'][0]+=1
    mass=copy.deepcopy(trace);mass['graph']['nodes'][0]['mass']+=.01
    logical=copy.deepcopy(trace);logical['graph']['nodes'][0]['logical']+=1
    positions=copy.deepcopy(trace);positions['logical_positions'][-1]+=1
    for saved,current in ((trace,changed),(bad,c),(mass,c),(logical,c),(positions,c)):
        try:validate_graph_trace(saved,current,Tokens(),10,7)
        except AssertionError:pass
        else:raise AssertionError('accepted a stale or inconsistent graph trace')
    print('PASS resume rejects changed current mass, saved token/node mass/logical fields and position geometry',flush=True)


def actual_audio():
    out=ROOT/'runs/20261004_m1_lattice/cpu_checks';out.mkdir(parents=True,exist_ok=True)
    path=out/'delayed_audio.mkv'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc=size=64x64:rate=8:duration=2',
        '-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=1',
        '-filter_complex','[1:a]asetpts=PTS+0.5/TB[a]','-map','0:v','-map','[a]',
        '-c:v','libx264','-pix_fmt','yuv420p','-c:a','pcm_f32le',str(path)],check=True)
    samples,observed,meta=decode_audio(path,2.)
    assert meta['video_origin']==0 and np.flatnonzero(observed)[0]==8000
    assert np.flatnonzero(observed)[-1]==23999 and int(observed.sum())==16000
    a,r=crop_audio(samples,observed,0,.4);assert not len(a) and not r['samples']
    a,r=crop_audio(samples,observed,.2,.8);assert len(a)==4800 and r['samples']==[8000,12800]
    assert r['actual_interval']==[.5,.8] and r['observed_intervals']==[[8000,12800]] and not r['gaps']
    observed[10000:11000]=False;samples[10000:11000]=0
    a,r=crop_audio(samples,observed,.5,.8);assert len(a)==4800 and r['gaps']==[[10000,11000]]
    assert r['observed_intervals']==[[8000,10000],[11000,12800]]
    print('PASS actual PyAV resampling/PTS delayed audio, absent crop, partial crop and uncompressed internal gap',flush=True)


if __name__=='__main__':
    graph_oracle();alignment_oracle();resume_binding();actual_audio()
