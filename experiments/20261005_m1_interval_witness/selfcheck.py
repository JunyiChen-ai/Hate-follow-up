#!/usr/bin/env python3
"""Executable composition, local ownership and actual-source CPU checks; no GT."""
import copy
import json
import logging
import socket
import sys
import time
from inputs import ROOT,selected_rows,windows_for,overview_for,local_content,parent_content,parent_record
from interface import SPEC,catalog,coverage,write_record,compile_record,partition,compose,repair_plan,final_tree,resolve_handle,source_path,canonical,model_visible
from src.mllm_renderer import cpu_renderer
from src.actual_video_frames import validate_frames
from src.video_inputs import load_asr,fixed_windows,window_text
from src.structured_source_generation import Stream,image_rope_delta
from PIL import Image


def fixture(status,witnesses=(),description='Observed fixture',cov=None):
    return dict(status=status,description=description,witnesses=list(witnesses),coverage=cov or dict(visual=True,speech=True),error=None,generated_status=status)


def rejected(fn):
    try:fn()
    except (AssertionError,ValueError,KeyError,IndexError):return True
    raise AssertionError('Corrupt fixture was accepted')


def logical_checks(j):
    tests=0
    w=[dict(i=i,start=8*i,end=8*(i+1),body='A person discusses a quoted target',frames=[dict(id='p0',path='fixture.png',index=i,time=8*i+2)]) for i in range(4)]
    sources=catalog(w[0]);assert sources and all(resolve_handle(w,h)==r for h,r in sources.items());tests+=1
    for n in (1,2,3,4,79,134):
        nodes=partition(n);assert len(nodes)==2*n-1
        assert [(x['lo'],x['hi']) for x in nodes if x['hi']-x['lo']==1]==[(i,i+1) for i in range(n)]
        for node in nodes:
            if 'children' in node:
                children=[next(x for x in nodes if x['id']==c) for c in node['children']]
                assert children[0]['lo']==node['lo'] and children[0]['hi']==children[1]['lo'] and children[1]['hi']==node['hi']
        tests+=1
    for statuses,expected in [(('absent','absent'),'absent'),(('present','absent'),'present'),(('UNKNOWN','absent'),'UNKNOWN'),(('UNKNOWN','UNKNOWN'),'UNKNOWN')]:
        assert compose([fixture(s) for s in statuses])[0]==expected;tests+=1
    assert compose([fixture('absent',cov=dict(visual=True,speech=False)),fixture('absent')])[0]=='UNKNOWN';tests+=1
    raw=dict(status='absent',description='',witnesses=[]);g=dict(truncated=False,selection=raw,text=canonical(raw))
    assert compile_record(g,sources,dict(visual=True,speech=False))['status']=='UNKNOWN';tests+=1
    raw=dict(status='present',description='',witnesses=['w0001:p0']);g=dict(truncated=False,selection=raw,text=canonical(raw))
    assert rejected(lambda:compile_record(g,sources,coverage([w[0]])));tests+=1
    raw=dict(status='present',description='',witnesses=[]);g=dict(truncated=False,selection=raw,text=canonical(raw))
    assert rejected(lambda:compile_record(g,sources,coverage([w[0]])));tests+=1
    repeated=dict(w[0],body='same words same words');assert not any(r.get('quote')=='same words' for r in catalog(repeated).values());tests+=1
    overlapping=dict(w[0],body='a a a');assert not any(r.get('quote')=='a a' for r in catalog(overlapping).values());tests+=1
    assert rejected(lambda:resolve_handle([overlapping], 'w0000:t0000_0002'));tests+=1
    leaves=[fixture('absent'),fixture('UNKNOWN'),fixture('present',['w0002:p0']),fixture('absent')]
    parents={};records={}
    for node in partition(4):
        lo,hi=node['lo'],node['hi']
        if hi-lo==1:records[node['id']]=dict(**node,**leaves[lo]);continue
        status='present' if lo==0 else 'absent';witness='w0001:p0' if lo==0 else 'w0002:p0'
        p=fixture(status,[witness]);parents[node['id']]=dict(record=p)
        records[node['id']]=parent_record(node,p,[records[c] for c in node['children']],w)
    plan=repair_plan(w,leaves,parents);assert [(x['leaf'],x['parent']) for x in plan]==[(1,'n0000_0002'),(2,'n0002_0004')];tests+=1
    assert all(set(x['context'])==set(SPEC['repair_context_fields']) for x in plan);tests+=1
    assert len({x['leaf'] for x in plan})==len(plan)<=len(w);tests+=1
    repaired=list(leaves);repaired[1]=fixture('present',['w0001:p0']);repaired[2]=fixture('absent')
    tree=final_tree(w,repaired,parents);assert not any(r.get('disagreement',False) for r in tree);tests+=1
    m=dict(windows=w,final_tree=tree);path=source_path(m,0,'visual')
    assert path['local_final_record']['lo']==0 and path['ancestor_records'][-1]['hi']==4
    assert all(r['role']==('local' if r['owner']==0 else 'interpretation_context') for r in path['literal_witness_sources']);tests+=1
    # Grammar replay fixtures are synthetic choices, never actual model observations.
    handle=next(iter(sources));text='{"status":"present","description":"Observed source","witnesses":['+canonical(handle)+']}'
    tokens=[];tokens+=j.tok.encode('{"status":',add_special_tokens=False);tokens+=j.tok.encode('"present"',add_special_tokens=False)
    tokens+=j.tok.encode(',"description":"',add_special_tokens=False);tokens+=j.tok.encode('Observed source',add_special_tokens=False);tokens+=j.tok.encode('"',add_special_tokens=False)
    tokens+=j.tok.encode(',"witnesses":[',add_special_tokens=False);tokens+=j.tok.encode(canonical(handle),add_special_tokens=False);tokens+=j.tok.encode(']',add_special_tokens=False);tokens+=j.tok.encode('}',add_special_tokens=False)
    j.description_tokens=[i for i in tokens if i not in j.tok.all_special_ids]
    stream=Stream(j,512,tokens=tokens);chosen=write_record(stream,sources)
    assert chosen==json.loads(text) and stream.tokens==tokens;tests+=1
    return tests


def main():
    out=ROOT/'runs/20261005_m1_interval_witness/cpu_checks';out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    cfg=dict(GT_read=False,host=socket.gethostname(),code='experiments/20261005_m1_interval_witness/selfcheck.py; sources2026-10-05',command='python -u '+' '.join(sys.argv))
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');j=cpu_renderer();tests=logical_checks(j)
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};sizes=[];total_spans=0;windows_count=0;largest=None
    for row in selected_rows(False):
        segments=asr[row['dataset']].get(row['video_id'],[])
        for i,(a,b) in enumerate(fixed_windows(float(row['duration']),8)):
            w=dict(i=i,start=a,end=b,body=window_text(segments,a,b),frames=[]);c=catalog(w);total_spans+=len(c);windows_count+=1
            if largest is None or len(c)>largest['spans']:largest=dict(dataset=row['dataset'],video_id=row['video_id'],window=i,spans=len(c),body_chars=len(w['body']))
    for row in selected_rows(True):
        # Existing actual source pixels are read for author CPU input checks only.
        # Scientific extraction will decode its own source and charge its own cost.
        folder=ROOT/'data/temporal_entity_discourse_graph'/row['dataset']/row['video_id']
        old=json.loads((folder/'metadata.json').read_text());validate_frames(old['source'],row,folder/'frames')
        segments=asr[row['dataset']].get(row['video_id'],[]);windows=windows_for(row,segments,old['source'],folder);overview=overview_for(row)
        w=windows[0];content,paths=local_content(overview,w);text=j.render([j.turn('system',SPEC['leaf_system']),dict(role='user',content=content)],True)
        images=[Image.open(ROOT/p).convert('RGB') for p in paths]
        try:enc=j.encode(text,images)
        finally:
            for im in images:im.close()
        assert all('/HateMM/' not in part.get('text','') and 'hate_video_' not in part.get('text','') and '.png' not in part.get('text','') for part in content);tests+=1
        sizes.append(dict(dataset=row['dataset'],video_id=row['video_id'],input_tokens=len(enc['input_ids'][0]),images=len(paths),rope_delta=image_rope_delta(j,enc),source_checked=str(folder.relative_to(ROOT))))
        logging.info('actual CPU source %s/%s images=%d input=%d',row['dataset'],row['video_id'],len(paths),len(enc['input_ids'][0]))
    result=dict(**cfg,logical_checks=tests,all_source_windows=windows_count,total_legal_text_spans=total_spans,largest=largest,actual_fixed5_source_inputs=sizes,
        semantic_claim=False,GPU_run=False,fixture_note='Grammar/composition fixture choices synthetic; real fixed5 pixels/rawPTS verified, no model-generated semantics measured.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');logging.info('CPU_CHECKS_DONE checks=%d windows=%d',tests,windows_count)


if __name__=='__main__':main()
