"""Actual native/local sources and source-bound generation input replay."""
from pathlib import Path
import sys
import math
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from interface import SPEC,VERSION,CONSTANTS,canonical,model_visible,catalog,source_table,coverage,write_record,compile_record,partition,compose,repair_plan,final_tree
from src.video_inputs import load_manifest,frame_paths,fixed_windows,window_text
from src.actual_video_frames import validate_frames
from src.mllm_judge import MODEL,YOUTUBE_RULES
DATASETS=('HateMM','HateClipSeg');CACHE=ROOT/'data/temporal_interval_witness'
assert SPEC['policy']==YOUTUBE_RULES


def selected_rows(smoke=False):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333);return rows


def windows_for(row,segments,source,folder):
    entries={e['index']:e for e in source['entries']};windows=[]
    for i,((a,b),indices) in enumerate(zip(fixed_windows(float(row['duration']),8),source['selected_indices'])):
        frames=[dict(id=f'p{k}',path=str((folder/'frames'/f'frame_{index:08d}.png').relative_to(ROOT)),**entries[index]) for k,index in enumerate(indices)]
        windows.append(dict(i=i,start=a,end=b,body=window_text(segments,a,b),frames=frames))
    return windows


def overview_for(row):return [dict(id=f'overview{k}',time=t,path=str(p.relative_to(ROOT)),time_kind='nominal overview only') for k,(t,p) in enumerate(frame_paths(row['dataset'],row['video_id'],20))]


def local_content(overview,window,context=None):
    content=[];paths=[]
    content.append(dict(type='text',text='Native overview for interpretation only; local witnesses below own occurrences.\n'))
    for frame in overview:
        content.extend([dict(type='text',text=canonical({k:v for k,v in frame.items() if k!='path'})+'\n'),dict(type='image')]);paths.append(frame['path'])
    content.append(dict(type='text',text='Actual LOCAL frames:\n'))
    for frame in window['frames']:
        content.extend([dict(type='text',text=canonical({k:v for k,v in frame.items() if k!='path'})+'\n'),dict(type='image')]);paths.append(frame['path'])
    content.append(dict(type='text',text='Source catalog:\n'+canonical(source_table(window))+'\nRules:\n'+SPEC['policy']+'\n'))
    if context is not None:
        assert set(context)==set(SPEC['repair_context_fields'])
        content.append(dict(type='text',text='Interpretation context:\n'+canonical(model_visible(context))+'\n'+SPEC['repair_instruction']))
    else:content.append(dict(type='text',text=SPEC['leaf_instruction']))
    return content,paths


def parent_content(node,children,windows):
    return [dict(type='text',text=canonical(dict(interval=dict(lo=node['lo'],hi=node['hi'],start=windows[node['lo']]['start'],end=windows[node['hi']-1]['end']),
        children=children,descendant_sources=[source_table(w) for w in windows[node['lo']:node['hi']]]))+'\nRules:\n'+SPEC['policy']+'\n'+SPEC['parent_instruction'])]


def parent_record(node,p,children,windows):
    status,cov=compose(children)
    return dict(**node,start=windows[node['lo']]['start'],end=windows[node['hi']-1]['end'],status=status,coverage=cov,
        description=p['description'],witnesses=p['witnesses'],proposed_status=p['status'],proposal_error=p['error'])


def validate(m,row,segments,j=None):
    from src.structured_source_generation import validate_generation
    folder=CACHE/row['dataset']/row['video_id']
    assert (m['version'],m['constants'],m['spec'],m['model'])==(VERSION,CONSTANTS,SPEC,MODEL)
    assert (m['dataset'],m['video_id'],m['duration'])==(row['dataset'],row['video_id'],float(row['duration']))
    assert m['GT_read'] is False and m['segments']==[list(s) for s in segments]
    validate_frames(m['source'],row,folder/'frames');windows=windows_for(row,segments,m['source'],folder)
    assert m['windows']==windows and m['overview']==overview_for(row);nodes=partition(len(windows))
    records={};parents={};leaves=[];generations=[]
    assert len(m['leaves'])==len(windows) and len(m['parents'])==len(windows)-1
    for node in nodes:
        lo,hi=node['lo'],node['hi']
        if hi-lo==1:
            saved=m['leaves'][lo];sources=catalog(windows[lo]);cov=coverage([windows[lo]])
            system=SPEC['leaf_system'];content,paths=local_content(m['overview'],windows[lo])
        else:
            saved=m['parents'][node['id']];sources={h:r for w in windows[lo:hi] for h,r in catalog(w).items()};cov=coverage(windows[lo:hi])
            system=SPEC['parent_system'];content=parent_content(node,[records[c] for c in node['children']],windows);paths=[]
        parsed=compile_record(saved['generation'],sources,cov);assert saved['record']==parsed;generations.append(saved['generation'])
        if j:validate_generation(j,ROOT,saved['generation'],system,content,paths,512,lambda s:write_record(s,sources))
        if hi-lo==1:
            leaves.append(parsed);records[node['id']]=dict(**node,start=windows[lo]['start'],end=windows[lo]['end'],**parsed)
        else:
            parents[node['id']]=saved;records[node['id']]=parent_record(node,parsed,[records[c] for c in node['children']],windows)
    assert leaves==[l['record'] for l in m['leaves']]
    plan=repair_plan(windows,leaves,parents);assert m['repair_plan']==plan and len(m['repairs'])==len(plan)
    final=list(leaves)
    for target,saved in zip(plan,m['repairs']):
        i=target['leaf'];assert saved['target']==target;sources=catalog(windows[i]);cov=coverage([windows[i]])
        parsed=compile_record(saved['generation'],sources,cov);assert saved['record']==parsed;final[i]=parsed;generations.append(saved['generation'])
        if j:
            content,paths=local_content(m['overview'],windows[i],target['context'])
            validate_generation(j,ROOT,saved['generation'],SPEC['repair_system'],content,paths,512,lambda s:write_record(s,sources))
    assert m['final_leaves']==final and m['final_tree']==final_tree(windows,final,parents)
    assert m['actual_forwards']==sum(g['actual_forwards'] for g in generations)
    assert m['actual_vision_forwards']==sum(g['actual_vision_forwards'] for g in generations)
    assert abs(m['generation_seconds']-sum(g['seconds'] for g in generations))<1e-6
    assert m['standalone_seconds']>=m['generation_seconds']+m['source']['decode_seconds']
    assert all(math.isfinite(m[k]) and m[k]>=0 for k in ('standalone_seconds','generation_seconds','peak_GiB'))
    return m
