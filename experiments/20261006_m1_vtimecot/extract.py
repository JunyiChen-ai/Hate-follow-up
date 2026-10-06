"""Real retrieval → executed timeline tools → actual current/history views."""
from timeline import INTERFACE,OUTPUT_SUFFIX
import argparse
import copy
import json
import logging
import socket
import time
import numpy as np
import torch
from timeline import ROOT,SPEC,uniform,retrieval,execute
from inputs import CACHE,DATASETS,selected_rows,acquire_sources,validate_sources,windows_for,media_content
from interface import query_writer,compile_queries,compile_relevance,plan_writer,compile_plan,feedback_writer
from relevance import read_clip,validate_clip
from compound_closure import generate,validate_generation
from src.video_inputs import frame_paths,load_asr
from src.mllm_judge import Judge,MODEL


def query_content(row,segments):
    items=[dict(type='text',text='Original literal ASR with approximate segment bounds:\n'+json.dumps(segments,ensure_ascii=False)+'\nExisting native overview images follow. Their sampling coordinates are nominal, not exact temporal witnesses.')];paths=[]
    for _,path in frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']):
        relative=str(path.relative_to(ROOT));items.append(dict(type='image',image=relative));paths.append(relative)
    return items,paths


def planner_content(source,folder,state,queries,tables,history,step,save):
    # Model sees executed actions/source coordinates, never saved prompts,
    # path provenance, expanded token arrays or nested generation records.
    visible_history=[dict(step=i,unverified_reason=s['plan']['reason'],action={k:v for k,v in s['plan'].items() if k!='reason'},result=s['result'],after=s['after']) for i,s in enumerate(history)]
    context=dict(current_state=state,search_conditions_are_unverified=queries,retrieval_intervals=tables,previous_tool_history=visible_history)
    items=[dict(type='text',text=json.dumps(context,ensure_ascii=False))]
    media,paths=media_content(source,state['memory_ids'],folder,state,save=save,stem='state_'+str(step));return items+media,paths


def final_content(source,folder,window,state,steps,queries,save):
    content=[dict(type='text',text='Actual source observations and actual executed tool history. Previous and remote views retain original timestamps and are not events in the current window. Search conditions are unverified requests, not factual evidence.\n')];paths=[]
    for i,step in enumerate(steps):
        if not step['result']['executed']:continue
        before=step['before'];content.append(dict(type='text',text=json.dumps(dict(previous_step=i,actual_action=step['plan']['action'],actual_parameters={k:v for k,v in step['plan'].items() if k not in ('reason','action')},unverified_search_condition=queries[step['plan']['query_id']] if step['plan']['action']=='HIGHLIGHT' else None),ensure_ascii=False)))
        ids=uniform(before['memory_ids'],SPEC['history_frames_per_state']);media,pp=media_content(source,ids,folder,before,save=save,stem='history_'+str(i));content+=media;paths+=pp
    ids=list(dict.fromkeys(window['sample_ids']+uniform(state['memory_ids'],SPEC['cut_memory_frames'])))
    content.append(dict(type='text',text=json.dumps(dict(current_LOCAL_window=window['i'],bounds=[window['start'],window['end']],literal_current_ASR=window['body']),ensure_ascii=False)))
    media,pp=media_content(source,ids,folder,state,current=[window['start'],window['end']],save=save,stem='final_'+str(window['i']));content+=media;paths+=pp
    assert len(paths)<=SPEC['final_source_images_cap'];return dict(content=content,paths=paths)


def acquire(j,row,segments,folder):
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();source=acquire_sources(row,folder/'frames');windows=windows_for(row,segments,source)
    items,paths=query_content(row,segments);query_generation=generate(j,ROOT,SPEC['query_system'],items,paths,SPEC['query_generation_tokens'],query_writer);queries=compile_queries(query_generation)
    clips=[read_clip(j,w,source,folder,queries) for w in windows];values=[[compile_relevance(c['generations'][q]) if c is not None else None for c in clips] for q in range(len(queries))]
    tables=[retrieval(v,windows) for v in values];state=dict(progress=False,query_id=None,cut=None,highlights=[],memory_ids=uniform(source['selected_indices'],SPEC['initial_memory_frames']));steps=[]
    for i in range(SPEC['tool_steps']):
        items,paths=planner_content(source,folder,state,queries,tables,steps,i,True)
        g=generate(j,ROOT,SPEC['planner_system'],items,paths,SPEC['planner_generation_tokens'],plan_writer(state,tables));plan=compile_plan(g);before=copy.deepcopy(state)
        state,result=execute(state,plan,windows,tables,source['selected_indices']);steps.append(dict(before=before,generation=g,plan=plan,result=result,after=copy.deepcopy(state)))
        if not result['executed']:break
    items,paths=planner_content(source,folder,state,queries,tables,steps,len(steps),True)
    feedback=generate(j,ROOT,SPEC['feedback_system'],items,paths,SPEC['feedback_generation_tokens'],feedback_writer)
    final=[final_content(source,folder,w,state,steps,queries,True) for w in windows]
    gens=[query_generation,*[s['generation'] for s in steps],feedback];cc=[c for c in clips if c is not None]
    return dict(version=SPEC['version'],spec=SPEC,model=MODEL,host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),dataset=row['dataset'],video_id=row['video_id'],GT_read=False,segments=[list(s) for s in segments],source=source,windows=windows,
        query_generation=query_generation,queries=queries,clips=clips,relevance_values=values,tables=tables,steps=steps,final_state=state,feedback=feedback,final_inputs=final,peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        cost=dict(standalone_seconds=time.perf_counter()-start,decode_seconds=source['decode_seconds'],generation_seconds=sum(g['seconds'] for g in gens),clip_prefix_seconds=sum(c['prefix_seconds'] for c in cc),relevance_seconds=sum(g['seconds'] for c in cc for g in c['generations']),
            query_calls=1,clip_prefix_calls=len(cc),relevance_calls=len(cc)*len(queries),planner_calls=len(steps),feedback_calls=1,
            actual_forwards=sum(g['actual_forwards'] for g in gens)+sum(c['prefix_forwards']+sum(g['actual_forwards'] for g in c['generations']) for c in cc),actual_vision_forwards=sum(g['actual_vision_forwards'] for g in gens)+sum(c['prefix_vision'] for c in cc)))


def validate(j,m,row,segments,folder):
    assert m['spec']==SPEC and m['version']==SPEC['version'] and m['model']==MODEL and m['GT_read'] is False
    assert (m['dataset'],m['video_id'],m['segments'])==(row['dataset'],row['video_id'],[list(s) for s in segments])
    validate_sources(m['source'],row,folder/'frames');source=m['source'];windows=windows_for(row,segments,source);assert windows==m['windows']
    items,paths=query_content(row,segments);validate_generation(j,ROOT,m['query_generation'],SPEC['query_system'],items,paths,SPEC['query_generation_tokens'],query_writer)
    queries=compile_queries(m['query_generation']);assert queries==m['queries'] and len(m['clips'])==len(windows)
    for window,c in zip(windows,m['clips']):validate_clip(j,c,window,source,folder,queries)
    values=[[compile_relevance(c['generations'][q]) if c is not None else None for c in m['clips']] for q in range(len(queries))];assert values==m['relevance_values']
    tables=[retrieval(v,windows) for v in values];assert tables==m['tables']
    state=dict(progress=False,query_id=None,cut=None,highlights=[],memory_ids=uniform(source['selected_indices'],SPEC['initial_memory_frames']));history=[];continuing=True
    assert 1<=len(m['steps'])<=SPEC['tool_steps']
    for i,step in enumerate(m['steps']):
        assert continuing and step['before']==state
        items,paths=planner_content(source,folder,state,queries,tables,history,i,False)
        validate_generation(j,ROOT,step['generation'],SPEC['planner_system'],items,paths,SPEC['planner_generation_tokens'],plan_writer(state,tables));plan=compile_plan(step['generation']);assert plan==step['plan']
        state,result=execute(state,plan,windows,tables,source['selected_indices']);assert result==step['result'] and state==step['after'];history.append(step);continuing=result['executed']
    assert len(history)==SPEC['tool_steps'] or not continuing;assert state==m['final_state']
    items,paths=planner_content(source,folder,state,queries,tables,history,len(history),False);validate_generation(j,ROOT,m['feedback'],SPEC['feedback_system'],items,paths,SPEC['feedback_generation_tokens'],feedback_writer)
    assert len(m['final_inputs'])==len(windows)
    for w,final in zip(windows,m['final_inputs']):assert final==final_content(source,folder,w,state,history,queries,False)
    gens=[m['query_generation'],*[s['generation'] for s in history],m['feedback']];cc=[c for c in m['clips'] if c is not None];cost=m['cost']
    assert cost['actual_forwards']==sum(g['actual_forwards'] for g in gens)+sum(c['prefix_forwards']+sum(g['actual_forwards'] for g in c['generations']) for c in cc)
    assert cost['actual_vision_forwards']==sum(g['actual_vision_forwards'] for g in gens)+sum(c['prefix_vision'] for c in cc)
    assert (cost['query_calls'],cost['clip_prefix_calls'],cost['relevance_calls'],cost['planner_calls'],cost['feedback_calls'])==(1,len(cc),len(cc)*len(queries),len(history),1)
    assert all(np.isfinite(v) and v>=0 for v in cost.values())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();out=ROOT/'runs/20261006_m1_vtimecot'/(('source_smoke' if a.smoke else 'source_main')+OUTPUT_SUFFIX);out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()]);logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    (out/'config.json').write_text(json.dumps(dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,spec=SPEC,GT_read=False,smoke=a.smoke,torch=torch.__version__,transformers=transformers.__version__,code='experiments/20261006_m1_vtimecot/{extract,relevance,timeline,interface,inputs}.py;2026-10-06',command='python -u '+' '.join(__import__('sys').argv)),indent=2)+'\n')
    torch.manual_seed(0);torch.set_num_threads(4);j=Judge(MODEL);j.forward_calls=j.vision_calls=0;hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    rows=selected_rows(a.smoke);asr={ds:load_asr(ds) for ds in DATASETS}
    for number,row in enumerate(rows,1):
        segments=asr[row['dataset']].get(row['video_id'],[]);folder=CACHE/row['dataset']/row['video_id'];folder.mkdir(parents=True,exist_ok=True);path=folder/'metadata.json'
        if path.exists():m=json.loads(path.read_text())
        else:
            m=acquire(j,row,segments,folder);validate(j,m,row,segments,folder);temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(m,allow_nan=False)+'\n');temporary.replace(path)
        validate(j,m,row,segments,folder);logging.info('%d/%d %s/%s %.2fs queries=%d executed=%s',number,len(rows),row['dataset'],row['video_id'],m['cost']['standalone_seconds'],len(m['queries']),[s['plan']['action'] for s in m['steps'] if s['result']['executed']])
    for hook in hooks:hook.remove()
    (CACHE/'PROVENANCE.md').write_text('# Actual visual time tools\n\nGenerated by experiments/20261006_m1_vtimecot/extract.py, R1 sources2026-10-06, Qwen/Qwen3-VL-8B-Instruct. Metadata includes actual generatinghost/date, readable original/native/raw paths, literalASR, actualsource PTS/RGB, sourceprefix/currentquery tokens, clip-cache branch positions, query/relevance records, real typedtool history and allpixel views, common spec and fullnew-video costs. No GT/labels/contentchecksums. Final evidence contains real original pixels, rendered real time coordinates and actual executed requests; generated reason/feedback/relevance scores excluded.\n')
    logging.info('EXTRACTION_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
