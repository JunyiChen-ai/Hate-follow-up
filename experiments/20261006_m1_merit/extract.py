"""Actual media → neutral keys → same-model embeddings → bounded source IDs."""
import argparse
import json
import logging
import socket
import time
import numpy as np
import torch
from inputs import ROOT,SPEC,CACHE,DATASETS,selected_rows,windows_for,caption_content,filter_content
from interface import key_writer,filter_writer,compile_keys,compile_filter,unavailable
from retrieval import retrieve,compile_filter as selected_ids,union_sources
from embedding import embed,validate_embedding
from src.actual_video_frames import acquire_window_frames,validate_frames
from src.structured_source_generation import generate,validate_generation
from src.mllm_judge import Judge,MODEL
from src.video_inputs import load_asr


def initial_query(window,caption):
    parts=[part for part in (window['body'],caption) if not unavailable(part)]
    return SPEC['query_prefix']+'\n'+'\n'.join(parts) if parts else None


def index_arrays(records,embeddings,dimension):
    values=np.zeros((len(records),4,dimension),np.float32);available=np.zeros((len(records),4),bool)
    for i,rr in enumerate(embeddings):
        for k,key in enumerate(SPEC['key_order']):
            e=rr[key]
            if e is not None and e['available']:values[i,k]=e['vector'];available[i,k]=True
    return values,available


def acquire(j,row,segments,folder):
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    source=acquire_window_frames(row,folder/'frames',SPEC['window_seconds']);windows=windows_for(row,segments,source,folder)
    generations=[];records=[];embeddings=[]
    for window in windows:
        items,paths=caption_content(window)
        g=generate(j,ROOT,SPEC['caption_system'],items,paths,SPEC['caption_key_generation_tokens'],key_writer)
        r=compile_keys(g);generations.append(g);records.append(r)
        embeddings.append({key:None if unavailable(r[key]) else embed(j,r[key]) for key in SPEC['key_order']})
    dimension=j.model.config.text_config.hidden_size;values,available=index_arrays(records,embeddings,dimension)
    packets=[];queries=[];filters=[]
    for i,window in enumerate(windows):
        query=initial_query(window,records[i]['caption']);seen=set();chosen=[];rounds=[]
        for round_index in range(SPEC['max_rounds']):
            if query is None:break
            e=embed(j,query);queries.append(e);vector=e['vector'] if e['available'] else None
            selection=retrieve(values,available,vector,i,seen);candidates=selection['candidates'];seen.update(candidates)
            if not candidates:
                rounds.append(dict(query=query,embedding_index=len(queries)-1,retrieval=selection,filter_index=None,selected=[]));break
            items=filter_content(window,records[i]['caption'],query,windows,records,candidates)
            g=generate(j,ROOT,SPEC['filter_system'],items,[],SPEC['filter_generation_tokens'],filter_writer(candidates))
            result=compile_filter(g);ids=selected_ids(result,candidates);chosen=union_sources(chosen,ids)
            filters.append(g);rounds.append(dict(query=query,embedding_index=len(queries)-1,retrieval=selection,filter_index=len(filters)-1,selected=ids))
            if result['status']!='INSUFFICIENT' or unavailable(result['new_query']):break
            query=result['new_query']
        packets.append(dict(window=i,rounds=rounds,source_ids=chosen))
    gens=generations+filters;ee=[e for rr in embeddings for e in rr.values() if e is not None]+queries
    return dict(version=SPEC['version'],spec=SPEC,model=MODEL,host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),
        dataset=row['dataset'],video_id=row['video_id'],GT_read=False,segments=[list(s) for s in segments],source=source,windows=windows,
        generations=generations,records=records,key_embeddings=embeddings,dimension=dimension,query_embeddings=queries,filters=filters,packets=packets,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        cost=dict(standalone_seconds=time.perf_counter()-start,generation_seconds=sum(g['seconds'] for g in gens),embedding_seconds=sum(e['seconds'] for e in ee),
            caption_calls=len(generations),filter_calls=len(filters),embedding_calls=len(ee),actual_forwards=sum(g['actual_forwards'] for g in gens)+len(ee),
            actual_vision_forwards=sum(g['actual_vision_forwards'] for g in gens)))


def validate(j,m,row,segments,folder):
    assert m['version']==SPEC['version'] and m['spec']==SPEC and m['model']==MODEL and m['GT_read'] is False
    assert (m['dataset'],m['video_id'],m['segments'])==(row['dataset'],row['video_id'],[list(s) for s in segments])
    validate_frames(m['source'],row,folder/'frames');windows=windows_for(row,segments,m['source'],folder);assert windows==m['windows']
    assert len(windows)==len(m['generations'])==len(m['records'])==len(m['key_embeddings'])
    for window,g,r,ee in zip(windows,m['generations'],m['records'],m['key_embeddings']):
        items,paths=caption_content(window)
        validate_generation(j,ROOT,g,SPEC['caption_system'],items,paths,SPEC['caption_key_generation_tokens'],key_writer)
        assert r==compile_keys(g)
        for key in SPEC['key_order']:
            if unavailable(r[key]):assert ee[key] is None
            else:assert ee[key] is not None;validate_embedding(j,ee[key],r[key])
    values,available=index_arrays(m['records'],m['key_embeddings'],m['dimension']);query_indices=[];filter_indices=[]
    assert len(m['packets'])==len(windows)
    for i,(window,packet) in enumerate(zip(windows,m['packets'])):
        assert packet['window']==i and len(packet['rounds'])<=SPEC['max_rounds']
        query=initial_query(window,m['records'][i]['caption']);seen=set();chosen=[];continuing=True
        for rr in packet['rounds']:
            assert continuing and query is not None and rr['query']==query
            index=rr['embedding_index'];assert index==len(query_indices);query_indices.append(index)
            e=m['query_embeddings'][index];validate_embedding(j,e,query)
            selection=retrieve(values,available,e['vector'] if e['available'] else None,i,seen)
            assert selection==rr['retrieval'];candidates=selection['candidates'];seen.update(candidates)
            if not candidates:assert rr['filter_index'] is None and rr['selected']==[];continuing=False;continue
            index=rr['filter_index'];assert index==len(filter_indices);filter_indices.append(index);g=m['filters'][index]
            items=filter_content(window,m['records'][i]['caption'],query,windows,m['records'],candidates)
            validate_generation(j,ROOT,g,SPEC['filter_system'],items,[],SPEC['filter_generation_tokens'],filter_writer(candidates))
            result=compile_filter(g);ids=selected_ids(result,candidates);assert ids==rr['selected'];chosen=union_sources(chosen,ids)
            continuing=result['status']=='INSUFFICIENT' and not unavailable(result['new_query']);query=result['new_query'] if continuing else None
        assert packet['source_ids']==chosen and all(k!=i for k in chosen)
        assert len(packet['rounds'])==SPEC['max_rounds'] or not continuing or initial_query(window,m['records'][i]['caption']) is None
    assert len(query_indices)==len(m['query_embeddings']) and len(filter_indices)==len(m['filters'])
    gens=m['generations']+m['filters'];ee=[e for rr in m['key_embeddings'] for e in rr.values() if e is not None]+m['query_embeddings']
    cost=m['cost'];assert cost['actual_forwards']==sum(g['actual_forwards'] for g in gens)+len(ee)
    assert cost['actual_vision_forwards']==sum(g['actual_vision_forwards'] for g in gens)
    assert cost['caption_calls']==len(windows) and cost['filter_calls']==len(m['filters']) and cost['embedding_calls']==len(ee)
    assert all(np.isfinite(v) and v>=0 for v in cost.values())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261006_m1_merit'/('source_smoke' if a.smoke else 'source_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,spec=SPEC,GT_read=False,smoke=a.smoke,torch=torch.__version__,transformers=transformers.__version__,code='experiments/20261006_m1_merit/{extract,embedding,retrieval,interface,inputs}.py;sources2026-10-06',command='python -u '+' '.join(__import__('sys').argv))
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');torch.manual_seed(0);torch.set_num_threads(4);j=Judge(MODEL);j.forward_calls=j.vision_calls=0
    hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    rows=selected_rows(a.smoke);asr={ds:load_asr(ds) for ds in DATASETS}
    for number,row in enumerate(rows,1):
        segments=asr[row['dataset']].get(row['video_id'],[]);folder=CACHE/row['dataset']/row['video_id'];folder.mkdir(parents=True,exist_ok=True);path=folder/'metadata.json'
        if path.exists():m=json.loads(path.read_text())
        else:
            m=acquire(j,row,segments,folder);validate(j,m,row,segments,folder);p=path.with_suffix('.partial');p.write_text(json.dumps(m,allow_nan=False)+'\n');p.replace(path)
        validate(j,m,row,segments,folder);logging.info('%d/%d %s/%s %.2fs remote_windows=%d',number,len(rows),row['dataset'],row['video_id'],m['cost']['standalone_seconds'],sum(bool(p['source_ids']) for p in m['packets']))
    for hook in hooks:hook.remove()
    (CACHE/'PROVENANCE.md').write_text('# MERIT functional single-Qwen source memory\n\nGenerated by experiments/20261006_m1_merit/extract.py, R1 sources2026-10-06, Qwen/Qwen3-VL-8B-Instruct. Metadata records actual host/date, readable native/raw paths, original ASR, actualPTS frames, allcurrent inputtokens/user-content pooling offsets, same-model hidden vectors, literalquery/filter/source IDs, common spec and fullnew-video costs. No GT/labels/contentchecksums. Current and remote media are real sources, not generated filter evidence.\n')
    logging.info('EXTRACTION_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
