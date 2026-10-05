"""Actual caption → relevance → continuous event → actual background acquisition."""
import argparse
import json
import logging
import socket
import time
import numpy as np
import torch
from events import candidates,select
from inputs import ROOT,SPEC,CACHE,DATASETS,selected_rows,windows_for,caption_content,relevance_content,background_content
from interface import description_writer,compile_description,relevance_writer,compile_relevance,unavailable
from src.actual_video_frames import acquire_window_frames,validate_frames
from src.structured_source_generation import generate,validate_generation
from src.mllm_judge import Judge,MODEL
from src.video_inputs import load_asr


def acquire(j,row,segments,folder):
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();source=acquire_window_frames(row,folder/'frames',SPEC['window_seconds']);windows=windows_for(row,segments,source,folder)
    captions=[];records=[]
    for window in windows:
        items,paths=caption_content(window);g=generate(j,ROOT,SPEC['caption_system'],items,paths,SPEC['caption_generation_tokens'],description_writer('caption'));captions.append(g);records.append(compile_description(g,'caption'))
    relevance=[];packets=[];backgrounds={}
    for i,window in enumerate(windows):
        ids=candidates(i,len(windows));available=[bool(windows[k]['frames']) and not unavailable(records[k]) for k in ids]
        request_available=bool(window['body'].strip()) or not unavailable(records[i])
        if request_available and any(available):
            items=relevance_content(window,records[i],ids,windows,records);g=generate(j,ROOT,SPEC['relevance_system'],items,[],SPEC['relevance_generation_tokens'],relevance_writer(available));values=compile_relevance(g,len(ids))
        else:g=None;values=[None]*len(ids)
        relevance.append(g);packet=select(ids,values,i);rep=packet['representative']
        if rep is not None and not windows[rep]['frames']:raise AssertionError('representative has no actual frame')
        if rep is not None and str(rep) not in backgrounds:
            items,paths=background_content(windows[rep]);bg=generate(j,ROOT,SPEC['background_system'],items,paths,SPEC['background_generation_tokens'],description_writer('background'))
            backgrounds[str(rep)]=dict(generation=bg,description=compile_description(bg,'background'),frame=windows[rep]['frames'][0])
        packets.append(packet)
    gens=captions+[g for g in relevance if g is not None]+[b['generation'] for b in backgrounds.values()]
    return dict(version=SPEC['version'],spec=SPEC,model=MODEL,host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),dataset=row['dataset'],video_id=row['video_id'],GT_read=False,segments=[list(s) for s in segments],source=source,windows=windows,captions=captions,records=records,relevance=relevance,packets=packets,backgrounds=backgrounds,peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        cost=dict(standalone_seconds=time.perf_counter()-start,generation_seconds=sum(g['seconds'] for g in gens),caption_calls=len(captions),relevance_calls=sum(g is not None for g in relevance),background_calls=len(backgrounds),actual_forwards=sum(g['actual_forwards'] for g in gens),actual_vision_forwards=sum(g['actual_vision_forwards'] for g in gens)))


def validate(j,m,row,segments,folder):
    assert m['version']==SPEC['version'] and m['spec']==SPEC and m['model']==MODEL and m['GT_read'] is False
    assert (m['dataset'],m['video_id'],m['segments'])==(row['dataset'],row['video_id'],[list(s) for s in segments])
    validate_frames(m['source'],row,folder/'frames');windows=windows_for(row,segments,m['source'],folder);assert windows==m['windows']
    assert len(windows)==len(m['captions'])==len(m['records'])==len(m['relevance'])==len(m['packets'])
    for window,g,r in zip(windows,m['captions'],m['records']):
        items,paths=caption_content(window);validate_generation(j,ROOT,g,SPEC['caption_system'],items,paths,SPEC['caption_generation_tokens'],description_writer('caption'));assert r==compile_description(g,'caption')
    representatives=set()
    for i,(window,g,packet) in enumerate(zip(windows,m['relevance'],m['packets'])):
        ids=candidates(i,len(windows));available=[bool(windows[k]['frames']) and not unavailable(m['records'][k]) for k in ids];request_available=bool(window['body'].strip()) or not unavailable(m['records'][i])
        if request_available and any(available):
            assert g is not None;items=relevance_content(window,m['records'][i],ids,windows,m['records']);validate_generation(j,ROOT,g,SPEC['relevance_system'],items,[],SPEC['relevance_generation_tokens'],relevance_writer(available));values=compile_relevance(g,len(ids))
        else:assert g is None;values=[None]*len(ids)
        assert all(v is None or available[k] for k,v in enumerate(values));assert packet==select(ids,values,i)
        if packet['representative'] is not None:representatives.add(str(packet['representative']))
    assert representatives==m['backgrounds'].keys()
    for key,bg in m['backgrounds'].items():
        window=windows[int(key)];assert bg['frame']==window['frames'][0];items,paths=background_content(window);g=bg['generation'];validate_generation(j,ROOT,g,SPEC['background_system'],items,paths,SPEC['background_generation_tokens'],description_writer('background'));assert bg['description']==compile_description(g,'background')
    gens=m['captions']+[g for g in m['relevance'] if g is not None]+[b['generation'] for b in m['backgrounds'].values()];cost=m['cost']
    assert cost['caption_calls']==len(windows) and cost['relevance_calls']==sum(g is not None for g in m['relevance']) and cost['background_calls']==len(representatives)
    assert cost['actual_forwards']==sum(g['actual_forwards'] for g in gens) and cost['actual_vision_forwards']==sum(g['actual_vision_forwards'] for g in gens)
    assert abs(cost['generation_seconds']-sum(g['seconds'] for g in gens))<1e-6 and all(np.isfinite(v) and v>=0 for v in cost.values())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();out=ROOT/'runs/20261006_m1_videoevent'/('source_smoke' if a.smoke else 'source_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()]);logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    (out/'config.json').write_text(json.dumps(dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,spec=SPEC,GT_read=False,smoke=a.smoke,torch=torch.__version__,transformers=transformers.__version__,code='experiments/20261006_m1_videoevent/{extract,events,interface,inputs}.py;2026-10-06',command='python -u '+' '.join(__import__('sys').argv)),indent=2)+'\n')
    torch.manual_seed(0);torch.set_num_threads(4);j=Judge(MODEL);j.forward_calls=j.vision_calls=0;hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    asr={ds:load_asr(ds) for ds in DATASETS};rows=selected_rows(a.smoke)
    for n,row in enumerate(rows,1):
        segments=asr[row['dataset']].get(row['video_id'],[]);folder=CACHE/row['dataset']/row['video_id'];folder.mkdir(parents=True,exist_ok=True);path=folder/'metadata.json'
        if path.exists():m=json.loads(path.read_text())
        else:m=acquire(j,row,segments,folder);validate(j,m,row,segments,folder);tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(m,allow_nan=False)+'\n');tmp.replace(path)
        validate(j,m,row,segments,folder);logging.info('%d/%d %s/%s %.2fs events=%d backgrounds=%d',n,len(rows),row['dataset'],row['video_id'],m['cost']['standalone_seconds'],sum(bool(p['selected_windows']) for p in m['packets']),len(m['backgrounds']))
    for hook in hooks:hook.remove()
    (CACHE/'PROVENANCE.md').write_text('# Query-relevant events/background\n\nGenerated by experiments/20261006_m1_videoevent/extract.py, sources2026-10-06, Qwen/Qwen3-VL-8B-Instruct. Metadata records actualhost/date, original raw/native paths/ASR, actualPTS/pixels, allcurrent source/input/generated tokens, neutral relevance-driven event selection and owned background frame/interpretation, common spec and complete new-video costs. No labels/contentchecksums. Descriptions are unverified interpretations; actual media and provenance preserved.\n');logging.info('EXTRACTION_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
