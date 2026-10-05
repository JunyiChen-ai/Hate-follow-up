"""Real media → sequential captions → hypothetical slots → timed source IDs."""
import argparse
import json
import logging
import socket
import time
import numpy as np
import torch
from inputs import ROOT,SPEC,CACHE,DATASETS,selected_rows,windows_for,caption_content,slots_content
from interface import caption_writer,slot_writer,compile_caption,compile_slots,unavailable
from retrieval import assign
from embedding import embed,validate_embedding
from src.actual_video_frames import acquire_window_frames,validate_frames
from src.structured_source_generation import generate,validate_generation
from src.mllm_judge import Judge,MODEL
from src.video_inputs import load_asr


def caption_arrays(embeddings,dimension):
    values=np.zeros((len(embeddings),dimension),np.float32);available=np.zeros(len(embeddings),bool)
    for i,e in enumerate(embeddings):
        if e is not None and e['available']:values[i]=e['vector'];available[i]=True
    return values,available


def slot_batches(count):
    return [list(range(i,min(i+SPEC['slot_batch_windows'],count))) for i in range(0,count,SPEC['slot_batch_windows'])]


def acquire(j,row,segments,folder):
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    source=acquire_window_frames(row,folder/'frames',SPEC['window_seconds']);windows=windows_for(row,segments,source,folder)
    generations=[];records=[];embeddings=[];previous='UNKNOWN'
    for window in windows:
        items,paths=caption_content(window,previous)
        g=generate(j,ROOT,SPEC['caption_system'],items,paths,SPEC['caption_generation_tokens'],caption_writer)
        r=compile_caption(g);generations.append(g);records.append(r)
        embeddings.append(None if unavailable(r['caption']) else embed(j,r['caption']))
        previous=r['caption']
    slot_generations=[];slots=[];slot_embeddings=[]
    for indices in slot_batches(len(windows)):
        g=generate(j,ROOT,SPEC['slot_system'],slots_content(windows,records,indices),[],SPEC['slot_generation_tokens'],slot_writer(indices))
        slot_generations.append(g);slots.extend(compile_slots(g,indices))
    for row_slots in slots:
        slot_embeddings.append({role:None if unavailable(row_slots[role]) else embed(j,row_slots[role]) for role in SPEC['slot_order']})
    dimension=j.model.config.text_config.hidden_size;values,available=caption_arrays(embeddings,dimension);packets=[]
    for i,ee in enumerate(slot_embeddings):
        queries={role:e['vector'] if e is not None and e['available'] else None for role,e in ee.items()}
        packets.append(dict(window=i,**assign(values,available,queries,i)))
    gens=generations+slot_generations;ee=[e for e in embeddings if e is not None]+[e for rr in slot_embeddings for e in rr.values() if e is not None]
    return dict(version=SPEC['version'],spec=SPEC,model=MODEL,host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),
        dataset=row['dataset'],video_id=row['video_id'],GT_read=False,segments=[list(s) for s in segments],source=source,windows=windows,
        generations=generations,records=records,caption_embeddings=embeddings,dimension=dimension,slot_generations=slot_generations,slots=slots,slot_embeddings=slot_embeddings,packets=packets,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        cost=dict(standalone_seconds=time.perf_counter()-start,generation_seconds=sum(g['seconds'] for g in gens),embedding_seconds=sum(e['seconds'] for e in ee),
            caption_calls=len(generations),slot_calls=len(slot_generations),embedding_calls=len(ee),actual_forwards=sum(g['actual_forwards'] for g in gens)+len(ee),
            actual_vision_forwards=sum(g['actual_vision_forwards'] for g in gens)))


def validate(j,m,row,segments,folder):
    assert m['version']==SPEC['version'] and m['spec']==SPEC and m['model']==MODEL and m['GT_read'] is False
    assert (m['dataset'],m['video_id'],m['segments'])==(row['dataset'],row['video_id'],[list(s) for s in segments])
    validate_frames(m['source'],row,folder/'frames');windows=windows_for(row,segments,m['source'],folder);assert windows==m['windows']
    assert len(windows)==len(m['generations'])==len(m['records'])==len(m['caption_embeddings'])==len(m['slots'])==len(m['slot_embeddings'])==len(m['packets'])
    previous='UNKNOWN'
    for window,g,r,e in zip(windows,m['generations'],m['records'],m['caption_embeddings']):
        items,paths=caption_content(window,previous)
        validate_generation(j,ROOT,g,SPEC['caption_system'],items,paths,SPEC['caption_generation_tokens'],caption_writer)
        assert r==compile_caption(g)
        if unavailable(r['caption']):assert e is None
        else:assert e is not None;validate_embedding(j,e,r['caption'])
        previous=r['caption']
    batches=slot_batches(len(windows));assert len(batches)==len(m['slot_generations']);slots=[]
    for indices,g in zip(batches,m['slot_generations']):
        validate_generation(j,ROOT,g,SPEC['slot_system'],slots_content(windows,m['records'],indices),[],SPEC['slot_generation_tokens'],slot_writer(indices))
        slots.extend(compile_slots(g,indices))
    assert slots==m['slots'];values,available=caption_arrays(m['caption_embeddings'],m['dimension'])
    for i,(rr,ee,packet) in enumerate(zip(slots,m['slot_embeddings'],m['packets'])):
        assert rr['id']==i and packet['window']==i
        for role in SPEC['slot_order']:
            if unavailable(rr[role]):assert ee[role] is None
            else:assert ee[role] is not None;validate_embedding(j,ee[role],rr[role])
        queries={role:e['vector'] if e is not None and e['available'] else None for role,e in ee.items()}
        assert packet==dict(window=i,**assign(values,available,queries,i))
        assert all(k!=i for k in packet['source_ids']) and len(packet['source_ids'])<=SPEC['max_remote_windows']
    gens=m['generations']+m['slot_generations'];ee=[e for e in m['caption_embeddings'] if e is not None]+[e for rr in m['slot_embeddings'] for e in rr.values() if e is not None]
    cost=m['cost'];assert cost['actual_forwards']==sum(g['actual_forwards'] for g in gens)+len(ee)
    assert cost['actual_vision_forwards']==sum(g['actual_vision_forwards'] for g in gens)
    assert cost['caption_calls']==len(windows) and cost['slot_calls']==len(batches) and cost['embedding_calls']==len(ee)
    assert all(np.isfinite(v) and v>=0 for v in cost.values())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261006_m1_ordered_slots'/('source_smoke' if a.smoke else 'source_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,spec=SPEC,GT_read=False,smoke=a.smoke,torch=torch.__version__,transformers=transformers.__version__,code='experiments/20261006_m1_ordered_slots/{extract,embedding,retrieval,interface,inputs}.py;sources2026-10-06',command='python -u '+' '.join(__import__('sys').argv))
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
    (CACHE/'PROVENANCE.md').write_text('# Ordered conditional actual-source slots\n\nGenerated by experiments/20261006_m1_ordered_slots/extract.py, R1 sources2026-10-06, Qwen/Qwen3-VL-8B-Instruct. Metadata records actualhost/date/readable native/raw paths, originalASR, actualPTS/pixels, sequentialcaptions and hypotheticalslots, allcurrent tokens and exactuser pooling offsets, same-model vectors, ordered real sourceIDs or NONE, common spec and fullnew-video costs. No GT/labels/contentchecksums. Captions/slots/scores are selection conditions and excluded from final reader evidence.\n')
    logging.info('EXTRACTION_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
