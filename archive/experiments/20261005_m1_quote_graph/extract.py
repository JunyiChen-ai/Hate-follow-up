#!/usr/bin/env python3
"""Generate and execute exact-source quotation graphs with one frozen Qwen."""
import argparse
import json
import logging
from pathlib import Path
import socket
import sys
import time
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.mllm_judge import Judge,MODEL
from src.mllm_generate import generate_text
from src.video_inputs import load_manifest,load_asr,fixed_windows,window_text
from graph import VERSION,CONSTANTS,SYSTEM,chunk_indices,previous_subset,prompt,parse_chunk,merge,packet
DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/temporal_quotation_graph'


def selected_rows(smoke):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:
        rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def source_windows(row,segments):
    return [dict(i=i,start=a,end=b,body=window_text(segments,a,b))
            for i,(a,b) in enumerate(fixed_windows(float(row['duration']),8))]


def validate(metadata,row,segments,renderer):
    assert metadata['version']==VERSION and metadata['constants']==CONSTANTS and metadata['model']==MODEL
    assert metadata['GT_read'] is False and metadata['dataset']==row['dataset'] and metadata['video_id']==row['video_id']
    assert metadata['duration']==float(row['duration']) and metadata['manifest_video_path']==row['video_path']
    assert metadata['segments']==[list(s) for s in segments]
    windows=source_windows(row,segments); assert windows==metadata['windows']
    chunks=chunk_indices(len(windows)); assert len(chunks)==len(metadata['chunks'])
    graph=dict(nodes=[],edges=[]);forwards=0;seconds=0.
    for indices,record in zip(chunks,metadata['chunks']):
        previous=previous_subset(graph,indices);text=prompt(windows,indices,previous);g=record['generation']
        assert record['indices']==indices and record['previous']==previous
        expected=renderer.render([renderer.turn('system',SYSTEM),renderer.turn('user',text)],True)
        assert g['prompt']==expected and g['input_tokens']==renderer.tok.encode(expected,add_special_tokens=False)
        assert g['max_tokens']==CONSTANTS['max_generation_tokens'] and len(g['tokens'])<=g['max_tokens']
        assert renderer.tok.decode(g['tokens'],skip_special_tokens=True).strip()==g['text']
        assert g['actual_forwards']==1+len(g['tokens'])
        parsed,error=parse_chunk(g['text'],g['truncated'],windows,indices,previous)
        assert record['parsed']==parsed and record['error']==error
        graph=merge(graph,parsed);forwards+=g['actual_forwards'];seconds+=g['seconds']
    assert metadata['graph']==graph and metadata['packets']==[packet(graph,windows,i) for i in range(len(windows))]
    assert metadata['actual_forwards']==forwards and metadata['standalone_seconds']>=seconds


@torch.no_grad()
def acquire(j,row,segments):
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();first=j.forward_calls
    windows=source_windows(row,segments);graph=dict(nodes=[],edges=[]);chunks=[]
    for indices in chunk_indices(len(windows)):
        previous=previous_subset(graph,indices)
        generation=generate_text(j,SYSTEM,prompt(windows,indices,previous),CONSTANTS['max_generation_tokens'])
        parsed,error=parse_chunk(generation['text'],generation['truncated'],windows,indices,previous)
        chunks.append(dict(indices=indices,previous=previous,generation=generation,parsed=parsed,error=error))
        graph=merge(graph,parsed)
    packets=[packet(graph,windows,i) for i in range(len(windows))];torch.cuda.synchronize()
    return dict(version=VERSION,constants=CONSTANTS,model=MODEL,dataset=row['dataset'],video_id=row['video_id'],
        duration=float(row['duration']),manifest_video_path=row['video_path'],GT_read=False,host=socket.gethostname(),
        date=time.strftime('%Y-%m-%d'),segments=[list(s) for s in segments],windows=windows,chunks=chunks,
        graph=graph,packets=packets,standalone_seconds=time.perf_counter()-start,actual_forwards=j.forward_calls-first,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    out=ROOT/'runs/20261005_m1_quote_graph'/('r1_extract_'+('smoke' if args.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    config=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),GT_read=False,smoke=args.smoke,model=MODEL,
        version=VERSION,constants=CONSTANTS,torch=torch.__version__,transformers=transformers.__version__,
        code='experiments/20261005_m1_quote_graph/{graph,extract}.py + src/mllm_generate.py; sources2026-10-05',
        command='python -u '+' '.join(sys.argv))
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');torch.manual_seed(0)
    j=Judge(MODEL);j.forward_calls=0
    hook=j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1))
    asr={ds:load_asr(ds) for ds in DATASETS};rows=selected_rows(args.smoke)
    for i,row in enumerate(rows,1):
        path=CACHE/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        segments=asr[row['dataset']].get(row['video_id'],[])
        if path.exists():
            validate(json.loads(path.read_text()),row,segments,j);logging.info('%d/%d reuse %s/%s',i,len(rows),row['dataset'],row['video_id']);continue
        metadata=acquire(j,row,segments);validate(metadata,row,segments,j)
        temp=path.with_suffix('.partial');temp.write_text(json.dumps(metadata)+'\n');temp.replace(path)
        logging.info('%d/%d %s/%s %.2fs graphnodes=%d validchunks=%d',i,len(rows),row['dataset'],row['video_id'],
            metadata['standalone_seconds'],len(metadata['graph']['nodes']),sum(c['error'] is None for c in metadata['chunks']))
    hook.remove()
    (CACHE/'PROVENANCE.md').write_text('# Source-bound quotation graph\n\n'
        'Sources2026-10-05: experiments/20261005_m1_quote_graph/{graph,extract}.py and src/mllm_generate.py.\n'
        'Same frozen Qwen/Qwen3-VL-8B-Instruct; native timestamped ASR through shared window_text.\n'
        'Literal source windows/character coordinates, actual prompts/generated tokens, host/date/cost in each JSON.\n'
        'No gold entity/quote/speaker list or hate labels/GT. Source bindings validate existence, not semantic truth.\n'
        'Host '+config['host']+', date '+config['date']+'; command '+config['command']+'.\n')
    logging.info('EXTRACTION_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
