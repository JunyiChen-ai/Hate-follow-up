#!/usr/bin/env python3
"""R3 control acquisition, isolated from main caches and all predictions/labels."""
import argparse
import copy
import json
import logging
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.mllm_judge import Judge,MODEL
from src.video_inputs import fixed_windows
from extract import CACHE,selected_rows,validate_cached,tick
from tree import CACHE_VERSION,CONSTANTS,CAP_SYSTEM,CAP_QUESTION,Witness,caption,relevance,record
from control_inputs import ARMS,temporal_tree,fresh_priority,arm_inputs

CONTROL_VERSION='R3 exact-topology temporal witnesses and fresh root priorities; sources2026-10-05'
CONTROL_CACHE=ROOT/'data/semantic_tree_controls/r3'


def validate_control(folder,row,main,features,tokenizer=None):
    """Replay all actual input transforms and source binding, without scoring."""
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text())
    assert meta['control_version']==CONTROL_VERSION and meta['main_cache_version']==CACHE_VERSION
    assert meta['constants']==CONSTANTS and meta['model']==MODEL and meta['GT_read'] is False
    assert meta['dataset']==row['dataset'] and meta['video_id']==row['video_id']
    assert meta['duration']==float(row['duration']) and meta['entries']==main['entries']
    assert meta['main_folder']==str((CACHE/row['dataset']/row['video_id']).relative_to(ROOT))
    caps=meta['temporal']['captions'];events=[]
    def saved(i):
        events.append(i);c=caps[str(i)]
        assert c['path']==str((folder/'witnesses'/f'frame_{main["entries"][i]["index"]:08d}.png').relative_to(ROOT))
        assert c['system']==CAP_SYSTEM and c['question']==CAP_QUESTION
        assert len(c['tokens'])<=96 and all(type(v) is int and v>=0 for v in c['tokens'])
        assert tokenizer.decode(c['tokens'],skip_special_tokens=True).strip()==c['text']
        from PIL import Image
        e=main['entries'][i]
        with Image.open(folder/'witnesses'/f'frame_{e["index"]:08d}.png') as im:assert im.size==(e['width'],e['height'])
        return copy.deepcopy(c)
    rebuilt=temporal_tree(main['tree'],features,main['entries'],saved)
    assert rebuilt==meta['temporal'] and events==meta['temporal']['caption_events']
    priority=meta['fresh_priority'];roots=[n for n in rebuilt['nodes'] if n['parent'] is None]
    assert priority['root_observations']==''.join(record(n) for n in roots)
    scores=priority['scores'];logprobs=priority['logprobs']
    assert np.asarray(logprobs).shape==(len(roots),3) and np.isfinite(logprobs).all()
    assert scores==[int(np.argmax(v))+1 for v in logprobs]
    assert fresh_priority(rebuilt,scores,logprobs)==meta['fresh']
    for arm in ARMS:
        expected=arm_inputs(arm,main['tree'],rebuilt,meta['fresh'],features,main['entries'],fixed_windows(meta['duration'],8))
        assert meta['arms'][arm]==expected
        for p in expected['packets']:
            for i in p['pool_members']:
                from PIL import Image
                e=main['entries'][i]
                with Image.open(folder/'witnesses'/f'frame_{e["index"]:08d}.png') as im:
                    assert im.size==(e['width'],e['height'])
    costs=meta['cost'];assert costs['new_caption_count']==len(events)
    assert costs['new_caption_tokens']==sum(len(caps[str(i)]['tokens']) for i in events)
    assert costs['actual_forwards']['vision']==len(events)
    assert costs['actual_forwards']['language']==len(events)+costs['new_caption_tokens']+1+len(roots)
    assert costs['fresh_priority_forwards']==1+len(roots)
    assert set(costs['input_extra_seconds_by_arm'])==set(ARMS)
    main_frames={p.name for p in (CACHE/row['dataset']/row['video_id']/'witnesses').glob('frame_*.png')}
    new_caps={main['entries'][i]['index'] for i in events}
    for arm in ARMS:
        required={main['entries'][i]['index'] for p in meta['arms'][arm]['packets'] for i in p['pool_members']}
        new_required={i for i in required if f'frame_{i:08d}.png' not in main_frames}
        expected=sum(costs['witness_decode_seconds'][str(i)] for i in new_required-(new_caps if arm.startswith('temporal') else set()))
        if arm.startswith('temporal'):expected+=costs['tree_seconds']
        if arm=='temporal_fresh_priority':expected+=costs['priority_seconds']
        assert costs['input_extra_seconds_by_arm'][arm]==expected
    assert all(np.isfinite(costs[k]) and costs[k]>=0 for k in ('acquisition_seconds','tree_seconds','priority_seconds','witness_seconds','peak_GiB'))
    return meta


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261004_m1_tree'/('r3_control_extract_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    config=dict(control_version=CONTROL_VERSION,main_cache_version=CACHE_VERSION,constants=CONSTANTS,
        model=MODEL,host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),GT_read=False,smoke=a.smoke,
        code='experiments/20261004_m1_tree/{tree,extract,control_inputs,control_extract}.py; sources2026-10-05',
        command='python -u '+' '.join(sys.argv))
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    torch.manual_seed(0);j=Judge(MODEL);counts=dict(language=0,vision=0)
    def language_hook(*_):counts['language']+=1
    def vision_hook(*_):counts['vision']+=1
    hooks=[j.model.model.register_forward_pre_hook(language_hook),j.model.model.visual.register_forward_pre_hook(vision_hook)]
    rows=selected_rows(a.smoke)
    for number,row in enumerate(rows,1):
        main_folder=CACHE/row['dataset']/row['video_id'];main,features=validate_cached(main_folder,row)
        folder=CONTROL_CACHE/row['dataset']/row['video_id'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'metadata.json').exists():
            validate_control(folder,row,main,features,j.tok);logging.info('%d/%d reuse %s/%s',number,len(rows),row['dataset'],row['video_id']);continue
        first=dict(counts);torch.cuda.reset_peak_memory_stats();start=tick()
        target=folder/'witnesses';target.mkdir(parents=True,exist_ok=True)
        for p in sorted((main_folder/'witnesses').glob('frame_*.png')):
            link=target/p.name
            if not link.exists():link.symlink_to(os.path.relpath(p,link.parent))
        witness=Witness(main,target);decode_seconds={}
        original_image=witness.image
        main_frames={p.name for p in (main_folder/'witnesses').glob('frame_*.png')}
        def image(entry):
            key=str(entry['index']);t=tick();result=original_image(entry)
            if f'frame_{entry["index"]:08d}.png' not in main_frames and key not in decode_seconds:decode_seconds[key]=tick()-t
            return result
        witness.image=image
        try:
            def describe(i):
                c=caption(j,witness,main['entries'][i]);c['path']=str(Path(c['path']).relative_to(ROOT));return c
            t=tick();temporal=temporal_tree(main['tree'],features,main['entries'],describe);tree_seconds=tick()-t
            roots=copy.deepcopy([n for n in temporal['nodes'] if n['parent'] is None]);root_observations=''.join(record(n) for n in roots)
            t=tick();scores=relevance(j,roots);priority_seconds=tick()-t
            logprobs=[n['relevance_logprobs'] for n in roots];fresh=fresh_priority(temporal,scores,logprobs)
            t=tick();arms={arm:arm_inputs(arm,main['tree'],temporal,fresh,features,main['entries'],fixed_windows(float(row['duration']),8)) for arm in ARMS}
            for inputs in arms.values():
                for p in inputs['packets']:
                    for i in p['pool_members']:
                        image,path=witness.image(main['entries'][i]);image.close()
            witness_seconds=tick()-t
        finally:witness.close()
        costs=dict(acquisition_seconds=tick()-start,tree_seconds=tree_seconds,priority_seconds=priority_seconds,
            witness_seconds=witness_seconds,peak_GiB=torch.cuda.max_memory_allocated()/2**30,
            actual_forwards={k:counts[k]-first[k] for k in counts},fresh_priority_forwards=1+len(roots),
            new_caption_count=len(temporal['caption_events']),
            new_caption_tokens=sum(len(temporal['captions'][str(i)]['tokens']) for i in temporal['caption_events']))
        costs['witness_decode_seconds']=decode_seconds;costs['input_extra_seconds_by_arm']={}
        new_caps={main['entries'][i]['index'] for i in temporal['caption_events']}
        for arm in ARMS:
            required={main['entries'][i]['index'] for p in arms[arm]['packets'] for i in p['pool_members']}
            new_required={i for i in required if f'frame_{i:08d}.png' not in main_frames}
            seconds=sum(decode_seconds[str(i)] for i in new_required-(new_caps if arm.startswith('temporal') else set()))
            if arm.startswith('temporal'):seconds+=tree_seconds
            if arm=='temporal_fresh_priority':seconds+=priority_seconds
            costs['input_extra_seconds_by_arm'][arm]=seconds
        meta=dict(config,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),
            main_folder=str(main_folder.relative_to(ROOT)),entries=main['entries'],temporal=temporal,fresh=fresh,
            fresh_priority=dict(root_observations=root_observations,scores=scores,logprobs=logprobs),arms=arms,cost=costs)
        partial=folder/'metadata.partial';partial.write_text(json.dumps(meta)+'\n');partial.replace(folder/'metadata.json')
        validate_control(folder,row,main,features,j.tok)
        logging.info('%d/%d %s/%s newcaps=%d %.2fs',number,len(rows),row['dataset'],row['video_id'],costs['new_caption_count'],costs['acquisition_seconds'])
    for h in hooks:h.remove()
    metas=[validate_control(CONTROL_CACHE/r['dataset']/r['video_id'],r,*validate_cached(CACHE/r['dataset']/r['video_id'],r),tokenizer=j.tok) for r in rows]
    (out/'summary.json').write_text(json.dumps(dict(coverage=len(metas),GT_read=False,datasets={
        ds:{k:sum(m['cost'][k] for m in metas if m['dataset']==ds) for k in ('acquisition_seconds','new_caption_count','new_caption_tokens','fresh_priority_forwards')}
        for ds in ('HateMM','HateClipSeg')}),indent=2)+'\n')
    (CONTROL_CACHE/'PROVENANCE.md').write_text('# R3 semantic tree control inputs\n\n'
        'Source: experiments/20261004_m1_tree/{tree,control_inputs,control_extract}.py; sources2026-10-05.\n'
        'Frozen Qwen/Qwen3-VL-8B-Instruct. Main source features/actual PTS: data/semantic_cluster_tree/.\n'
        'Main cache stays read-only; exact source-frame captions reused, additional temporal caption tokens and fresh root relevance saved in metadata.\n'
        'Witness symlinks point to the exact original cache frame; additional pixels decoded from original indexed source PTS.\n'
        'Generation command '+config['command']+'. Hosts '+', '.join(sorted({m['host'] for m in metas}))+'; date '+config['date']+'.\n')
    logging.info('CONTROL_EXTRACTION_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
