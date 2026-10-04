#!/usr/bin/env python3
"""Acquire a whole video's complete cluster tree with one frozen Qwen."""
import argparse
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
from src.video_inputs import load_manifest,fixed_windows
from tree import CACHE_VERSION,CONSTANTS,decode_pool,Witness,Builder,caption,relevance,window_packet
DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/semantic_cluster_tree'


def selected_rows(smoke):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:
        rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def validate_cached(folder,row):
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text())
    assert meta['cache_version']==CACHE_VERSION and meta['constants']==CONSTANTS and meta['model']==MODEL
    assert meta['dataset']==row['dataset'] and meta['video_id']==row['video_id']
    assert meta['duration']==float(row['duration']) and meta['manifest_video_path']==row['video_path']
    assert Path(meta['input_video']).stem==row['video_id'] and meta['GT_read'] is False
    features=np.load(folder/'features.npy');entries=meta['entries']
    assert features.shape==(len(entries),meta['feature_dimension']) and features.dtype==np.float32
    assert np.isfinite(features).all() and (np.linalg.norm(features,axis=1)<=1.00001).all()
    assert len({e['index'] for e in entries})==len(entries)
    assert all(0<=e['time']<meta['duration'] for e in entries)
    assert all((a['time'],a['index'])<(b['time'],b['index']) for a,b in zip(entries,entries[1:]))
    nodes=meta['tree']['nodes'];ids={n['id']:n for n in nodes};assert len(ids)==len(nodes)<=112
    roots=[n for n in nodes if n['parent'] is None]
    assert sorted(i for n in roots for i in n['members'])==list(range(len(entries)))
    for n in nodes:
        assert len(n['members'])>0 and len(set(n['members']))==len(n['members'])
        assert n['representative'] in n['members'] and n['time']==entries[n['representative']]['time']
        assert n['caption']==meta['tree']['captions'][str(n['representative'])]['text']
        center=np.asarray(n['center'],dtype=np.float32)
        np.testing.assert_array_equal(center,features[n['members']].mean(0,dtype=np.float32))
        if n['parent'] is not None:
            p=ids[n['parent']];assert set(n['members'])<=set(p['members'])
            assert n['depth']==p['depth']+1 and n['root_relevance']==p['root_relevance']
        else:assert n['depth']==0
        children=[c for c in nodes if c['parent']==n['id']]
        if children:assert sorted(i for c in children for i in c['members'])==sorted(n['members'])
    for packet,(a,b) in zip(meta['packets'],fixed_windows(meta['duration'],8)):
        expected=window_packet(meta['tree'],features,entries,a,b)
        assert packet==expected and all(a<=entries[i]['time']<b for i in packet['pool_members'])
        for i in packet['pool_members']:
            assert (folder/'witnesses'/f'frame_{entries[i]["index"]:08d}.png').is_file()
    assert len(meta['packets'])==len(fixed_windows(meta['duration'],8))
    return meta,features


def tick():torch.cuda.synchronize();return time.perf_counter()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261004_m1_tree'/('r1_extract_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    config=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,cache_version=CACHE_VERSION,
        constants=CONSTANTS,GT_read=False,smoke=a.smoke,torch=torch.__version__,transformers=transformers.__version__,
        command='python -u '+' '.join(sys.argv),
        code='experiments/20261004_m1_tree/{tree,extract}.py + src/{video_inputs,mllm_judge}.py; sources2026-10-04')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');torch.manual_seed(0)
    rows=selected_rows(a.smoke);j=Judge(MODEL);counts=dict(language=0,vision=0)
    def lm_hook(*_):counts['language']+=1
    def v_hook(*_):counts['vision']+=1
    hooks=[j.model.model.register_forward_pre_hook(lm_hook),j.model.model.visual.register_forward_pre_hook(v_hook)]
    for number,row in enumerate(rows,1):
        folder=CACHE/row['dataset']/row['video_id'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'metadata.json').exists():validate_cached(folder,row);logging.info('%d/%d reuse %s/%s',number,len(rows),row['dataset'],row['video_id']);continue
        first=dict(counts);torch.cuda.reset_peak_memory_stats();t0=tick()
        features,source=decode_pool(j,row);feature_seconds=tick()-t0
        witness=Witness(source,folder/'witnesses');t1=tick()
        builder=Builder(features,source['entries'],lambda i:caption(j,witness,source['entries'][i]),lambda nodes:relevance(j,nodes))
        tree=builder.build();tree_seconds=tick()-t1;t2=tick();packets=[]
        for start,end in fixed_windows(float(row['duration']),8):
            packet=window_packet(tree,features,source['entries'],start,end);packets.append(packet)
            for i in packet['pool_members']:
                image,path=witness.image(source['entries'][i]);image.close()
        witness.close();witness_seconds=tick()-t2
        meta=dict(config,**source,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),
            feature_dimension=features.shape[1],tree=tree,packets=packets,
            feature_seconds=feature_seconds,tree_seconds=tree_seconds,witness_seconds=witness_seconds,
            standalone_seconds=tick()-t0,peak_GiB=torch.cuda.max_memory_allocated()/2**30,
            actual_forwards={k:counts[k]-first[k] for k in counts},caption_count=len(tree['captions']),
            caption_tokens=sum(len(v['tokens']) for v in tree['captions'].values()))
        np.save(folder/'features.npy',features)
        temporary=folder/'metadata.partial';temporary.write_text(json.dumps(meta)+'\n');temporary.replace(folder/'metadata.json')
        validate_cached(folder,row)
        logging.info('%d/%d %s/%s pool=%d nodes=%d captions=%d %.2fs',number,len(rows),row['dataset'],row['video_id'],len(source['entries']),len(tree['nodes']),len(tree['captions']),meta['standalone_seconds'])
    for h in hooks:h.remove()
    metas=[validate_cached(CACHE/r['dataset']/r['video_id'],r)[0] for r in rows]
    summary=dict(coverage=len(metas),GT_read=False,datasets={})
    for ds in DATASETS:
        rr=[r for r in metas if r['dataset']==ds]
        summary['datasets'][ds]={k:sum(r[k] for r in rr) for k in ('feature_seconds','tree_seconds','witness_seconds','standalone_seconds','caption_count','caption_tokens','decoded_frames')}
        summary['datasets'][ds]['peak_GiB']=max(r['peak_GiB'] for r in rr)
        summary['datasets'][ds]['actual_forwards']={k:sum(r['actual_forwards'][k] for r in rr) for k in counts}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (CACHE/'PROVENANCE.md').write_text('# Semantic cluster tree provenance\n\n'
        'Source code experiments/20261004_m1_tree/{tree,extract}.py, sources2026-10-04.\n'
        'Frozen Qwen/Qwen3-VL-8B-Instruct; actual source video paths/PTS and generation hosts/dates in metadata.json.\n'
        'Input identity/duration/path: data/omsl_v6_inputs/manifests/all_test.jsonl, no GT.\n'
        'Features are FP32 mean final merged Qwen image embeddings, normed; witnessed frames are actual source pixels.\n'
        'Complete captions, rejected breadth rounds, memberships/parents/wording/costs in metadata.json.\n'
        'Command: '+config['command']+'; called by launch/run_lab.sh in an allocated lab Slurm job.\n'
        'Generating hosts: '+', '.join(sorted({r['host'] for r in metas}))+'; date '+config['date']+'.\n')
    logging.info('EXTRACTION_DONE coverage=%d',len(metas))


if __name__=='__main__':main()
