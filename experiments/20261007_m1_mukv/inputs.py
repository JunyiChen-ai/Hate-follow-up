"""Frozen two-cohort source sampling and full-image spatial quadrant identities."""
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
SPEC=json.loads((Path(__file__).parent/'spec.json').read_text())
DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/temporal_multigrain_kv'
from src.video_inputs import load_manifest,fixed_windows,window_text
from src.fractional_window_frames import validate


def selected_rows(smoke=False):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333);return rows


def frames_for(row):
    folder=CACHE/row['dataset']/row['video_id'];meta=json.loads((folder/'metadata.json').read_text())
    assert meta['spec']==SPEC and meta['GT_read'] is False and (meta['dataset'],meta['video_id'])==(row['dataset'],row['video_id'])
    source=meta['source'];validate(source,row,folder/'frames',SPEC['source_fractions'],SPEC['window_seconds']);entries={e['index']:e for e in source['entries']}
    return source,[[dict(**entries[i],path=str((folder/'frames'/f'frame_{i:08d}.png').relative_to(ROOT))) for i in pp] for pp in source['selected_indices']]


def windows_for(row,segments):
    return [dict(i=i,start=a,end=b,body=window_text(segments,a,b)) for i,(a,b) in enumerate(fixed_windows(float(row['duration']),SPEC['window_seconds']))]


def quadrants(positions,visual_rows):
    """Actual spatial coordinates, never flattened quarter strips or fake grids."""
    assert positions.shape[0]==3 and positions.shape[1]==1 and visual_rows
    h=positions[1,0,visual_rows];w=positions[2,0,visual_rows]
    hh=int(h.max()-h.min())+1;ww=int(w.max()-w.min())+1;hcut=int(h.min())+hh//2;wcut=int(w.min())+ww//2
    masks=[(h<hcut)&(w<wcut),(h<hcut)&(w>=wcut),(h>=hcut)&(w<wcut),(h>=hcut)&(w>=wcut)]
    result=[mask.nonzero().flatten().tolist() for mask in masks]
    assert sorted(i for ids in result for i in ids)==list(range(len(visual_rows)))
    return result
