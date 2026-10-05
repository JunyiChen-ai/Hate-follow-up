"""Actual current-window source pixels, no annotations or scores."""
from pathlib import Path
from geometry import ROOT,SPEC
from src.video_inputs import load_manifest,fixed_windows,window_text
DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/temporal_spatial_search'


def selected_rows(smoke=False):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def windows_for(row,segments,source,folder):
    entries={e['index']:e for e in source['entries']};windows=[]
    for i,((a,b),indices) in enumerate(zip(fixed_windows(float(row['duration']),SPEC['window_seconds']),source['selected_indices'])):
        frames=[]
        for k,index in enumerate(indices):
            e=entries[index];assert a<=e['time']<b
            frames.append(dict(id='p'+str(k),time=e['time'],shape=[e['width'],e['height']],index=index,
                path=str((folder/'frames'/f'frame_{index:08d}.png').relative_to(ROOT))))
        windows.append(dict(i=i,start=a,end=b,body=window_text(segments,a,b),frames=frames))
    return windows
