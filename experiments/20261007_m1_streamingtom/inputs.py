"""Current validated actual0.5fps pixels, original ASR/cohort; no annotations."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.video_inputs import load_manifest,fixed_windows,window_text
from src.uniform_video_inputs import validate

SPEC=json.loads((Path(__file__).parent/'spec.json').read_text())
DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/temporal_retrieved_kv'


def selected_rows(smoke=False):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:
        rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def frames_for(row):
    folder=CACHE/row['dataset']/row['video_id'];meta=json.loads((folder/'metadata.json').read_text())
    assert meta['GT_read'] is False and (meta['dataset'],meta['video_id'])==(row['dataset'],row['video_id'])
    validate(meta['source'],row,folder/'frames',SPEC['source_fps'])
    entries={e['index']:e for e in meta['source']['entries']}
    frames=[dict(**entries[i],path=str((folder/'frames'/f'frame_{i:08d}.png').relative_to(ROOT))) for i in meta['source']['selected_indices']]
    return meta['source'],frames


def windows_for(row,segments):
    return [dict(i=i,start=a,end=b,body=window_text(segments,a,b)) for i,(a,b) in enumerate(fixed_windows(float(row['duration']),SPEC['window_seconds']))]


def local_ids(frames,window):
    return [i for i,f in enumerate(frames) if window['start']<=f['time']<window['end']]
