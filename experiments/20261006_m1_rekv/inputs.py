"""The two fixed test corpora and source-frame inputs, without annotations."""
from retrieval import ROOT, SPEC
from source_frames import validate
from src.video_inputs import load_manifest, fixed_windows, window_text

DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/temporal_retrieved_kv'


def selected_rows(smoke=False):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:
        rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def source_frames(row,meta):
    entries={entry['index']:entry for entry in meta['entries']}
    folder=CACHE/row['dataset']/row['video_id']/'frames'
    return [dict(**entries[index],path=str((folder/f'frame_{index:08d}.png').relative_to(ROOT))) for index in meta['selected_indices']]


def windows_for(row,segments):
    return [dict(i=i,start=a,end=b,body=window_text(segments,a,b))
        for i,(a,b) in enumerate(fixed_windows(float(row['duration']),SPEC['window_seconds']))]


def validate_input(meta,row):
    assert meta['version']==SPEC['version'] and meta['spec']==SPEC and meta['GT_read'] is False
    assert (meta['dataset'],meta['video_id'])==(row['dataset'],row['video_id'])
    validate(meta['source'],row,CACHE/row['dataset']/row['video_id']/'frames')
