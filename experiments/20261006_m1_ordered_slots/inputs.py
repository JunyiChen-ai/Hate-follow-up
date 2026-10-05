"""Original native and actual timed source observations, no labels."""
import json
from retrieval import ROOT,SPEC,INTERFACE
from src.video_inputs import load_manifest,fixed_windows,window_text
DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/('data/temporal_ordered_slots' if INTERFACE=='A' else 'data/temporal_ordered_slots_B')


def selected_rows(smoke=False):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def windows_for(row,segments,source,folder):
    entries={e['index']:e for e in source['entries']};result=[]
    for i,((a,b),indices) in enumerate(zip(fixed_windows(float(row['duration']),SPEC['window_seconds']),source['selected_indices'])):
        frames=[]
        for k,index in enumerate(indices):
            e=entries[index];assert a<=e['time']<b
            frames.append(dict(id='p'+str(k),index=index,time=e['time'],shape=[e['width'],e['height']],path=str((folder/'frames'/f'frame_{index:08d}.png').relative_to(ROOT))))
        result.append(dict(i=i,start=a,end=b,body=window_text(segments,a,b),frames=frames))
    return result


def caption_content(window,previous):
    context=dict(current_window=window['i'],current_bounds=[window['start'],window['end']],literal_ASR=window['body'],previous_window_unverified_caption=previous)
    items=[dict(type='text',text=json.dumps(context,ensure_ascii=False)+'\nDescribe only current observable content; the previous caption is unverified context, not current evidence.')];paths=[]
    for frame in window['frames']:
        items.extend([dict(type='text',text=f"Actual current frame {frame['id']} at PTS {frame['time']:.9f}s."),dict(type='image',image=frame['path'])]);paths.append(frame['path'])
    return items,paths


def slots_content(windows,records,indices):
    rows=[dict(id=i,bounds=[windows[i]['start'],windows[i]['end']],caption=records[i]['caption'],literal_ASR=windows[i]['body']) for i in indices]
    return [dict(type='text',text=json.dumps(rows,ensure_ascii=False)+'\nFor each listed ID in order output prequel/current/sequel hypothetical search conditions, NONE or UNKNOWN. These are retrieval conditions and will not be factual evidence.')]
