"""Current observed windows and native inputs, without labels or scores."""
from retrieval import ROOT,SPEC
from src.video_inputs import load_manifest,fixed_windows,window_text
DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/temporal_multikey_memory'


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


def caption_content(window):
    items=[dict(type='text',text=f"Current observed window [{window['start']},{window['end']}); literal ASR:\n{window['body']}\nExtract neutral caption and event/dialogue/object/summary retrieval keys. UNKNOWN for unobserved or unreadable facts.")]
    paths=[]
    for frame in window['frames']:
        items.extend([dict(type='text',text=f"Actual frame {frame['id']} at PTS time {frame['time']:.9f}s."),dict(type='image',image=frame['path'])]);paths.append(frame['path'])
    return items,paths


def filter_content(current,caption,query,windows,records,candidates):
    import json
    context=dict(current_window=current['i'],current_bounds=[current['start'],current['end']],current_ASR=current['body'],
        current_caption=caption,query=query,candidates=[dict(id=i,bounds=[windows[i]['start'],windows[i]['end']],caption=records[i]['caption'],ASR=windows[i]['body']) for i in candidates])
    return [dict(type='text',text=json.dumps(context,ensure_ascii=False)+'\nSelect only supplied candidate IDs. Return bounded status/IDs/new_query; no generated evidence text. Remote observations remain at their own times.')]
