"""Fixed333/fixed5 unlabeled inputs and actual source cache validation."""
import json
import math
from pathlib import Path
import numpy as np
from partition import ROOT,SPEC,partition,scope
from alignment import dtw,timed_words
from src.video_inputs import load_manifest,fixed_windows
from src.audio_inputs import resolve_video,decode_audio

DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/temporal_discourse_speech'


def selected_rows(smoke=False):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:
        rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def validate_source(processor,row,record,folder,decode=True):
    assert record['version']==SPEC['version'] and record['spec']==SPEC and record['GT_read'] is False
    assert (record['dataset'],record['video_id'],record['duration'])==(row['dataset'],row['video_id'],float(row['duration']))
    samples=np.load(folder/'audio.npy',allow_pickle=False);observed=np.load(folder/'observed.npy',allow_pickle=False)
    assert samples.dtype==np.float32 and observed.dtype==np.bool_ and samples.shape==observed.shape
    if decode:
        current,good,timeline=decode_audio(resolve_video(row),float(row['duration']))
        assert np.array_equal(samples,current) and np.array_equal(observed,good) and record['timeline']==timeline
    words=[]
    nominal=fixed_windows(float(row['duration']),SPEC['source_audio_seconds']);assert len(record['blocks'])==len(nominal)
    for number,(block,bounds) in enumerate(zip(record['blocks'],nominal)):
        assert block['i']==number and block['nominal']==list(bounds)
        from src.audio_inputs import crop_audio
        _,crop=crop_audio(samples,observed,*bounds);assert block['crop']==crop
        if not block['crop']['samples']:assert block['generation'] is None;continue
        generation=block['generation'];content=generation['content_tokens'];start,end=block['crop']['actual_interval']
        assert generation['block']==block['i'] and generation['prefix_tokens']==record['prefix_tokens']
        tokens=generation['tokens'];assert tokens[:len(record['prefix_tokens'])]==record['prefix_tokens']
        generated=tokens[len(record['prefix_tokens']):];eos=processor.tokenizer.eos_token_id
        assert generated==(content if generation['truncated'] else content+[eos])
        assert generation['text']==processor.tokenizer.decode(content,skip_special_tokens=True)
        if content:
            matrix=np.load(folder/'alignment'/f"block_{block['i']:06d}.npy",allow_pickle=False)
            lo,hi=block['crop']['samples']
            assert list(matrix.shape)==generation['alignment_matrix_shape'] and matrix.shape==(len(content)+1,min(1500,int(math.ceil((hi-lo)/320)))) and np.isfinite(matrix).all()
            ti,ai=dtw(-matrix);jumps=ai[np.r_[True,np.diff(ti)!=0]]*SPEC['alignment_frame_seconds']
            assert jumps.tolist()==generation['raw_jump_seconds']
            expected=timed_words(processor.tokenizer,content,record['language'],jumps,start,end,block['i'])
            assert generation['words']==expected
            words.extend(expected)
        else:assert not generation['words'] and not generation['raw_jump_seconds']
    for i,word in enumerate(words):word['id']=i
    assert record['words']==words and record['partition']==partition(words)
    assert record['windows']==[scope(words,record['partition'],a,b) for a,b in fixed_windows(float(row['duration']),8)]
    assert all(math.isfinite(record[key]) and record[key]>=0 for key in ('decode_seconds','language_seconds','proof_write_seconds','source_seconds'))
    assert abs(record['source_seconds']-record['decode_seconds']-record['language_seconds']-record['proof_write_seconds']-sum(b['seconds'] for b in record['blocks']))<1e-6


def source(row):
    folder=CACHE/row['dataset']/row['video_id']
    return json.loads((folder/'metadata.json').read_text()),folder
