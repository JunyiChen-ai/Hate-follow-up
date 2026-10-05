#!/usr/bin/env python3
"""Whole333 native JPEG/token/ASR interval-domain preflight, without GT/model weights."""
import json
import os
import socket
import numpy as np
import torch
from transformers import AutoConfig
from rote import ROOT, temporal_pairs, coefficients, prefix_sources, question_sources, source_intervals
from measure import selected_rows, DATASETS
from src.mllm_renderer import cpu_renderer
from src.mllm_judge import MODEL, VIDEO_QUESTION, yesno_question
from src.video_inputs import frame_paths, load_asr, fixed_windows, window_text


def main():
    torch.set_num_threads(4)
    j=cpu_renderer();asr={ds:load_asr(ds) for ds in DATASETS}
    pairs,freq=temporal_pairs(AutoConfig.from_pretrained(MODEL,local_files_only=True).text_config)
    result=[]
    for n,row in enumerate(selected_rows(),1):
        ds,vid=row['dataset'],row['video_id'];segments=asr[ds].get(vid,[])
        frames=frame_paths(ds,vid,20);msgs,paths=j.prefix_messages(frames,segments);text,enc=j.encode_prefix(msgs,paths)
        mapping=prefix_sources(j,text,enc,segments);intervals=source_intervals(mapping,segments)
        _,_,norm=coefficients(np.asarray(intervals).reshape(-1,2),freq)
        qids,qtext=j.branch_ids(msgs,VIDEO_QUESTION,head_text=text)
        aids,atext=j.answer_ids(msgs,VIDEO_QUESTION,'Yes')
        history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant','Yes')]
        head=text+qtext+atext
        windows=[];wins=fixed_windows(float(row['duration']),8)
        for i,(a,b) in enumerate(wins):
            body=window_text(segments,a,b)
            if not body.strip():continue
            q=yesno_question(i,len(wins),a,b,body,'speech')
            ids,suffix=j.branch_ids(msgs,q,history,head_text=head)
            current=question_sources(j,suffix,ids,segments,a,b,body);assert current
            timed=source_intervals(current,segments)
            _,_,current_norm=coefficients(timed,freq)
            windows.append(dict(i=i,suffix_tokens=len(ids),source_tokens=len(current),normalization_min=float(current_norm.min()),
                normalization_max=float(current_norm.max()),multi_source_tokens=sum(len(r['source_ids'])>1 for r in current)))
        result.append(dict(dataset=ds,video_id=vid,actual_frames=len(frames),native_prefix_tokens=int(enc['input_ids'].shape[1]),
            prefix_source_tokens=len(mapping),prefix_multi_source_tokens=sum(len(r['source_ids'])>1 for r in mapping),
            normalization_min=float(norm.min()) if len(norm) else None,normalization_max=float(norm.max()) if len(norm) else None,
            speech_windows=windows))
        print(f'{n}/333 {ds}/{vid}',flush=True)
    out=ROOT/'runs/20261005_m1_rote/full_input_preflight';out.mkdir(parents=True,exist_ok=True)
    norms=[r['normalization_min'] for r in result if r['normalization_min'] is not None]+[w['normalization_min'] for r in result for w in r['speech_windows']]
    summary=dict(GT_read=False,host=socket.gethostname(),coverage=len(result),pass_all=True,normalization_min=min(norms),
        frequency_pairs=pairs.tolist(),frequencies=freq.tolist(),speech_windows=sum(len(r['speech_windows']) for r in result),
        max_native_prefix=max(r['native_prefix_tokens'] for r in result),max_speech_query=max(w['suffix_tokens'] for r in result for w in r['speech_windows']))
    (out/'source_inputs.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));print('PREFLIGHT_PASS',flush=True)


if __name__=='__main__':main()
