"""Actual processor timestamp/pixel/position binding on unchanged fixed five."""
import json
import os
import socket
import time
import torch
from inputs import ROOT,CACHE,selected_rows,source_frames
from reader import encode_source
from controls import record,time_mapping
from src.mllm_renderer import cpu_position_renderer


def main():
    start=time.perf_counter();out=ROOT/'runs/20261006_m1_rekv/control_expanded_preflight';out.mkdir(parents=True,exist_ok=True)
    print(socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    j=cpu_position_renderer();rows=[];root=ROOT/'runs/20261006_m1_rekv/r1_full_main'
    for row in selected_rows(True):
        reference=record(root,row);meta=json.loads((CACHE/row['dataset']/row['video_id']/'metadata.json').read_text())
        frames=source_frames(row,meta['source']);mapping=time_mapping(reference);changed=0
        for frame,presented,block in zip(frames,mapping['presented_times'],reference['source_blocks']):
            before,relative,image_rows,evidence=encode_source(j,reference['native_ctx'],frame)
            after,new_relative,new_image_rows,new=encode_source(j,reference['native_ctx'],{**frame,'presented_time':presented})
            assert evidence==block['input_evidence']
            assert after['input_ids'].shape==before['input_ids'].shape
            assert new_image_rows==image_rows and torch.equal(new_relative,relative)
            assert torch.equal(after['image_grid_thw'],before['image_grid_thw']) and torch.equal(after['pixel_values'],before['pixel_values'])
            if presented!=frame['time']:
                assert not torch.equal(before['input_ids'],after['input_ids']),'timestamp change never entered source input'
                changed+=1
            assert new['message']['content'][1]==evidence['message']['content'][1]
        rows.append(dict(dataset=row['dataset'],video_id=row['video_id'],source_frames=len(frames),changed_actual_source_inputs=changed))
        print('EXPANDED_BOUND',row['dataset'],row['video_id'],flush=True)
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,coverage=5,per_video=rows,
        scope='actual native tokenizer/processor/pixels/mRoPE; unchanged fixed5, no model forward/GT/performance',elapsed_seconds=time.perf_counter()-start),indent=2)+'\n')
    print('EXPANDED_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':main()
