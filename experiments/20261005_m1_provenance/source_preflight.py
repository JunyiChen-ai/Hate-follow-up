#!/usr/bin/env python3
"""Full333 native/source-text CPU audit before GPU, without generated facts."""
import json
import logging
import socket
from PIL import Image
from inputs import ROOT,selected_rows
from src.actual_video_frames import resolve_video
from src.video_inputs import load_asr,frame_paths,fixed_windows,window_text
from src.mllm_renderer import cpu_renderer


def main():
    out=ROOT/'runs/20261005_m1_provenance/source_preflight';out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    j=cpu_renderer();asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};records=[]
    for number,row in enumerate(selected_rows(False),1):
        frames=frame_paths(row['dataset'],row['video_id'],20);segments=asr[row['dataset']].get(row['video_id'],[])
        assert frames and resolve_video(row).is_file();msgs,paths=j.prefix_messages(frames,segments)
        _,enc=j.encode_prefix(msgs,paths)
        bodies=[window_text(segments,a,b) for a,b in fixed_windows(float(row['duration']),8)]
        records.append(dict(dataset=row['dataset'],video_id=row['video_id'],raw=str(resolve_video(row)),
            actual_native_count=len(frames),native_expanded_tokens=len(enc['input_ids'][0]),
            native_image_grid=enc['image_grid_thw'].tolist(),windows=len(bodies),
            maximum_local_speech_tokens=max(len(j.tok.encode(b,add_special_tokens=False)) for b in bodies)))
        if number%25==0:logging.info('%d/333 actual native/text source audit',number)
    report=dict(host=socket.gethostname(),GT_read=False,coverage=333,
        scope='Actual complete native image/text source audit; local new pixels are actual in fixed5 CPU checks and acquired at GPU extraction; generated link-table sizes are measured only once actual ledgers exist',records=records)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');logging.info('SOURCE_PREFLIGHT_DONE')


if __name__=='__main__':main()
