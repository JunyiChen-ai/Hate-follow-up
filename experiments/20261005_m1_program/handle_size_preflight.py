#!/usr/bin/env python3
"""NoGT full-source size audit after observed fixed-five prefill OOM."""
import json
import logging
import os
import socket
import sys
from pathlib import Path
from PIL import Image
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from handle_inputs import selected_rows,sources
from handle_interface import planner_content,PLANNER_SYSTEM,CONSTANTS
from src.video_inputs import load_asr
from src.mllm_renderer import cpu_renderer


def main():
    out=ROOT/'runs/20261005_m1_program/handle_full_size_preflight';out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    j=cpu_renderer();asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};records=[];largest=[]
    for i,row in enumerate(selected_rows(False),1):
        source,windows,paths=sources(row,asr[row['dataset']].get(row['video_id'],[]));chunks=[]
        for offset in range(0,len(windows),CONSTANTS['planner_windows']):
            requested=windows[offset:offset+CONSTANTS['planner_windows']]
            rendered=j.render([j.turn('system',PLANNER_SYSTEM),dict(role='user',content=planner_content(source,requested))],True)
            size=len(j.tok.encode(rendered,add_special_tokens=False))
            chunks.append(dict(offset=offset,plain_tokens=size,characters=len(rendered)))
            largest.append((size,row['dataset'],row['video_id'],offset,rendered,paths))
            largest=sorted(largest,reverse=True,key=lambda x:x[:4])[:3]
        records.append(dict(dataset=row['dataset'],video_id=row['video_id'],segments=len(source['segments']),windows=len(windows),chunks=chunks))
        if i%25==0:logging.info('%d/333 source size audit',i)
    actual=[]
    for size,ds,vid,offset,rendered,paths in largest:
        images=[Image.open(ROOT/p).convert('RGB') for p in paths]
        try:enc=j.encode(rendered,images)
        finally:
            for im in images:im.close()
        actual.append(dict(dataset=ds,video_id=vid,offset=offset,plain_tokens=size,image_expanded_tokens=len(enc['input_ids'][0])))
    report=dict(GT_read=False,coverage=len(records),cache_version='R1 source-handle interface B; sources2026-10-05',
        purpose='Source-only size audit after actual OOM; no predicted/performance scores, no parameter selection',largest_actual=actual,records=records)
    (out/'source_sizes.json').write_text(json.dumps(report,indent=2)+'\n')
    logging.info('PREFLIGHT_DONE %s',json.dumps(actual))


if __name__=='__main__':main()
