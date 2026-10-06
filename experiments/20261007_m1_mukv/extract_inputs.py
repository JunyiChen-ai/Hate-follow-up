"""CPU actual media cache acquisition, separated from model inference."""
import argparse
import json
import logging
import os
import socket
import sys
import time
from inputs import ROOT,SPEC,CACHE,selected_rows
from src.fractional_window_frames import acquire,validate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261007_m1_mukv'/('source_smoke' if a.smoke else 'source_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),smoke=a.smoke,spec=SPEC,GT_read=False,
        code='experiments/20261007_m1_mukv/extract_inputs.py + src/fractional_window_frames.py;2026-10-07',command='python -u '+' '.join(sys.argv))
    path=out/'config.json'
    if path.exists():
        old=json.loads(path.read_text());assert all(old[k]==cfg[k] for k in cfg if k not in ('date','command'))
    else:path.write_text(json.dumps(cfg,indent=2)+'\n')
    n=1
    while (out/f'pipeline_attempt_{n:04d}.json').exists():n+=1
    audit_path=out/f'pipeline_attempt_{n:04d}.json';start=time.perf_counter();audit=dict(completed=False,GT_read=False,host=cfg['host'],rows=[])
    try:
        rows=selected_rows(a.smoke)
        for i,row in enumerate(rows,1):
            tick=time.perf_counter();folder=CACHE/row['dataset']/row['video_id'];folder.mkdir(parents=True,exist_ok=True);path=folder/'metadata.json';reused=path.exists()
            if reused:
                m=json.loads(path.read_text());assert m['spec']==SPEC and m['GT_read'] is False and (m['dataset'],m['video_id'])==(row['dataset'],row['video_id'])
            else:
                source=acquire(row,folder/'frames',SPEC['source_fractions'],SPEC['window_seconds'])
                m=dict(spec=SPEC,dataset=row['dataset'],video_id=row['video_id'],GT_read=False,host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),source=source,
                    code=cfg['code'],command=cfg['command'])
            validate(m['source'],row,folder/'frames',SPEC['source_fractions'],SPEC['window_seconds'])
            if not reused:
                temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(m)+'\n');temporary.replace(path)
            audit['rows'].append(dict(dataset=row['dataset'],video_id=row['video_id'],reused=reused,elapsed_seconds=time.perf_counter()-tick,
                frames=sum(len(p) for p in m['source']['selected_indices']),missing=len(m['source']['uncovered_windows'])))
            audit['elapsed_seconds']=time.perf_counter()-start;audit_path.write_text(json.dumps(audit,indent=2)+'\n');logging.info('%d/%d %s/%s',i,len(rows),row['dataset'],row['video_id'])
        CACHE.mkdir(parents=True,exist_ok=True)
        (CACHE/'PROVENANCE.md').write_text('# Actual MuKV source pixels\n\nGenerated2026-10-07 on '+socket.gethostname()+' by experiments/20261007_m1_mukv/extract_inputs.py + src/fractional_window_frames.py. Current spec and run commands/date/input host/path recorded in each metadata and runs/20261007_m1_mukv/source_{smoke,main}/. Four fractional8s in-window targets1/8,3/8,5/8,7/8; actual first PTS at/after each, explicit missing/dedup, fullcurrentrawPTS/RGB parse replay. No model/GT/label input or contentchecksum/Git-ID provenance. Model inference reads this cache only; source acquisition/decode/verification/I/O cost remains per-new-video work.\n')
        audit['completed']=True;logging.info('SOURCE_INPUTS_DONE coverage=%d',len(rows))
    finally:
        audit['elapsed_seconds']=time.perf_counter()-start;audit_path.write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
