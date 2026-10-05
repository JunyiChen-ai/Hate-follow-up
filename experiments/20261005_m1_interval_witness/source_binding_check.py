"""Read-only CPU portability check of complete source and native-path inputs."""
import argparse
import json
import logging
import socket
import time
from inputs import ROOT,CACHE,DATASETS,selected_rows,validate
from measure_r2 import native_input
from reader import encoding
from src.mllm_renderer import cpu_position_renderer
from src.video_inputs import load_asr


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--out',default='runs/20261005_m1_interval_witness/r2_source_binding_check');a=ap.parse_args();out=ROOT/a.out;assert out.resolve().is_relative_to(ROOT);out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()]);logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),GT_read=False,smoke=a.smoke,scope='current raw pixels/source tokens and native Yes/No path binding; no model margins/GT',code='experiments/20261005_m1_interval_witness/source_binding_check.py + unchanged inputs/reader/native_input;2026-10-06')
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');j=cpu_position_renderer();asr={ds:load_asr(ds) for ds in DATASETS};results=[]
    for row in selected_rows(a.smoke):
        path=CACHE/row['dataset']/row['video_id']/'metadata.json';text=path.read_text();stamp=path.stat().st_mtime_ns;m=json.loads(text);segments=asr[row['dataset']].get(row['video_id'],[]);validate(m,row,segments,j);counts={}
        for stance in ('Yes','No'):
            ctx=native_input(j,row,segments,stance);_,trace=encoding(j,ctx,m,0,'visual');assert trace['input_tokens'][:ctx['stance_cache_tokens']]==ctx['native_stance_ids'];counts[stance]=len(trace['input_tokens'])
        assert text==path.read_text() and stamp==path.stat().st_mtime_ns
        results.append(dict(dataset=row['dataset'],video_id=row['video_id'],windows=len(m['windows']),source_current=True,readonly=True,branch_tokens=counts,original_source_host=m.get('host'),source_standalone_seconds=m['standalone_seconds']));logging.info('source binding exact %s/%s',row['dataset'],row['video_id'])
    (out/'summary.json').write_text(json.dumps(dict(**cfg,PASS=True,coverage=len(results),rows=results),indent=2)+'\n');logging.info('SOURCE_BINDING_DONE coverage=%d',len(results))


if __name__=='__main__':main()
