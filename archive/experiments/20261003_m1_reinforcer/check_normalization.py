#!/usr/bin/env python3
"""Six-video no-GT witness for the finalNorm boundary check, including failed input."""
import json
import logging
import os
import socket
import time
import torch
from measure import ROOT,Judge,MODEL,ResidualReinforcer,load_manifest,load_asr,read_video


def main():
    out=ROOT/'runs/20261003_m1_reinforcer/normalization_probe';out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    torch.manual_seed(0);allrows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',['HateMM','HateClipSeg'])
    rows=[r for ds in ('HateMM','HateClipSeg') for r in [v for v in allrows if v['dataset']==ds][:2]]
    rows += [r for r in allrows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    rows.append(allrows[103]);assert len({(r['dataset'],r['video_id']) for r in rows})==6
    historical={(r['dataset'],r['video_id']):r for r in map(json.loads,(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl').open())}
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};j=Judge(MODEL);eng=ResidualReinforcer(j);j.forward_calls=0
    def count(*_):j.forward_calls+=1
    counter=j.model.model.register_forward_pre_hook(count);ordinary_read=eng.read;diagnostics=[]
    def capture(*args,**kwargs):
        r=ordinary_read(*args,**kwargs);diagnostics.append(r['normalization_recompute']);return r
    eng.read=capture;checks=[]
    for row in rows:
        diagnostics.clear();start=time.perf_counter();recs,check,tokens=read_video(j,eng,row,asr[row['dataset']].get(row['video_id'],[]),True)
        old=historical[row['dataset'],row['video_id']];base=recs['base']
        assert base['extra']['z_video']==old['extra']['z_video'] and base['score_curve']==old['score_curve']
        for w,v in zip(base['extra']['windows'],old['extra']['windows']):
            assert all(w.get(k)==v.get(k) for k in ('start','end','z_visual','z_speech'))
        checks.append({'dataset':row['dataset'],'video_id':row['video_id'],'native_exact':True,
            'normalization_recompute_max':{k:max(d[k] for d in diagnostics) for k in ('hidden_max','logit_max')},
            'actual_forwards':check['actual_forwards'],'seconds':time.perf_counter()-start})
        logging.info('PASS %s %s %s',row['dataset'],row['video_id'],checks[-1]['normalization_recompute_max'])
    counter.remove();eng.close()
    (out/'checks.json').write_text(json.dumps({'no_GT':True,'checks':checks},indent=2)+'\n');logging.info('PROBE_DONE')


if __name__=='__main__':main()
