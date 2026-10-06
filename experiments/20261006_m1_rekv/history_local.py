"""The predeclared H0+L0 cell, with fresh exact H0 replay; no annotations."""
import argparse
import json
import logging
import os
import socket
import sys
import time
import torch
from inputs import ROOT,SPEC,CACHE,DATASETS,selected_rows,validate_input
from controls import record,equal_r0,selection_batch,validate_selection,validate_variant
from measure import read_video
from binding import IMPLEMENTATION
from src.mllm_judge import Judge,MODEL
from src.video_inputs import load_asr

VERSION='H0-plus-L0-predeclared-2026-10-07-v1'


def folder(smoke):
    return ROOT/'runs/20261006_m1_rekv'/('history_local_'+('smoke' if smoke else 'main'))


def validate(j,row,segments,meta,bundle,smoke):
    main=record(ROOT/'runs/20261006_m1_rekv/r1_full_main',row)
    prior=record(ROOT/'runs/20261006_m1_rekv/controls_main_history',row)
    validate_variant(j,row,segments,meta,bundle,smoke,'history',main)
    # Fresh H0 is an engineering replay. Ignore only prior smoke clone flags;
    # all source/query/key/selection/margin/native evidence must be identical.
    equal_r0(bundle,prior)
    assert set(bundle['controls'])=={'L0'}
    arm=bundle['controls']['L0'];validate_selection(bundle,arm,smoke)
    assert not any(layer['remote_ids'] for t in arm['traces'] if t['branch'] for layer in t['branch']['layers'])
    assert not any(b['ancestors'] or b['direct_ancestors'] for b in bundle['source_blocks'])
    assert bundle['batch_actual_forwards']==bundle['checks']['actual_forwards']+arm['actual_forwards']
    assert bundle['batch_actual_vision']==bundle['checks']['actual_vision']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=folder(a.smoke);out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import av,transformers
    cfg=dict(version=VERSION,host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),smoke=a.smoke,GT_read=False,
        model=MODEL,spec=SPEC,implementation=IMPLEMENTATION,torch=torch.__version__,transformers=transformers.__version__,av=av.__version__,
        source='experiments/20261006_m1_rekv/{history_local,controls,measure,reader,validate}.py;2026-10-07',command='python -u '+' '.join(sys.argv),
        arm='HL0',source_previous_frames=0,explicit_remote=0,fresh_H0_identity_replay=True)
    number=1
    while (out/f'pipeline_attempt_{number:04d}.json').exists():number+=1
    audit_path=out/f'pipeline_attempt_{number:04d}.json';start=time.perf_counter()
    audit=dict(completed=False,GT_read=False,host=socket.gethostname(),videos=[]);hooks=[];j=None;predictions=[]
    try:
        path=out/'config.json'
        if path.exists():
            old=json.loads(path.read_text());assert all(old[k]==cfg[k] for k in cfg if k not in ('date','command'))
        else:path.write_text(json.dumps(cfg,indent=2)+'\n')
        asr={ds:load_asr(ds) for ds in DATASETS}
        tick=time.perf_counter();torch.manual_seed(0);j=Judge(MODEL);audit['model_load_seconds']=time.perf_counter()-tick
        j.forward_calls=j.vision_calls=0
        hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),
            j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
        rows=selected_rows(a.smoke)
        for ordinal,row in enumerate(rows,1):
            tick=time.perf_counter();first=j.forward_calls;vision=j.vision_calls;memory=None
            segments=asr[row['dataset']].get(row['video_id'],[])
            meta=json.loads((CACHE/row['dataset']/row['video_id']/'metadata.json').read_text());validate_input(meta,row)
            path=out/'records'/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True);reused=path.exists()
            try:
                if reused:bundle=json.loads(path.read_text())
                else:
                    callback=lambda j,c,x,m,r:selection_batch(j,c,x,m,r,row,out,a.smoke,arms=('L0',))
                    bundle,memory=read_video(j,row,segments,meta,out,a.smoke,previous_frames=0,after_read=callback)
                    bundle['intervention']=dict(arm='H0',source_previous_frames=0,native_global_context_kept=True)
                    bundle['batch_actual_forwards']=j.forward_calls-first;bundle['batch_actual_vision']=j.vision_calls-vision
                    bundle['batch_peak_GiB']=torch.cuda.max_memory_allocated()/2**30 if j.device.type=='cuda' else 0.
                validate(j,row,segments,meta,bundle,a.smoke)
                if not reused:
                    partial=path.with_suffix('.partial');partial.write_text(json.dumps(bundle)+'\n');partial.replace(path)
            finally:
                if memory is not None:memory.close(release=path.exists())
            pred=dict(bundle['controls']['L0']['prediction']);pred['method']='m1_rekv_HL0';pred['code_path']='experiments/20261006_m1_rekv/history_local.py'
            predictions.append(pred)
            audit['videos'].append(dict(dataset=row['dataset'],video_id=row['video_id'],reused=reused,elapsed_seconds=time.perf_counter()-tick,
                current_actual_forwards=j.forward_calls-first,current_actual_vision=j.vision_calls-vision))
            audit['elapsed_seconds']=time.perf_counter()-start;audit_path.write_text(json.dumps(audit,indent=2)+'\n')
            logging.info('%d/%d %s/%s HL0',ordinal,len(rows),row['dataset'],row['video_id'])
        arm=out/'HL0';arm.mkdir(exist_ok=True);(arm/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (arm/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
        audit['completed']=True;logging.info('HISTORY_LOCAL_DONE coverage=%d',len(rows))
    finally:
        for h in hooks:h.remove()
        audit['total_actual_forwards']=j.forward_calls if j else 0;audit['total_actual_vision']=j.vision_calls if j else 0
        audit['elapsed_seconds']=time.perf_counter()-start;audit_path.write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
