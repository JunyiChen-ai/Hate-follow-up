#!/usr/bin/env python3
"""Read each R3 factual-control arm independently from one native stance cache."""
import argparse
import copy
import json
import logging
import math
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.mllm_judge import Judge,MODEL,yesno_question
from src.video_inputs import FPS,frame_paths,load_asr,fixed_windows,window_text
from src.mllm_renderer import cpu_renderer
from extract import CACHE,selected_rows,validate_cached
from control_extract import CONTROL_VERSION,CONTROL_CACHE,validate_control
from control_inputs import ARMS
from measure import tick,context,standard,local_visual,extend_observations,record,validate_extension_inputs


class BindingRenderer:
    """Encode the identical current native image prefix once per video bundle."""
    def __init__(self,renderer):self.renderer=renderer;self.saved=None
    def __getattr__(self,name):return getattr(self.renderer,name)
    def encode_prefix(self,msgs,files):
        if self.saved is None or self.saved[0]!=msgs or self.saved[1]!=files:
            self.saved=(copy.deepcopy(msgs),copy.deepcopy(files),self.renderer.encode_prefix(msgs,files))
        return self.saved[2]


@torch.no_grad()
def read_controls(j,row,segments,metadata,controls,folder,smoke):
    first=j.forward_calls;torch.cuda.reset_peak_memory_stats();start=tick()
    frames=frame_paths(row['dataset'],row['video_id'],20)
    wins=fixed_windows(float(row['duration']),8);texts=[window_text(segments,a,b) for a,b in wins]
    t=tick();native_cache,native=context(j,frames,segments);prefix_seconds=tick()-t
    vs=[];ss=[];t=tick()
    for i,((a,b),body) in enumerate(zip(wins,texts)):
        vs.append(standard(j,native_cache,native,yesno_question(i,len(wins),a,b,body,'visual')))
        ss.append(standard(j,native_cache,native,yesno_question(i,len(wins),a,b,body,'speech')) if body.strip() else None)
    reference_seconds=tick()-t;shared_actual=j.forward_calls-first
    base=record(row,native,vs,ss,prefix_seconds+reference_seconds)
    assert shared_actual==base['calls']
    outputs={};details={};arm_costs={}
    for arm in ARMS:
        before=j.forward_calls;t=tick();cache=copy.deepcopy(native_cache);copy_seconds=tick()-t
        inputs=controls['arms'][arm];arm_meta={**metadata,'packets':inputs['packets']}
        t=tick();ctx,extension=extend_observations(j,cache,native,metadata,smoke,inputs['observation']);extension_seconds=tick()-t
        nv=[];ns=[];traces=[];visual_seconds=speech_seconds=diagnostic_seconds=0.;diagnostics=0;speech_cloned=False
        for i,((a,b),body,packet) in enumerate(zip(wins,texts,inputs['packets'])):
            vq=yesno_question(i,len(wins),a,b,body,'visual');sq=yesno_question(i,len(wins),a,b,body,'speech')
            t=tick();v,trace=local_visual(j,cache,ctx,vq,packet,arm_meta,folder,smoke and i==0);visual_seconds+=tick()-t;nv.append(v)
            t=tick();s=standard(j,cache,ctx,sq) if body.strip() else None;speech_seconds+=tick()-t;ns.append(s)
            if smoke and i==0:
                t=tick();cloned=copy.deepcopy(cache);replay,_=local_visual(j,cloned,ctx,vq,packet,arm_meta,folder)
                assert replay==v;del cloned;diagnostic_seconds+=tick()-t;diagnostics+=1;trace['cloned_margin']=replay
            if smoke and s is not None and not speech_cloned:
                t=tick();cloned=copy.deepcopy(cache);replay=standard(j,cloned,ctx,sq)
                assert replay==s;del cloned;diagnostic_seconds+=tick()-t;diagnostics+=1;speech_cloned=True;trace['cloned_speech_margin']=replay
            traces.append(trace)
        del cache
        read_seconds=prefix_seconds+extension_seconds+visual_seconds+speech_seconds
        additional_input_seconds=controls['cost']['input_extra_seconds_by_arm'][arm]
        outputs[arm]=record(row,ctx,nv,ns,metadata['standalone_seconds']+additional_input_seconds+read_seconds)
        # Control acquisition is reported as the measured joint diagnostic work;
        # inherited main topology means these are not independent deployable methods.
        arm_costs[arm]=dict(actual_forwards=j.forward_calls-before,diagnostic_forwards=diagnostics,
            cache_copy_seconds=copy_seconds,extension_seconds=extension_seconds,visual_seconds=visual_seconds,
            speech_seconds=speech_seconds,diagnostic_seconds=diagnostic_seconds,read_seconds=read_seconds,
            main_input_preprocessing_seconds=metadata['standalone_seconds'],
            control_input_acquisition_seconds=additional_input_seconds,
            additional_control_acquisition_required=additional_input_seconds>0)
        assert arm_costs[arm]['actual_forwards']==outputs[arm]['calls']-3+diagnostics
        details[arm]=dict(extension=extension,traces=traces,base_prefix_tokens=native['prefix_tokens'],
            new_prefix_tokens=ctx['prefix_tokens'],native_conversation={k:native[k] for k in ('msgs','head','history')})
    del native_cache
    actual=j.forward_calls-first
    assert actual==shared_actual+sum(c['actual_forwards'] for c in arm_costs.values())
    checks=dict(GT_read=False,shared_reference_forwards=shared_actual,actual_joint_forwards=actual,
        prefix_seconds=prefix_seconds,reference_seconds=reference_seconds,arm_costs=arm_costs,
        reader_joint_wall_seconds=tick()-start,additional_control_acquisition=controls['cost'],
        main_input_preprocessing_seconds=metadata['standalone_seconds'],
        main_input_actual_forwards=metadata['actual_forwards'],
        peak_reader_GiB=torch.cuda.max_memory_allocated()/2**30)
    return dict(dataset=row['dataset'],video_id=row['video_id'],control_version=CONTROL_VERSION,
        base=base,arms=outputs,checks=checks,details=details,segments=[list(s) for s in segments])


def validate_bundle(row,bundle,metadata,controls,segments,renderer,smoke):
    assert bundle['control_version']==CONTROL_VERSION and bundle['checks']['GT_read'] is False
    assert (bundle['dataset'],bundle['video_id'])==(row['dataset'],row['video_id'])
    assert bundle['segments']==[list(s) for s in segments]
    assert set(bundle['arms'])==set(bundle['details'])==set(bundle['checks']['arm_costs'])==set(ARMS)
    wins=fixed_windows(float(row['duration']),8);base=bundle['base'];c=bundle['checks']
    for name,r in [('base',base)]+list(bundle['arms'].items()):
        assert (r['dataset'],r['video_id'])==(row['dataset'],row['video_id'])
        assert r['duration']==float(row['duration']) and r['native_rate']==FPS and r['error'] is None
        assert r['extra']['z_video']==base['extra']['z_video'] and np.isfinite(r['extra']['z_video'])
        assert r['extra']['stance']==('Yes' if r['extra']['z_video']>0 else 'No')
        ww=r['extra']['windows'];assert len(ww)==len(wins)
        for i,(w,(a,b),original) in enumerate(zip(ww,wins,base['extra']['windows'])):
            assert (w['i'],w['start'],w['end'])==(i,a,b)
            assert w['z']==max(w['z_visual'],w.get('z_speech',float('-inf')))
            assert ('z_speech' in w)==('z_speech' in original)
        idx=np.clip(((np.arange(math.ceil(r['duration']*FPS))+.5)/FPS//8).astype(int),0,len(wins)-1)
        assert np.array_equal(r['score_curve'],np.asarray([w['z'] for w in ww])[idx]) and np.isfinite(r['score_curve']).all()
        assert r['calls']==3+int(name!='base')+len(wins)+sum('z_speech' in w for w in ww)
    assert c['shared_reference_forwards']==base['calls']
    assert c['actual_joint_forwards']==c['shared_reference_forwards']+sum(v['actual_forwards'] for v in c['arm_costs'].values())
    assert c['additional_control_acquisition']==controls['cost'] and c['main_input_preprocessing_seconds']==metadata['standalone_seconds']
    for arm in ARMS:
        r=bundle['arms'][arm];detail=bundle['details'][arm];cost=c['arm_costs'][arm];inputs=controls['arms'][arm]
        validate_extension_inputs(renderer,row,segments,base,detail,metadata,inputs['observation'])
        e=detail['extension'];assert e['prefix_positions_exact'] is True and e['fresh_render_verified']==smoke
        assert len(detail['traces'])==len(wins)
        diagnostics=0
        for trace,packet,w in zip(detail['traces'],inputs['packets'],r['extra']['windows']):
            assert trace['packet']==packet and trace['margin']==w['z_visual'] and trace['prefix_positions_exact'] is True
            assert len(trace['paths'])==len(trace['new_image_counts'])==len(packet['pool_members'])
            expected_paths=[str((CONTROL_CACHE/row['dataset']/row['video_id']/'witnesses'/f'frame_{metadata["entries"][i]["index"]:08d}.png').relative_to(ROOT)) for i in packet['pool_members']]
            assert trace['paths']==expected_paths
            if 'cloned_margin' in trace:assert trace['cloned_margin']==w['z_visual'];diagnostics+=1
            if 'cloned_speech_margin' in trace:assert trace['cloned_speech_margin']==w['z_speech'];diagnostics+=1
        assert sum(t['fresh_render_verified'] for t in detail['traces'])==int(smoke)
        assert diagnostics==cost['diagnostic_forwards']==(1+int(any('z_speech' in w for w in r['extra']['windows'])) if smoke else 0)
        assert cost['actual_forwards']==r['calls']-3+diagnostics
        assert cost['read_seconds']==c['prefix_seconds']+cost['extension_seconds']+cost['visual_seconds']+cost['speech_seconds']
        additional=controls['cost']['input_extra_seconds_by_arm'][arm]
        assert cost['control_input_acquisition_seconds']==additional
        assert r['extra']['standalone_seconds']==metadata['standalone_seconds']+additional+cost['read_seconds']
        assert cost['additional_control_acquisition_required']==(additional>0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261004_m1_tree'/('r3_controls_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    config=dict(model=MODEL,control_version=CONTROL_VERSION,host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),
        smoke=a.smoke,GT_read=False,arms=list(ARMS),seed=0,
        code='experiments/20261004_m1_tree/{measure,control_inputs,control_extract,control_measure}.py; sources2026-10-05')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows=selected_rows(a.smoke);asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')}
    renderer=BindingRenderer(cpu_renderer());torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=0
    def hook(*_):j.forward_calls+=1
    handle=j.model.model.register_forward_pre_hook(hook);bundles=[]
    for number,row in enumerate(rows,1):
        folder=CONTROL_CACHE/row['dataset']/row['video_id'];metadata,features=validate_cached(CACHE/row['dataset']/row['video_id'],row)
        controls=validate_control(folder,row,metadata,features,renderer.tok);segments=asr[row['dataset']].get(row['video_id'],[])
        path=out/'records'/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():bundle=json.loads(path.read_text())
        else:
            bundle=read_controls(j,row,segments,metadata,controls,folder,a.smoke)
            validate_bundle(row,bundle,metadata,controls,segments,renderer,a.smoke)
            partial=path.with_suffix('.partial');partial.write_text(json.dumps(bundle)+'\n');partial.replace(path)
        validate_bundle(row,bundle,metadata,controls,segments,renderer,a.smoke);bundles.append(bundle)
        logging.info('%d/%d %s/%s',number,len(rows),row['dataset'],row['video_id'])
    handle.remove()
    for name in ('base',)+ARMS:
        folder=out/name;folder.mkdir(parents=True,exist_ok=True);partial=folder/'predictions.partial'
        with partial.open('w') as f:
            for b in bundles:f.write(json.dumps(b['base'] if name=='base' else b['arms'][name])+'\n')
        partial.replace(folder/'predictions.jsonl')
    logging.info('CONTROL_READ_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
