"""Native-S-only revision; actual36-layer CPU and previous-R1 identity."""
import copy,json,os,socket,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from inputs import ROOT,SPEC
from model_selfcheck import fixture,Qwen3VLConfig,Qwen3VLModel
import measure as m


def main():
    assert m.REVISION==2
    out=ROOT/'runs/20261006_m1_merit/r2_reader_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    print(socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    prior={'__name__':'before_R2','__file__':str(ROOT/'experiments/20261006_m1_merit/measure.py')}
    exec(compile((out/'before_R2_measure.py').read_text(),'before_R2_measure.py','exec'),prior)
    class Clock:
        def __init__(self):self.i=0
        def __call__(self,*args):self.i+=1;return float(self.i)
    torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.reset_peak_memory_stats=lambda:None;torch.cuda.max_memory_allocated=lambda:0
    p=out/'pixels.png';Image.fromarray(np.random.default_rng(0).integers(0,256,(8,8,3),dtype=np.uint8)).save(p)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=36,
        num_attention_heads=32,num_key_value_heads=8,head_dim=128,max_position_embeddings=1024,
        rope_scaling={'rope_type':'default','mrope_section':[24,20,20]}),vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,
        num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),
        image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
    cases=[]
    for dt in (torch.float32,torch.bfloat16):
        for count in (18,20):
            model=Qwen3VLModel(cfg).eval().to(dt);j,frames=fixture(model,count,dt,p)
            m.frame_paths=lambda *args:frames;prior['frame_paths']=lambda *args:frames;j.forward_calls=j.vision_calls=0
            j.branch_ids=lambda msgs,q,*args,**kw:([60,61,63],'Qremote') if 'Actual remote speech context' in q else ([60,61],'Q')
            hooks=[model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
            try:
                for mode in ('no_remote','before','after','both','no_frames'):
                    row=dict(dataset='synthetic',video_id=mode,duration=24.);segments=[(0.,7.,'literal0'),(16.,23.,'literal2')]
                    windows=[dict(i=i,start=8*i,end=8*i+8,body=('literal'+str(i) if i!=1 else ''),frames=[] if mode=='no_frames' else
                        [dict(id='p'+str(k),time=8*i+2+3*k,path=str(p.relative_to(ROOT))) for k in range(2)]) for i in range(3)]
                    packets=[]
                    for i in range(3):
                        ids=([i-1] if i and mode in ('before','both','no_frames') else [])+([i+1] if i<2 and mode in ('after','both','no_frames') else [])
                        packets.append(dict(source_ids=ids))
                    meta=dict(windows=windows,packets=packets,records=['DO_NOT_USE_CAPTION']*3,cost=dict(standalone_seconds=43.,actual_forwards=29,
                        actual_vision_forwards=7,caption_calls=3,filter_calls=1,embedding_calls=9),peak_GiB=0.)
                    saved=copy.deepcopy(meta);m.REVISION=2;m.clock=Clock();b=m.read_video(j,row,segments,meta,True);m.validate_bundle(row,b,segments,meta,j,True)
                    assert meta==saved and all(t['native_speech']==t['new_speech'] for t in b['traces'])
                    m.REVISION=1;m.clock=Clock();r1=m.read_video(j,row,segments,meta,True);m.validate_bundle(row,r1,segments,meta,j,True)
                    prior['clock']=Clock();old=prior['read_video'](j,row,segments,meta,True);assert r1==old
                    assert b['base']['score_curve']==r1['base']['score_curve'] and b['native_ctx']==r1['native_ctx'] and b['native_rope']==r1['native_rope']
                    for a,c in zip(b['traces'],r1['traces']):
                        assert a['new_visual']==c['new_visual'] and a['source_branch']==c['source_branch']
                        assert 'Actual remote speech context' not in (a['new_speech_question'] or '')
                    m.REVISION=2
                    for fault in ('revision','speech','question','source_cost'):
                        bad=copy.deepcopy(b)
                        if fault=='revision':bad.pop('reader_revision')
                        elif fault=='speech':bad['traces'][0]['new_speech']+=1
                        elif fault=='question':bad['traces'][0]['new_speech_question']+=' REMOTE'
                        else:bad['checks']['source_seconds']=0
                        try:m.validate_bundle(row,bad,segments,meta,j,True)
                        except AssertionError:pass
                        else:raise AssertionError(fault)
                    cases.append(dict(dtype=str(dt),native_frames=count,mode=mode,layers=36,R1_snapshot_exact=True,R2_V_R1_exact=True,
                        native_S_exact=True,source_cost_kept=True,clones_exact=True,mutations_rejected=True))
                    print('R2_CASE_PASS',dt,count,mode,flush=True)
            finally:
                for h in hooks:h.remove()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,pretrained=False,cases=cases,
        scope='actual36layer randomweights/ASR-presence-sensitive fixture; realprocessor/GPU separately required'),indent=2)+'\n')
    print('READER_R2_CPU_PASS',flush=True)


if __name__=='__main__':main()
