"""Complete-video predeclared controls; fresh KV, no labels or score routing."""
import argparse
import copy
import json
import logging
import os
import socket
import sys
import time
from functools import cache
import numpy as np
import torch
from inputs import ROOT,SPEC,CACHE,DATASETS,selected_rows,validate_input,windows_for
from control_selection import ARMS,Selection,exposure
from control_time import permutation
from measure import read_video,prediction
from reader import visual_margin,clock
from validate import validate_bundle
from retrieval import remote_selection
from layout import pack_positions
from binding import IMPLEMENTATION
from src.mllm_judge import Judge,MODEL
from src.stance_cache import margin
from src.video_inputs import load_asr

VERSION='predeclared-controls-2026-10-06-v1'
MODES=('selection','time','history')


def record(root,row):
    return json.loads((root/'records'/row['dataset']/(row['video_id']+'.json')).read_text())


def equal_native(bundle,reference):
    assert bundle['native_ctx']==reference['native_ctx'] and bundle['native_rope']==reference['native_rope']
    assert bundle['segments']==reference['segments']
    assert bundle['base']['extra']['windows']==reference['base']['extra']['windows']
    assert bundle['base']['score_curve']==reference['base']['score_curve']
    for key in ('runtime','native','pixel_shape','pixel_dtype'):
        assert bundle['binding'][key]==reference['binding'][key]
    assert np.array_equal(np.load(ROOT/bundle['binding']['pixel_path'],allow_pickle=False),np.load(ROOT/reference['binding']['pixel_path'],allow_pickle=False))


def equal_r0(bundle,reference):
    equal_native(bundle,reference)
    assert bundle['source_blocks']==reference['source_blocks']
    assert bundle['source_acquisition']['records']==reference['source_acquisition']['records']
    assert bundle['optimized']['extra']['windows']==reference['optimized']['extra']['windows']
    assert bundle['optimized']['score_curve']==reference['optimized']['score_curve']
    for a,b in zip(bundle['traces'],reference['traces']):
        assert {k:v for k,v in a.items() if k!='clone_exact'}=={k:v for k,v in b.items() if k!='clone_exact'}
    for key in ('representative_path','query_path'):
        assert np.array_equal(np.load(ROOT/bundle[key],allow_pickle=False),np.load(ROOT/reference[key],allow_pickle=False)),key


def compact(details):
    return dict(input=details['input'],layers=[{k:v for k,v in details['layers'][i].items() if k not in ('query_vector','similarities')} for i in range(len(details['layers']))])


def selection_batch(j,cache,ctx,memory,r0,row,out,smoke):
    before=j.forward_calls;vision=j.vision_calls
    result={};blocks=r0['source_blocks'];reference=r0['traces']
    reps=np.load(ROOT/r0['representative_path'],allow_pickle=False)
    shape=np.load(ROOT/r0['query_path'],allow_pickle=False).shape
    for arm in ARMS:
        first=j.forward_calls;start=clock(j);traces=[];values=[];queries=np.zeros(shape,np.float32);clones=0;diagnostic=0.
        for w in reference:
            local=w['local_ids'];bounds=w['bounds'];question=w['question'];branch=None;reads=[]
            if local:
                selector=Selection(arm,blocks,local,bounds,w['branch']['layers'])
                z,details=visual_margin(j,cache,ctx,memory,local,question,selection=selector)
                assert selector.visited==list(range(shape[1]))
                branch=compact(details)
                for layer in range(shape[1]):
                    queries[w['i'],layer]=details['layers'][layer]['query_vector'].numpy()
                    reads.append(exposure(blocks,local,w['branch']['layers'][layer]['remote_ids'],details['layers'][layer]['remote_ids']))
            else:
                z=margin(j,cache,ctx,question);assert z==w['native_visual']
            trace={k:v for k,v in w.items() if k not in ('new_visual','branch','clone_exact')}
            trace.update(new_visual=z,branch=branch,exposure=reads)
            if smoke and clones==0:
                tick=clock(j);clone=copy.deepcopy(cache)
                if local:
                    other=Selection(arm,blocks,local,bounds,w['branch']['layers'])
                    replay,detail=visual_margin(j,clone,ctx,memory,local,question,selection=other)
                    assert compact(detail)==branch
                else:replay=margin(j,clone,ctx,question)
                assert replay==z and clone.get_seq_length()==cache.get_seq_length()
                assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,clone.layers))
                trace['clone_exact']=True;clones+=1;diagnostic+=clock(j)-tick;del clone
            values.append(z);traces.append(trace)
        elapsed=clock(j)-start;new_visual=elapsed-diagnostic
        path=out/'proof'/row['dataset']/row['video_id']/(arm+'_queries.npy')
        np.save(path,queries,allow_pickle=False)
        # Standalone budget includes native prefix/S and entire original source
        # acquisition even though this experimental batch shares those costs.
        times=r0['checks']['times'];seconds=r0['checks']['source_seconds']+times['prefix']+times['native_speech']+new_visual
        pred=prediction(row,ctx,values,[w['native_speech'] for w in reference],seconds,'m1_rekv_'+arm,len(blocks))
        pred['code_path']='experiments/20261006_m1_rekv/controls.py'
        result[arm]=dict(arm=arm,prediction=pred,traces=traces,query_path=str(path.relative_to(ROOT)),
            actual_forwards=j.forward_calls-first,actual_vision=j.vision_calls-vision,diagnostic_forwards=clones,
            new_visual_seconds=new_visual,diagnostic_seconds=diagnostic,elapsed_seconds=elapsed)
        assert result[arm]['actual_forwards']==len(reference)+clones and result[arm]['actual_vision']==0
        validate_selection(r0,result[arm],smoke)
    assert j.forward_calls-before==sum(a['actual_forwards'] for a in result.values()) and j.vision_calls==vision
    return result


def validate_selection(r0,arm,smoke):
    name=arm['arm'];assert name in ARMS
    blocks=r0['source_blocks'];ctx=r0['native_ctx'];rep=np.load(ROOT/r0['representative_path'],allow_pickle=False)
    queries=np.load(ROOT/arm['query_path'],allow_pickle=False);original=np.load(ROOT/r0['query_path'],allow_pickle=False)
    assert queries.shape==original.shape and queries.dtype==np.float32 and np.isfinite(queries).all()
    pred=arm['prediction'];reference=r0['traces'];indices=[b['source_index'] for b in blocks]
    assert len(arm['traces'])==len(reference)==len(pred['extra']['windows'])
    for t,w,pw in zip(arm['traces'],reference,pred['extra']['windows']):
        assert all(t[k]==w[k] for k in ('i','bounds','body','question','local_ids','native_visual','native_speech'))
        assert pw['z_visual']==t['new_visual'] and pw.get('z_speech')==w['native_speech']
        assert pw['z']==max(t['new_visual'],w['native_speech']) if w['native_speech'] is not None else pw['z']==t['new_visual']
        if not w['local_ids']:
            assert t['branch'] is None and t['new_visual']==w['native_visual'] and not queries[w['i']].any() and not t['exposure'];continue
        assert t['branch']['input']==w['branch']['input']
        assert len(t['branch']['layers'])==len(t['exposure'])==queries.shape[1]
        selector=Selection(name,blocks,w['local_ids'],w['bounds'],w['branch']['layers'])
        for layer,rr in enumerate(t['branch']['layers']):
            default,scores=remote_selection(torch.from_numpy(queries[w['i'],layer]),torch.from_numpy(rep[layer]),indices,w['local_ids'])
            remote=selector.choose(layer,default,scores);selected=sorted(set(w['local_ids']+remote),key=lambda i:indices[i])
            assert rr['local_ids']==w['local_ids'] and rr['remote_ids']==remote and rr['selected_ids']==selected
            assert t['exposure'][layer]==exposure(blocks,w['local_ids'],w['branch']['layers'][layer]['remote_ids'],remote)
            old=[torch.tensor(blocks[i]['positions'])[:,None] for i in selected];packed,end=pack_positions(old,ctx['stance_cache_logical_start'])
            assert rr['source_translations']==[int(a.min())-int(b.min()) for a,b in zip(packed,old)]
            assert rr['question_logical_start']==end and rr['prefix_tokens']==ctx['stance_cache_tokens']
            assert rr['source_tokens']==sum(blocks[i]['shape'][3] for i in selected)
            assert rr['suffix_tokens']==len(t['branch']['input']['suffix_ids']) and rr['source_times']==[blocks[i]['actual_time'] for i in selected]
        if name=='R0':
            assert t['branch']==w['branch'] and t['new_visual']==w['new_visual'] and np.array_equal(queries[w['i']],original[w['i']])
    assert pred['extra']['z_video']==r0['base']['extra']['z_video'] and pred['extra']['stance']==ctx['stance']
    ww=pred['extra']['windows'];duration=pred['duration']
    assert all(pred[k]==r0['base'][k] for k in ('dataset','video_id','duration','native_rate','error','seed'))
    idx=np.clip(((np.arange(int(np.ceil(duration*4)))+.5)/4//SPEC['window_seconds']).astype(int),0,len(ww)-1)
    assert np.array_equal(pred['score_curve'],np.asarray([w['z'] for w in ww])[idx]) and np.isfinite(pred['score_curve']).all()
    assert sum(t.get('clone_exact',False) for t in arm['traces'])==arm['diagnostic_forwards']==int(smoke)
    assert arm['actual_forwards']==len(reference)+int(smoke) and arm['actual_vision']==0
    assert min(arm[k] for k in ('new_visual_seconds','diagnostic_seconds','elapsed_seconds'))>=0
    assert abs(arm['elapsed_seconds']-arm['new_visual_seconds']-arm['diagnostic_seconds'])<1e-6
    times=r0['checks']['times']
    assert abs(pred['extra']['standalone_seconds']-r0['checks']['source_seconds']-times['prefix']-times['native_speech']-arm['new_visual_seconds'])<1e-6
    assert pred['calls']==r0['optimized']['calls']


class FrozenSelection:
    def __init__(self,layers):self.layers=layers;self.visited=[]
    def choose(self,layer,default,scores):
        assert layer==len(self.visited);self.visited.append(layer)
        return list(self.layers[layer]['remote_ids'])


@cache
def timestamp_tokenizer():
    from src.mllm_renderer import cpu_renderer
    return cpu_renderer().tok


def time_mapping(reference):
    blocks=reference['source_blocks'];times=[b['actual_time'] for b in blocks];indices=[b['source_index'] for b in blocks]
    # Raw suffix token budget includes actual chat syntax. Expanded processor
    # budget is separately asserted against original real block shapes below.
    tok=timestamp_tokenizer()
    lengths=[len(tok(b['input_evidence']['suffix_text'],add_special_tokens=False)['input_ids']) for b in blocks]
    donor,groups=permutation(times,lengths,indices)
    return dict(arm='T0',timestamp_donors=donor,presented_times=[times[d] for d in donor],groups=groups,
        reference=reference)


def validate_variant(j,row,segments,meta,bundle,smoke,mode,reference):
    equal_native(bundle,reference)
    intervention=time_mapping(reference) if mode=='time' else dict(arm='H0')
    validate_bundle(j,row,segments,meta,bundle,smoke,control=intervention)
    assert len(bundle['source_blocks'])==len(reference['source_blocks'])
    for old,new in zip(reference['source_blocks'],bundle['source_blocks']):
        for key in ('actual_time','source_index','image_rows','shape','encoding','storage_bytes'):
            assert old[key]==new[key],'actual pixel/token budget or source ownership changed'
        assert old['input_evidence']['image_grid']==new['input_evidence']['image_grid']
        assert old['input_evidence']['relative_positions']==new['input_evidence']['relative_positions']
        if mode=='history':assert old['input_evidence']==new['input_evidence'] and not new['ancestors']
    for w,t in zip(reference['traces'],bundle['traces']):
        assert all(w[k]==t[k] for k in ('i','bounds','body','question','local_ids','native_visual','native_speech'))
        if mode=='time' and w['branch']:
            assert t['branch']['input']==w['branch']['input']
            for a,b in zip(w['branch']['layers'],t['branch']['layers']):
                for key in ('remote_ids','selected_ids','source_tokens','source_translations','question_logical_start'):
                    assert a[key]==b[key],'T0 must freeze actual source IDs and packing'
    actual=[b['actual_time'] for b in bundle['source_blocks']]
    if mode=='time':
        changed=sum(int(a//8)!=int(b//8) for a,b in zip(actual,intervention['presented_times']))
        exposures=[]
        for trace in bundle['traces']:
            a,b=trace['bounds']
            if trace['branch']:
                for layer in trace['branch']['layers']:
                    ids=layer['selected_ids'];exposures.append(dict(reads=len(ids),role_changes=sum((a<=actual[i]<b)!=(a<=intervention['presented_times'][i]<b) for i in ids)))
        expected=dict(arm='T0',timestamp_donors=intervention['timestamp_donors'],presented_times=intervention['presented_times'],
            groups=intervention['groups'],cross_window_frames=changed,layer_exposures=exposures)
    else:expected=dict(arm='H0',source_previous_frames=0,native_global_context_kept=True)
    assert bundle['intervention']==expected


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--mode',choices=MODES,required=True);args=ap.parse_args()
    stem='controls_'+('smoke' if args.smoke else 'main')+'_'+args.mode
    out=ROOT/'runs/20261006_m1_rekv'/stem;out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers,av
    config=dict(version=VERSION,host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),mode=args.mode,smoke=args.smoke,
        model=MODEL,spec=SPEC,GT_read=False,implementation=IMPLEMENTATION,torch=torch.__version__,transformers=transformers.__version__,av=av.__version__,
        source='experiments/20261006_m1_rekv/{controls,measure,reader,attention,validate,control_selection,control_time}.py;2026-10-06',command='python -u '+' '.join(sys.argv))
    cfg=out/'config.json'
    attempt=1
    while (out/f'pipeline_attempt_{attempt:04d}.json').exists():attempt+=1
    audit_path=out/f'pipeline_attempt_{attempt:04d}.json';audit=dict(completed=False,GT_read=False,host=socket.gethostname(),videos=[])
    started=time.perf_counter();hooks=[];j=None
    reference_root=ROOT/'runs/20261006_m1_rekv/r1_full_main'
    names=ARMS if args.mode=='selection' else ('T0',) if args.mode=='time' else ('H0',)
    predictions={name:[] for name in names}
    try:
        if cfg.exists():
            old=json.loads(cfg.read_text());assert all(old[k]==config[k] for k in config if k not in ('date','command'))
        else:cfg.write_text(json.dumps(config,indent=2)+'\n')
        asr={d:load_asr(d) for d in DATASETS}
        tick=time.perf_counter();torch.manual_seed(0);j=Judge(MODEL);audit['model_load_seconds']=time.perf_counter()-tick
        j.forward_calls=j.vision_calls=0
        hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
        rows=selected_rows(args.smoke)
        for ordinal,row in enumerate(rows,1):
            tick=time.perf_counter();reference=record(reference_root,row);segments=asr[row['dataset']].get(row['video_id'],[])
            meta=json.loads((CACHE/row['dataset']/row['video_id']/'metadata.json').read_text());validate_input(meta,row)
            path=out/'records'/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True);memory=None
            first=j.forward_calls;vision=j.vision_calls;reused=path.exists()
            try:
                if reused:bundle=json.loads(path.read_text())
                elif args.mode=='selection':
                    callback=lambda j,cache,ctx,memory,r0:selection_batch(j,cache,ctx,memory,r0,row,out,args.smoke)
                    bundle,memory=read_video(j,row,segments,meta,out,args.smoke,after_read=callback)
                elif args.mode=='time':
                    mapping=time_mapping(reference)
                    transform=lambda frames:[{**f,'presented_time':t} for f,t in zip(frames,mapping['presented_times'])]
                    factory=lambda w,memory,local:FrozenSelection(reference['traces'][w['i']]['branch']['layers'])
                    bundle,memory=read_video(j,row,segments,meta,out,args.smoke,source_transform=transform,selector_factory=factory)
                else:bundle,memory=read_video(j,row,segments,meta,out,args.smoke,previous_frames=0)
                if args.mode=='selection':
                    validate_bundle(j,row,segments,meta,bundle,args.smoke);equal_r0(bundle,reference)
                    assert set(bundle['controls'])==set(ARMS)
                    for arm in bundle['controls'].values():validate_selection(bundle,arm,args.smoke)
                else:
                    if not reused:
                        if args.mode=='time':
                            actual=[b['actual_time'] for b in bundle['source_blocks']];reads=[]
                            for trace in bundle['traces']:
                                a,b=trace['bounds']
                                if trace['branch']:
                                    for layer in trace['branch']['layers']:
                                        ids=layer['selected_ids'];reads.append(dict(reads=len(ids),role_changes=sum((a<=actual[i]<b)!=(a<=mapping['presented_times'][i]<b) for i in ids)))
                            bundle['intervention']=dict(arm='T0',timestamp_donors=mapping['timestamp_donors'],presented_times=mapping['presented_times'],groups=mapping['groups'],
                                cross_window_frames=sum(int(a//8)!=int(b//8) for a,b in zip(actual,mapping['presented_times'])),layer_exposures=reads)
                        else:bundle['intervention']=dict(arm='H0',source_previous_frames=0,native_global_context_kept=True)
                        bundle['optimized']['method']='m1_rekv_'+names[0];bundle['optimized']['code_path']='experiments/20261006_m1_rekv/controls.py'
                    validate_variant(j,row,segments,meta,bundle,args.smoke,args.mode,reference)
                if not reused:
                    bundle['batch_actual_forwards']=j.forward_calls-first;bundle['batch_actual_vision']=j.vision_calls-vision
                    bundle['batch_peak_GiB']=torch.cuda.max_memory_allocated()/2**30 if j.device.type=='cuda' else 0.
                    expected=bundle['checks']['actual_forwards']+sum(a['actual_forwards'] for a in bundle.get('controls',{}).values())
                    assert bundle['batch_actual_forwards']==expected and bundle['batch_actual_vision']==bundle['checks']['actual_vision']
                    partial=path.with_suffix('.partial');partial.write_text(json.dumps(bundle)+'\n');partial.replace(path)
            finally:
                if memory is not None:memory.close(release=path.exists())
            assert bundle['batch_actual_forwards']==bundle['checks']['actual_forwards']+sum(a['actual_forwards'] for a in bundle.get('controls',{}).values())
            assert bundle['batch_actual_vision']==bundle['checks']['actual_vision']
            for name in names:predictions[name].append(bundle['controls'][name]['prediction'] if args.mode=='selection' else bundle['optimized'])
            audit['videos'].append(dict(dataset=row['dataset'],video_id=row['video_id'],reused=reused,elapsed_seconds=time.perf_counter()-tick,
                current_actual_forwards=j.forward_calls-first,current_actual_vision=j.vision_calls-vision))
            audit['elapsed_seconds']=time.perf_counter()-started;audit_path.write_text(json.dumps(audit,indent=2)+'\n')
            logging.info('%d/%d %s/%s mode=%s',ordinal,len(rows),row['dataset'],row['video_id'],args.mode)
        for name,rr in predictions.items():
            folder=out/name;folder.mkdir(exist_ok=True);(folder/'config.json').write_text(json.dumps({**config,'arm':name},indent=2)+'\n')
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
        audit['completed']=True;logging.info('DONE coverage=%d mode=%s',len(rows),args.mode)
    finally:
        for h in hooks:h.remove()
        audit['total_actual_forwards']=j.forward_calls if j is not None else 0
        audit['total_actual_vision']=j.vision_calls if j is not None else 0
        audit['elapsed_seconds']=time.perf_counter()-started;audit_path.write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
