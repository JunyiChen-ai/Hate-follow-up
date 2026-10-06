"""Current native/media/feature/group/layout/query proof replay; no annotations."""
import math
import numpy as np
import torch
from inputs import ROOT,SPEC,frames_for,windows_for,local_ids
from memory import read_tensor
from source_encoding import encode
from reader import encode_local_input
from ctr import group,aggregate
from src.pre_rotary_memory import translate,select
from src.native_input_binding import validate as validate_native
from src.mllm_judge import yesno_question


def validate_bundle(j,row,segments,bundle,smoke):
    # Generation explicitly uses four CPU threads. Match its cdist reduction
    # backend before exact grouping replay; preserve every equality check.
    torch.set_num_threads(4)
    from measure import IMPLEMENTATION
    assert bundle['version']==SPEC['version'] and bundle['spec']==SPEC and bundle['segments']==[list(s) for s in segments]
    ctx=bundle['native_ctx'];checks=bundle['checks'];base=bundle['base'];new=bundle['optimized']
    assert checks['GT_read'] is False
    validate_native(j,row,segments,ctx,SPEC,IMPLEMENTATION,bundle['binding'])
    source,frames=frames_for(row);assert source==bundle['source_input'] and frames==bundle['frames']
    assert len(frames)==len(bundle['source_blocks'])==len(bundle['source_acquisition']['records'])==checks['source_frames']
    cfg=j.model.model.config;layers=cfg.text_config.num_hidden_layers;heads=cfg.text_config.num_key_value_heads;dim=cfg.text_config.head_dim;width=cfg.vision_config.out_hidden_size
    reps=np.load(ROOT/bundle['representative_path'],allow_pickle=False);queries=np.load(ROOT/bundle['query_path'],allow_pickle=False)
    windows=windows_for(row,segments)
    assert reps.dtype==queries.dtype==np.float32 and reps.shape==(layers,len(frames),heads*dim) and queries.shape==(len(windows),layers,heads*dim)
    assert np.isfinite(reps).all() and np.isfinite(queries).all()
    previous=None;grid=None;positions=[]
    for i,(frame,block,record) in enumerate(zip(frames,bundle['source_blocks'],bundle['source_acquisition']['records'])):
        assert block['id']==record['block_id']==i and block['source_index']==frame['index'] and block['actual_time']==frame['time']
        encoded,input_evidence=encode(j,ctx,frame);assert input_evidence==record['input']
        f=read_tensor(record['full_projector']);ds=[read_tensor(m) for m in record['full_deepstack']]
        n=len(input_evidence['visual_rows']);assert f.shape==(n,width) and all(v.shape==f.shape for v in ds)
        plan=group(f,torch.tensor(record['saliency']),input_evidence['image_grid'][0],previous,grid);assert plan==record['plan']
        assert torch.equal(aggregate(f,plan),read_tensor(record['compressed_features']))
        assert len(ds)==len(record['compressed_deepstack'])==len(cfg.vision_config.deepstack_visual_indexes)
        assert all(torch.equal(aggregate(v,plan),read_tensor(meta)) for v,meta in zip(ds,record['compressed_deepstack']))
        roots=[r['root'] for r in plan['groups']];original=encoded['input_ids'][0];visual=(original==j.image_token_id).nonzero().flatten()
        keep=original!=j.image_token_id;keep[visual[roots]]=True;indices=keep.nonzero().flatten()
        evidence=record['packing'];relative=torch.tensor(input_evidence['relative_positions'],dtype=torch.long)[:,None]
        assert evidence['original_prefix_ids']==original.tolist() and evidence['sequence_indices']==indices.tolist()
        assert evidence['packed_ids']==original[indices].tolist() and evidence['packed_positions']==relative[:,0,indices].tolist()
        assert evidence['source_indices']==roots and evidence['grid_thw']==input_evidence['image_grid']
        assert evidence['visual_source_positions_exact'] and evidence['text_and_boundaries_preserved']
        history=list(range(max(0,i-SPEC['source_previous_frames']),i))
        assert record['history_ids']==block['direct_ancestors']==history
        inherited=set(history)
        for h in history:inherited.update(bundle['source_blocks'][h]['ancestors'])
        assert block['ancestors']==sorted(inherited)
        _,end=translate([positions[h] for h in history],ctx['stance_cache_logical_start'])
        position=relative[:,:,indices]+end;assert block['positions']==position[:,0].tolist()
        assert record['logical_start']==end and record['suffix_tokens']==len(indices)
        assert record['physical_start']==ctx['stance_cache_tokens']+sum(bundle['source_blocks'][h]['shape'][-2] for h in history)
        assert block['shape']==[2,layers,heads,len(indices),dim]
        vr=(original[indices]==j.image_token_id).nonzero().flatten().tolist();text=[k for k in range(len(indices)) if k not in vr]
        assert block['visual_rows']==vr and block['text_rows']==text
        assert block['quantization']['shape']==[2,layers,heads,len(vr),dim] and len(vr)==min(50,n)
        expected_dtype=str(j.dtype);assert block['quantization']['dtype']==block['text']['dtype']==expected_dtype
        kv_bytes=2*layers*heads*math.ceil(len(vr)/2)*dim+2*layers*heads*dim*8+2*layers*heads*len(text)*dim*(2 if j.dtype==torch.bfloat16 else 4)
        assert block['storage_bytes']==record['temporary_KV_bytes']==kv_bytes
        positions.append(position);previous=f;grid=input_evidence['image_grid'][0]
    assert len(bundle['traces'])==len(windows)==len(base['extra']['windows'])==len(new['extra']['windows'])
    clones=0
    for w,t,bw,nw in zip(windows,bundle['traces'],base['extra']['windows'],new['extra']['windows']):
        local=local_ids(frames,w);question=yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')
        assert t['i']==w['i'] and t['bounds']==[w['start'],w['end']] and t['body']==w['body'] and t['LOCAL']==local and t['question']==question
        assert t['native_visual']==bw['z_visual'] and t['native_speech']==bw.get('z_speech')==nw.get('z_speech') and t['new_visual']==nw['z_visual']
        if not local:assert t['branch'] is None and t['new_visual']==t['native_visual'] and not queries[w['i']].any()
        else:
            _,relative,rows,evidence=encode_local_input(j,ctx,frames,local,question);assert evidence==t['branch']['input'] and len(t['branch']['layers'])==layers
            for l,r in enumerate(t['branch']['layers']):
                remote,_=select(torch.from_numpy(queries[w['i'],l]),torch.from_numpy(reps[l]),[f['index'] for f in frames],set(local),SPEC['remote_frames_per_layer'])
                ordered=sorted(remote,key=lambda i:frames[i]['index']);packed,end=translate([positions[i] for i in ordered],ctx['stance_cache_logical_start'])
                assert r['remote_ids']==remote and r['packed_remote_ids']==ordered and r['suffix_logical_start']==end
                assert r['source_translations']==[int(p.min())-int(positions[i].min()) for i,p in zip(ordered,packed)]
                assert r['source_tokens']==sum(bundle['source_blocks'][i]['shape'][-2] for i in ordered)
                assert r['prefix_tokens']==ctx['stance_cache_tokens'] and r['suffix_tokens']==len(evidence['suffix_ids'])
                assert r['source_times']==[frames[i]['time'] for i in ordered]
        clones+=t.get('visual_clone_exact',False)+t.get('speech_clone_exact',False)
    assert clones==checks['diagnostic_forwards']==(1+int(any(t['native_speech'] is not None for t in bundle['traces'])) if smoke else 0)
    assert checks['actual_forwards']==base['calls']+len(frames)+len(windows)+clones and checks['actual_vision']==1+len(frames)
    for pred in (base,new):
        assert pred['extra']['z_video']==ctx['global_margin'] and pred['extra']['stance']==ctx['stance']
        ww=pred['extra']['windows'];idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//8).astype(int),0,len(ww)-1)
        assert np.array_equal(pred['score_curve'],np.asarray([w['z'] for w in ww])[idx]) and np.isfinite(pred['score_curve']).all()
        assert all(w['z']==(max(w['z_visual'],w['z_speech']) if 'z_speech' in w else w['z_visual']) for w in ww)
    assert ctx['stance']==('Yes' if ctx['global_margin']>0 else 'No')
    assert all(v>=0 for v in checks['times'].values()) and checks['peak_GiB']>=0
    assert abs(checks['source_seconds']-checks['decode_seconds']-checks['times']['source']-checks['native_binding_seconds'])<1e-5
    assert abs(new['extra']['standalone_seconds']-checks['source_seconds']-sum(checks['times'][k] for k in ('prefix','native_speech','new_visual')))<1e-5
