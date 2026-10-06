"""Current media/prompt/position/proof replay; no annotations or metric logic."""
import math
import numpy as np
import torch
from inputs import ROOT,SPEC,source_frames,windows_for
from source_frames import owned_source_ids
from reader import encode_source,encode_question
from retrieval import remote_selection
from layout import pack_positions
from src.mllm_judge import yesno_question
from binding import validate_native_binding


def validate_bundle(j,row,segments,meta,bundle,smoke,control=None):
    assert bundle['version']==SPEC['version'] and bundle['spec']==SPEC
    assert bundle['segments']==[list(s) for s in segments]
    base,new=bundle['base'],bundle['optimized'];ctx=bundle['native_ctx'];checks=bundle['checks']
    assert checks['GT_read'] is False
    for key in ('z_video','stance','prefix_tokens','stance_cache_tokens','stance_cache_logical_start'):
        assert base['extra'][key]==new['extra'][key]
    assert ctx['global_margin']==base['extra']['z_video'] and ctx['stance']==base['extra']['stance']
    assert ctx['stance']==('Yes' if ctx['global_margin']>0 else 'No')
    validate_native_binding(j,row,segments,ctx,bundle['binding'])
    frames=source_frames(row,meta['source']);blocks=bundle['source_blocks']
    previous_frames=SPEC['source_previous_frames']
    if control is not None:
        assert control['arm'] in ('T0','H0')
        if control['arm']=='H0':previous_frames=0
        else:
            assert len(control['presented_times'])==len(frames)
            frames=[{**f,'presented_time':t} for f,t in zip(frames,control['presented_times'])]
    assert len(blocks)==len(frames)==checks['source_frames']
    representatives=np.load(ROOT/bundle['representative_path'],allow_pickle=False)
    queries=np.load(ROOT/bundle['query_path'],allow_pickle=False)
    config=j.model.model.config.text_config
    layers=config.num_hidden_layers;dimension=config.num_key_value_heads*config.head_dim
    assert representatives.dtype==queries.dtype==np.float32
    assert representatives.shape==(layers,len(frames),dimension)
    windows=windows_for(row,segments)
    assert queries.shape==(len(windows),layers,dimension)
    assert np.isfinite(representatives).all() and np.isfinite(queries).all()
    positions=[]
    for index,(frame,block,record) in enumerate(zip(frames,blocks,bundle['source_acquisition']['records'])):
        assert (block['block_id'],block['source_index'],block['actual_time'])==(index,frame['index'],frame['time'])
        history=list(range(max(0,index-previous_frames),index))
        assert block['direct_ancestors']==record['history_ids']==history
        inherited=set(history)
        for ancestor in history:inherited.update(blocks[ancestor]['ancestors'])
        assert block['ancestors']==sorted(inherited)
        _,relative,image_rows,evidence=encode_source(j,ctx,frame)
        assert block['input_evidence']==evidence and block['image_rows']==image_rows
        _,end=pack_positions([positions[ancestor] for ancestor in history],ctx['stance_cache_logical_start'])
        actual=relative.to('cpu')+end
        assert actual[:,0].tolist()==block['positions'] and record['logical_start']==end
        assert record['suffix_tokens']==len(evidence['suffix_ids'])
        assert record['physical_start']==ctx['stance_cache_tokens']+sum(blocks[ancestor]['shape'][3] for ancestor in history)
        assert block['shape']==[2,layers,config.num_key_value_heads,len(evidence['suffix_ids']),config.head_dim]
        encoding='bfloat16_bits' if j.dtype==torch.bfloat16 else 'float32'
        assert block['encoding']==encoding
        assert block['storage_bytes']==math.prod(block['shape'])*(2 if encoding=='bfloat16_bits' else 4)
        positions.append(actual)
    source_indices=[frame['index'] for frame in frames]
    assert len(bundle['traces'])==len(windows)==len(base['extra']['windows'])==len(new['extra']['windows'])
    for w,trace,bw,nw in zip(windows,bundle['traces'],base['extra']['windows'],new['extra']['windows']):
        local=owned_source_ids(meta['source'],(w['start'],w['end']))
        assert (trace['i'],trace['bounds'],trace['body'],trace['local_ids'])==(w['i'],[w['start'],w['end']],w['body'],local)
        question=yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')
        assert trace['question']==question and trace['native_visual']==bw['z_visual'] and trace['new_visual']==nw['z_visual']
        assert trace['native_speech']==bw.get('z_speech')==nw.get('z_speech')
        if not local:
            assert trace['branch'] is None and bw['z_visual']==nw['z_visual'] and not queries[w['i']].any()
            continue
        ids,rows,evidence=encode_question(j,ctx,question)
        assert trace['branch']['input']==evidence and len(trace['branch']['layers'])==layers
        for layer,record in enumerate(trace['branch']['layers']):
            remote,_=remote_selection(torch.from_numpy(queries[w['i'],layer]),torch.from_numpy(representatives[layer]),source_indices,local)
            if control is not None and control['arm']=='T0':
                remote=list(control['reference']['traces'][w['i']]['branch']['layers'][layer]['remote_ids'])
            selected=sorted(set(local+remote),key=lambda index:source_indices[index])
            assert record['local_ids']==local and record['remote_ids']==remote and record['selected_ids']==selected
            packed,end=pack_positions([positions[index] for index in selected],ctx['stance_cache_logical_start'])
            assert record['source_translations']==[int(new.min())-int(old.min()) for old,new in zip([positions[index] for index in selected],packed)]
            assert record['question_logical_start']==end and record['source_times']==[frames[index]['time'] for index in selected]
            assert record['prefix_tokens']==ctx['stance_cache_tokens'] and record['suffix_tokens']==len(ids)
            assert record['source_tokens']==sum(blocks[index]['shape'][3] for index in selected)
    for pred in (base,new):
        assert (pred['dataset'],pred['video_id'],pred['duration'],pred['native_rate'],pred['error'])==(row['dataset'],row['video_id'],float(row['duration']),4,None)
        ww=pred['extra']['windows']
        assert all(w['z']==max(w['z_visual'],w.get('z_speech',-math.inf)) for w in ww)
        index=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//SPEC['window_seconds']).astype(int),0,len(ww)-1)
        assert np.array_equal(pred['score_curve'],np.asarray([w['z'] for w in ww])[index]) and np.isfinite(pred['score_curve']).all()
    clone_count=sum(trace.get('clone_exact',False) for trace in bundle['traces'])
    assert clone_count==int(smoke)==checks['diagnostic_forwards']
    assert base['calls']==3+len(windows)+sum(bool(w['body'].strip()) for w in windows)
    assert new['calls']==base['calls']+len(frames)
    assert checks['actual_forwards']==base['calls']+len(frames)+len(windows)+clone_count
    assert checks['actual_vision']==1+len(frames) and checks['source_forwards']==checks['source_vision']==len(frames)
    assert checks['temporary_dense_bytes']==sum(block['storage_bytes'] for block in blocks)
    assert checks['missing_local_windows']==sum(not trace['local_ids'] for trace in bundle['traces'])
    assert checks['remote_layer_reads']==sum(bool(layer['remote_ids']) for trace in bundle['traces'] if trace['branch'] for layer in trace['branch']['layers'])
    assert checks['decode_seconds']==meta['source']['decode_seconds']
    times=checks['times']
    assert all(np.isfinite(value) and value>=0 for value in times.values())
    assert checks['native_binding_seconds']>=0
    assert abs(checks['source_seconds']-times['source']-checks['decode_seconds']-checks['native_binding_seconds'])<1e-6
    assert abs(base['extra']['standalone_seconds']-sum(times[key] for key in ('prefix','native_visual','native_speech')))<1e-6
    assert abs(new['extra']['standalone_seconds']-checks['source_seconds']-sum(times[key] for key in ('prefix','new_visual','native_speech')))<1e-6
