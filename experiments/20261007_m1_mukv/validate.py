"""Current source/native/layout/DCP/coherence/score proof replay; no labels."""
import math
import numpy as np
import torch
from inputs import ROOT,SPEC,frames_for,windows_for
from memory import load_tensor
from encoding import source_input,source_geometry
from reader import local_geometry
from compression import selection
from coherence import retrieve
from src.pre_rotary_memory import translate
from src.native_input_binding import validate as validate_native
from src.mllm_judge import yesno_question


def validate_bundle(j,row,segments,bundle,smoke):
    from measure import IMPLEMENTATION,serial
    torch.set_num_threads(SPEC['cpu_threads'])
    assert bundle['version']==SPEC['version'] and bundle['spec']==SPEC and bundle['segments']==[list(s) for s in segments]
    ctx=bundle['native_ctx'];base=bundle['base'];new=bundle['optimized'];c=bundle['checks'];assert c['GT_read'] is False
    validate_native(j,row,segments,ctx,SPEC,IMPLEMENTATION,bundle['binding'])
    source,window_frames=frames_for(row);assert source==bundle['source_input'] and window_frames==bundle['window_frames']
    blocks=bundle['source_blocks'];records=bundle['source_acquisition']['records'];assert len(blocks)==len(records)==c['source_blocks']==c['source_LM']
    cfg=j.model.model.config;layers=cfg.text_config.num_hidden_layers;heads=cfg.text_config.num_key_value_heads;dim=cfg.text_config.head_dim
    reps=np.load(ROOT/bundle['representative_path'],allow_pickle=False);assert reps.shape==(len(blocks),heads*dim) and reps.dtype==np.float32 and np.isfinite(reps).all()
    cached={}
    for r in bundle['source_acquisition']['vision_records']:
        frame=r['frame'];_,_,evidence=source_input(j,ctx,[frame],'frame');assert evidence==r['input']
        f=load_tensor(r['feature']);ds=[load_tensor(m) for m in r['deepstack']]
        count=sum(token==j.image_token_id for token in evidence['suffix_ids'])
        assert f.shape==(count,cfg.vision_config.out_hidden_size) and len(ds)==len(cfg.vision_config.deepstack_visual_indexes) and all(v.shape==f.shape for v in ds)
        assert frame['index'] not in cached;cached[frame['index']]=dict(feature=f,deepstack=ds,grid=r['grid'])
    assert set(cached)=={f['index'] for media in window_frames for f in media} and c['source_vision']==len(cached)
    expected=[]
    for window,frames in enumerate(window_frames):
        if not frames:continue
        middle=[frames[(len(frames)-1)//2]]
        for grain,q,media in [('segment',None,frames),('frame',None,middle)]+[('patch',q,middle) for q in range(4)]:
            _,_,evidence=source_geometry(j,ctx,media,grain,q)
            if evidence is not None:expected.append((window,grain,q,media))
    assert len(expected)==len(blocks);positions=[]
    for i,(b,r,item) in enumerate(zip(blocks,records,expected)):
        window,grain,q,media=item;assert b['id']==r['id']==i and b['window']==r['window']==window and b['grain']==r['grain']==grain and b['quadrant']==r['quadrant']==q
        assert b['source_frames']==[{k:f[k] for k in ('index','time','path')} for f in media]
        _,relative,evidence=source_geometry(j,ctx,media,grain,q);assert evidence==r['input']
        history=[a['id'] for a in blocks[:i] if a['window']<window and a['grain']==grain]
        assert history==b['direct_ancestors']==r['history'];inherited=set(history)
        for h in history:inherited.update(blocks[h]['ancestors'])
        assert b['ancestors']==sorted(inherited)
        _,end=translate([positions[h] for h in history],ctx['stance_cache_logical_start']);actual=relative+end
        assert end==r['logical_start'] and actual[:,0].tolist()==r['positions']
        vis=evidence['visual_rows'];key=load_tensor(r['last_rotated_visual_keys']);raw=load_tensor(r['last_pre_rotary_visual_keys'])
        assert key.shape==raw.shape==(heads,len(vis),dim)
        attention=torch.tensor(r['attention'],dtype=torch.float32);roots,indicator=selection(key,attention,SPEC['retention'][grain])
        assert roots==r['retained_visual_roots']
        for label in ('frequency','normalized_attention','normalized_frequency'):assert indicator[label].tolist()==r[label]
        assert indicator['score'].tolist()==r['scores']
        selected={vis[k] for k in roots};keep=[k for k in range(len(evidence['packed_ids'])) if k not in vis or k in selected]
        assert b['kept_sequence_rows']==r['retained_sequence_rows']==keep and b['positions']==actual[:,0,keep].tolist()
        retained_vis=[k for k,t in enumerate(keep) if t in vis];assert b['visual_rows']==retained_vis
        assert b['shape']==[2,layers,heads,len(keep),dim]
        expected_bytes=math.prod(b['shape'])*(2 if j.dtype==torch.bfloat16 else 4)
        assert b['storage']['bytes']==r['KV_storage_bytes']==expected_bytes and b['storage']['dtype']==str(j.dtype)
        assert np.array_equal(raw[:,roots,:].float().mean(1).flatten().numpy(),reps[i])
        positions.append(actual[:,:,keep])
    windows=windows_for(row,segments);assert len(bundle['traces'])==len(windows)==len(window_frames)==len(base['extra']['windows'])==len(new['extra']['windows'])
    clones=0;probes=0
    for w,media,t,bw,nw in zip(windows,window_frames,bundle['traces'],base['extra']['windows'],new['extra']['windows']):
        question=yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')
        assert t['i']==w['i'] and t['LOCAL']==media and t['bounds']==[w['start'],w['end']] and t['question']==question and t['body']==w['body']
        assert t['native_visual']==bw['z_visual'] and t['native_speech']==bw.get('z_speech')==nw.get('z_speech') and t['new_visual']==nw['z_visual']
        if not media:assert t['probe'] is None and t['branch'] is None and not t['selected_source_blocks'] and t['new_visual']==t['native_visual']
        else:
            probes+=1;pr=t['probe'];neutral=SPEC['neutral_query_text'].format(body=w['body']);_,_,rows,evidence=local_geometry(j,ctx,media,neutral,'Unscored neutral retrieval query. Source observations retain actual times.')
            assert pr['input']==evidence and pr['actual_LM']==1 and pr['actual_vision']==0
            query=torch.tensor(pr['query'],dtype=torch.float32);assert query.shape==(heads*dim,) and torch.isfinite(query).all()
            mapping={};grain_reps={};order={};excluded={}
            for grain in SPEC['granularities']:
                ids=[b['id'] for b in blocks if b['grain']==grain];mapping[grain]=ids
                grain_reps[grain]=torch.from_numpy(reps[ids]) if ids else torch.empty(0,len(query))
                order[grain]=[(blocks[i]['window'],-1 if blocks[i]['quadrant'] is None else blocks[i]['quadrant']) for i in ids]
                excluded[grain]={k for k,i in enumerate(ids) if blocks[i]['window']==w['i']}
            result=retrieve(query,grain_reps,order,excluded);assert pr['grain_to_block_ids']==mapping and pr['retrieval']==serial(result)
            selected=[mapping[g][i] for g in SPEC['granularities'] for i in result['selected'][g]]
            selected.sort(key=lambda i:(blocks[i]['window'],SPEC['granularities'].index(blocks[i]['grain']),-1 if blocks[i]['quadrant'] is None else blocks[i]['quadrant']))
            assert selected==pr['selected_block_ids']==t['selected_source_blocks']
            _,relative,rows,evidence=local_geometry(j,ctx,media,question,SPEC['reader_role_text']);branch=t['branch'];assert branch['input']==evidence and branch['selected_blocks']==selected
            _,end=translate([positions[i] for i in selected],ctx['stance_cache_logical_start']);assert branch['suffix_logical_start']==end and branch['source_tokens']==sum(blocks[i]['shape'][-2] for i in selected)
            assert branch['actual_LM']==1 and branch['actual_vision']==0
        clones+=t.get('visual_clone_exact',False)+t.get('speech_clone_exact',False)
    assert probes==c['probe_calls'] and clones==c['diagnostic_forwards']==(1+int(any(t['native_speech'] is not None for t in bundle['traces'])) if smoke else 0)
    assert c['actual_forwards']==base['calls']+c['source_LM']+probes+len(windows)+clones and c['actual_vision']==1+c['source_vision']
    assert c['missing_LOCAL']==sum(not f for f in window_frames) and c['selected_context_blocks']==sum(len(t['selected_source_blocks']) for t in bundle['traces'])
    for p in (base,new):
        assert p['extra']['z_video']==ctx['global_margin'] and p['extra']['stance']==ctx['stance']
        ww=p['extra']['windows'];idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//8).astype(int),0,len(ww)-1)
        assert np.array_equal(p['score_curve'],np.asarray([r['z'] for r in ww])[idx]) and np.isfinite(p['score_curve']).all()
        assert all(r['z']==(max(r['z_visual'],r['z_speech']) if 'z_speech' in r else r['z_visual']) for r in ww)
    assert ctx['stance']==('Yes' if ctx['global_margin']>0 else 'No') and min(c['times'].values())>=0
    assert abs(c['source_seconds']-c['decode_seconds']-c['times']['source']-c['native_binding_seconds'])<1e-5
    assert abs(new['extra']['standalone_seconds']-c['source_seconds']-sum(c['times'][k] for k in ('prefix','native_speech','probe','new_visual')))<1e-5
