"""Current whole-source binding plus real 36-layer causal query-mask checks."""
import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import torch
from handle_measure import ROOT,question,usable_record,visual_frame_sources
from handle_inputs import CACHE,selected_rows
from src.source_key_mask import binding,margin as bound_margin,observe_prefix
from src.mllm_judge import Judge,yesno_question
from src.mllm_renderer import cpu_renderer
from src.stance_cache import build,margin
from src.video_inputs import frame_paths


def sources():
    j=cpu_renderer();totals={}
    for row in selected_rows(False):
        ds,vid=row['dataset'],row['video_id'];m=json.loads((CACHE/ds/(vid+'.json')).read_text());unchanged=copy.deepcopy(m)
        old={r:json.loads((ROOT/f'runs/20261005_m1_program/r{r}_handles_full_main/details'/ds/(vid+'.json')).read_text()) for r in (1,2,3)}
        conv=old[3]['native_conversation'];n=len(m['windows'])
        frames=frame_paths(ds,vid,20);segments=old[3]['segments']
        msgs,files=j.prefix_messages(frames,segments);text,enc=j.encode_prefix(msgs,files)
        input_ids=enc['input_ids'][0].tolist();counts=list(j.img_tokens)
        t=totals.setdefault(ds,dict(videos=0,windows=0,previous_query_tokens_exact=0,visual_applied=0,visual_fallback=0,speech_applied=0,speech_fallback=0,source_frames=0))
        t['videos']+=1
        for i,w in enumerate(m['windows']):
            trace=old[3]['traces'][i];a,b=trace['start'],trace['end'];t['windows']+=1
            for kind in ('visual','speech'):
                if not trace['branches'][kind]['available']:continue
                args=(w['execution'],i,n,a,b,trace['native_body'],kind)
                for revision in (1,2,3):
                    q,rec=question(*args,revision);prior=old[revision]['traces'][i]['branches'][kind]
                    assert q==prior['question'] and rec==prior['record']
                    ids,suffix=j.branch_ids(conv['msgs'],q,conv['history'],head_text=conv['head'])
                    assert ids==prior['suffix_ids'] and suffix==prior['suffix_text'];t['previous_query_tokens_exact']+=1
                q,rec=question(*args,4);active=usable_record(rec,kind,4)
                t[kind+('_applied' if active else '_fallback')]+=1
                if kind=='visual':
                    assert q==yesno_question(i,n,a,b,trace['native_body'],kind)
                    ids=visual_frame_sources(rec)
                    assert active==bool(ids)
                    for f in ids:assert a<=frames[f][0]<b
                    source=binding(input_ids,j.image_token_id,counts,ids)
                    assert set(source['kept_visual_keys']).isdisjoint(source['dropped_visual_keys'])
                    assert sorted(source['kept_visual_keys']+source['dropped_visual_keys'])==[k for k,tok in enumerate(input_ids) if tok==j.image_token_id]
                    t['source_frames']+=len(ids)
                else:assert (q,rec)==question(*args,3)
        assert m==unchanged
        if sum(t['videos'] for t in totals.values())%25==0:print(sum(t['videos'] for t in totals.values()),'/333',flush=True)
    assert sum(t['videos'] for t in totals.values())==333
    assert all(t['visual_applied'] and t['visual_fallback'] for t in totals.values())
    return dict(whole333_source_immutable=True,previousR1_R2_R3_tokens_exact=True,datasets=totals)


def fixture(model,frames,dtype):
    head=torch.nn.Linear(model.config.text_config.hidden_size,256,bias=False).to(dtype)
    j=Judge.__new__(Judge);j.model=SimpleNamespace(model=model,config=model.config,get_output_embeddings=lambda:head)
    j.device=torch.device('cpu');j.dtype=dtype;j.image_token_id=127;j.forward_params=set(inspect.signature(model.forward).parameters)
    ff=[(float(i),Path(str(i)+'.jpg')) for i in range(frames)]
    pixels=torch.randn(frames*16,24).to(dtype);grids=torch.tensor([[1,4,4]]*frames)
    ids=torch.tensor([[10]+[t for _ in ff for t in ([125]+[127]*4+[126])]+[50,51,52]])
    def messages(frames,segments):return frames,[p for _,p in frames]
    def encode(msgs,files):
        j.img_tokens=[4]*frames;j._prefix_text='fixtureprefix'
        enc=dict(input_ids=ids,pixel_values=pixels,image_grid_thw=grids,attention_mask=torch.ones_like(ids))
        if 'mm_token_type_ids' in j.forward_params:enc['mm_token_type_ids']=(ids==127).long()
        return j._prefix_text,enc
    j.prefix_messages=messages;j.encode_prefix=encode
    j.branch_ids=lambda *a,**k:([60,61],'Q');j.answer_ids=lambda *a,**k:([62],'A');j.turn=lambda role,text:dict(role=role,content=text)
    j.yes_ids=[70];j.no_ids=[71];j.softcap=None
    return j,ff


def model_checks():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=36,
        num_attention_heads=32,num_key_value_heads=8,head_dim=128,max_position_embeddings=512,
        rope_scaling={'rope_type':'default','mrope_section':[24,20,20]}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,
            temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1]),
        image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
    result=[]
    with torch.no_grad():
        for dtype in (torch.float32,torch.bfloat16):
            for frames in (18,20):
                model=Qwen3VLModel(cfg).eval().to(dtype);j,ff=fixture(model,frames,dtype)
                with observe_prefix(j) as prefix:cache,ctx=build(j,ff,[])
                initial=copy.deepcopy(cache);q='actual fixture visual question';native=margin(j,cache,ctx,q)
                dense=binding(prefix['input_ids'],127,prefix['image_counts'],list(range(frames)))
                dense_z=bound_margin(j,cache,ctx,q,dense);assert native==dense_z
                source=binding(prefix['input_ids'],127,prefix['image_counts'],[frames//2])
                seen=[]
                def hook(module,args,kwargs):seen.append(kwargs['attention_mask'].detach().clone())
                handles=[l.self_attn.register_forward_pre_hook(hook,with_kwargs=True) for l in model.language_model.layers]
                z=bound_margin(j,cache,ctx,q,source)
                for h in handles:h.remove()
                n=ctx['stance_cache_tokens'];expected=torch.tensor([[[[k<=n+i and k not in source['dropped_visual_keys'] for k in range(n+2)] for i in range(2)]]])
                assert len(seen)==36 and all(torch.equal(m,expected) for m in seen)
                assert all(bool(seen[0][0,0,:,k].all()) for k,t in enumerate(prefix['input_ids']) if t!=127)
                clone=copy.deepcopy(cache);clone_z=bound_margin(j,clone,ctx,q,source);assert clone_z==z
                assert z!=native,'actual visual-source mask did not change margin'
                assert torch.equal(model.rope_deltas,ctx['rope'])
                assert cache.get_seq_length()==clone.get_seq_length()==initial.get_seq_length()
                assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(initial.layers,cache.layers))
                result.append(dict(dtype=str(dtype),frames=frames,layers=36,native_dense_margin_exact=True,all36_causal_source_masks_exact=True,allnonvisual_keys_preserved=True,source_margin_changed=True,clone_margin_exact=True,allKV_restored_exact=True))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('sources','model'),required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(0)
    result=sources() if a.stage=='sources' else model_checks()
    out=ROOT/'runs/20261005_m1_program/bound_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    (out/(a.stage+'_summary.json')).write_text(json.dumps(dict(host=socket.gethostname(),GT_read=False,PASS=True,stage=a.stage,result=result),indent=2)+'\n')
    print(json.dumps(result,indent=2));print('BOUND_CPU_PASS '+a.stage,flush=True)


if __name__=='__main__':main()
