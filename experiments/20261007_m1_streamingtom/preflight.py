"""Full actual media/native/source/expanded-LOCAL processor binding; no labels."""
import json
import os
import socket
import time
import numpy as np
import torch
from inputs import ROOT,SPEC,DATASETS,selected_rows,frames_for,windows_for,local_ids
from source_encoding import encode
from reader import encode_local_input
from src.mllm_renderer import cpu_position_renderer
from src.video_inputs import frame_paths,load_asr
from src.mllm_judge import VIDEO_QUESTION,yesno_question
from src.stance_cache import positions


def main():
    torch.set_num_threads(4);print(socket.gethostname(),flush=True)
    out=ROOT/'runs/20261007_m1_streamingtom/full_input_preflight';out.mkdir(parents=True,exist_ok=True);(out/'run.pid').write_text(str(os.getpid()))
    start=time.perf_counter();j=cpu_position_renderer();asr={ds:load_asr(ds) for ds in DATASETS};rows=[]
    for ordinal,row in enumerate(selected_rows(),1):
        segments=asr[row['dataset']].get(row['video_id'],[]);source,frames=frames_for(row)
        # Prior native stance is only a declared renderer condition here; fresh
        # GPU G/S/stance must reproduce it in the later scientific run.
        old=json.loads((ROOT/'runs/20261006_m1_rekv/r1_full_main/records'/row['dataset']/(row['video_id']+'.json')).read_text())
        ctx=old['native_ctx'];msgs,files=j.prefix_messages(frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']),segments)
        text,enc=j.encode_prefix(msgs,files);qid,qtext=j.branch_ids(msgs,VIDEO_QUESTION,head_text=text);aid,atext=j.answer_ids(msgs,VIDEO_QUESTION,ctx['stance'])
        assert ctx['msgs']==msgs and ctx['head']==text+qtext+atext
        assert old['binding']['native']==dict(prefix_ids=enc['input_ids'][0].tolist(),image_grid=enc['image_grid_thw'].tolist(),
            paths=[str(p.relative_to(ROOT)) for p in files],global_question_ids=qid,stance_ids=aid)
        assert np.array_equal(enc['pixel_values'].float().numpy(),np.load(ROOT/old['binding']['pixel_path'],allow_pickle=False))
        ids=torch.cat([enc['input_ids'],torch.tensor([qid+aid])],1);p,delta=positions(j,ids,enc['image_grid_thw'])
        assert ids.shape[1]==ctx['stance_cache_tokens'] and int(p.max())+1==ctx['stance_cache_logical_start'] and delta.tolist()==old['native_rope']
        source_counts=[];grids=[]
        for frame in frames:
            _,evidence=encode(j,ctx,frame);source_counts.append(len(evidence['visual_rows']));grids.append(evidence['image_grid'][0])
        windows=windows_for(row,segments);branches=0;missing=0;expanded=0;local_images=0
        for w in windows:
            local=local_ids(frames,w)
            if not local:missing+=1;continue
            question=yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')
            _,_,question_rows,evidence=encode_local_input(j,ctx,frames,local,question)
            assert evidence['image_counts']==[source_counts[i] for i in local] and evidence['grid']==[grids[i] for i in local]
            assert question_rows and evidence['question_end']-evidence['question_start']==len(question)
            assert all(evidence['suffix_ids'][r]!=j.image_token_id for r in question_rows)
            branches+=1;expanded+=len(evidence['suffix_ids'])!=len(evidence['raw_token_ids']);local_images+=len(local)
        result=dict(dataset=row['dataset'],video_id=row['video_id'],source_frames=len(frames),source_visual_tokens=sum(source_counts),
            retained_visual_tokens=sum(min(SPEC['visual_budget'],n) for n in source_counts),max_source_visual_tokens=max(source_counts,default=0),
            windows=len(windows),LOCAL_branch_windows=branches,missing_LOCAL=missing,LOCAL_images=local_images,image_expanded_branches=expanded,
            max_native_question_chars=max((len(t['question']) for t in old['traces']),default=0))
        rows.append(result)
        if ordinal%10==0:print('PREFLIGHT_BOUND',ordinal,333,flush=True);(out/'progress.json').write_text(json.dumps(dict(coverage=ordinal,elapsed_seconds=time.perf_counter()-start))+'\n')
    assert len(rows)==333 and sum(r['windows'] for r in rows)==7359
    assert all(sum(r['image_expanded_branches'] for r in rows if r['dataset']==ds)>0 for ds in DATASETS)
    full=sum(r['source_visual_tokens'] for r in rows);reduced=sum(r['retained_visual_tokens'] for r in rows)
    cfg=j.model.model.config;width=cfg.vision_config.out_hidden_size;levels=1+len(cfg.vision_config.deepstack_visual_indexes)
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,host=socket.gethostname(),coverage=333,rows=rows,scope='actual RAW PTS/RGB, current native inputs/positions and source/LOCAL expanded processor binding only; old stance condition for rendering, not fresh GPU outputs',
        persistent_feature_proof_estimated_bytes=(full+reduced)*width*levels*2,feature_capacity_scope='actual token counts×current width/levels/BF16; does not include native/representatives/queries/JSON/temporaryKV/I/O',elapsed_seconds=time.perf_counter()-start),indent=2)+'\n')
    print('PREFLIGHT_PASS',flush=True)


if __name__=='__main__':main()
