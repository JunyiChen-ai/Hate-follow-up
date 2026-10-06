"""Actual333 media/native/source-token binding and declared capacity estimates."""
import json
import math
import os
import socket
import torch
from PIL import Image
from inputs import ROOT,SPEC,DATASETS,selected_rows
from source_frames import decode_entries
from reader import encode_source,encode_question
from src.mllm_renderer import cpu_position_renderer
from src.video_inputs import frame_paths,load_asr
from src.mllm_judge import VIDEO_QUESTION,yesno_question
from src.stance_cache import positions


def main():
    torch.set_num_threads(4)
    out=ROOT/'runs/20261006_m1_rekv/full_input_preflight';out.mkdir(parents=True,exist_ok=True)
    print(socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    (out/'config.json').write_text(json.dumps(dict(host=socket.gethostname(),GT_read=False,spec=SPEC,
        source='experiments/20261006_m1_rekv/preflight.py;2026-10-06',scope='CPU tokenizer/processor/realraw headers; no weights/scores/GT'),indent=2)+'\n')
    renderer=cpu_position_renderer();asr={dataset:load_asr(dataset) for dataset in DATASETS};rows=[]
    for number,row in enumerate(selected_rows(),1):
        segments=asr[row['dataset']].get(row['video_id'],[])
        frames=frame_paths(row['dataset'],row['video_id'],SPEC['native_requested_frames'])
        msgs,files=renderer.prefix_messages(frames,segments);text,encoded=renderer.encode_prefix(msgs,files)
        assert 18<=len(frames)<=20
        qids,qtext=renderer.branch_ids(msgs,VIDEO_QUESTION,head_text=text)
        decoded=decode_entries(row)
        try:
            actual=next((frame,entry,origin) for frame,entry,origin,available in decoded if available)
            frame,entry,origin=actual
            path=out/'source_probe'/row['dataset']/(row['video_id']+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            image=frame.to_image().convert('RGB')
            try:image.save(path)
            finally:image.close()
        finally:decoded.close()
        source=dict(**entry,path=str(path.relative_to(ROOT)))
        sizes=[]
        for stance in ('Yes','No'):
            aids,atext=renderer.answer_ids(msgs,VIDEO_QUESTION,stance)
            allids=torch.cat([encoded['input_ids'],torch.tensor([qids+aids])],1)
            p,delta=positions(renderer,allids,encoded['image_grid_thw'])
            ctx=dict(msgs=msgs,history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},renderer.turn('assistant',stance)],
                head=text+qtext+atext,stance_cache_logical_start=int(p.max())+1,stance_cache_tokens=allids.shape[1])
            _,relative,image_rows,evidence=encode_source(renderer,ctx,source)
            question=yesno_question(0,len(__import__('src.video_inputs',fromlist=['fixed_windows']).fixed_windows(float(row['duration']),8)),0,min(8,float(row['duration'])),'','visual')
            ids,query_rows,qevidence=encode_question(renderer,ctx,question)
            assert query_rows and len(image_rows)==int((torch.tensor(evidence['suffix_ids'])==renderer.image_token_id).sum())
            sizes.append(dict(stance=stance,native_tokens=allids.shape[1],source_tokens=len(evidence['suffix_ids']),image_tokens=len(image_rows),
                source_logical_span=int(relative.max())+1,question_tokens=len(ids),question_rows=len(query_rows)))
        targets=math.ceil(float(row['duration'])*SPEC['source_fps'])
        # This estimate uses the real first-frame geometry; it is not a claim
        # that all later frames have identical grids or measured peak memory.
        dense_estimate=targets*max(size['source_tokens'] for size in sizes)*36*2*8*128*2
        rows.append(dict(dataset=row['dataset'],video_id=row['video_id'],raw_shape=[entry['height'],entry['width']],origin=origin,
            native_frames=len(frames),source_target_upper_count=targets,actual_first_source_bindings=sizes,
            dense_KV_bytes_first_grid_estimate=dense_estimate,estimate_scope='first-frame-grid, actualsource timing text; not exactallframes/peak'))
        if number%25==0:print(number,'/333',flush=True)
    result=dict(PASS=True,host=socket.gethostname(),GT_read=False,coverage=len(rows),
        worst_dense_first_grid_estimate_bytes=max(row['dense_KV_bytes_first_grid_estimate'] for row in rows),
        all_dense_first_grid_estimate_bytes=sum(row['dense_KV_bytes_first_grid_estimate'] for row in rows),rows=rows)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('PREFLIGHT_PASS',flush=True)


if __name__=='__main__':main()
