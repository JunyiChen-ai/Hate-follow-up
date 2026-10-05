"""Full333 actual-video/header, JPEG/ASR and native-prefix geometry checks."""
import json
import os
import socket
import torch
from inputs import ROOT,selected_rows,DATASETS,SPEC
from src.actual_video_frames import resolve_video
from src.video_inputs import frame_paths,load_asr
from src.mllm_renderer import cpu_position_renderer
from src.stance_cache import positions


def main():
    import av
    torch.set_num_threads(4);print('host',socket.gethostname(),flush=True)
    out=ROOT/'runs/20261006_m1_ordered_slots/full_input_preflight';out.mkdir(parents=True,exist_ok=True);(out/'run.pid').write_text(str(os.getpid()))
    (out/'config.json').write_text(json.dumps(dict(host=socket.gethostname(),GT_read=False,spec=SPEC,code='experiments/20261006_m1_ordered_slots/preflight.py;2026-10-06',command='python -u '+' '.join(__import__('sys').argv)),indent=2)+'\n')
    j=cpu_position_renderer();asr={ds:load_asr(ds) for ds in DATASETS};result=[]
    for i,row in enumerate(selected_rows(),1):
        video=resolve_video(row)
        with av.open(str(video)) as container:
            stream=container.streams.video[0];assert stream.width>0 and stream.height>0
            frame=next(container.decode(stream));assert frame.pts is not None
            raw=dict(path=str(video),width=frame.width,height=frame.height,first_pts=int(frame.pts),time_base=[frame.time_base.numerator,frame.time_base.denominator])
        frames=frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']);segments=asr[row['dataset']].get(row['video_id'],[])
        msgs,files=j.prefix_messages(frames,segments);text,enc=j.encode_prefix(msgs,files);p,delta=positions(j,enc['input_ids'],enc['image_grid_thw'])
        assert 18<=len(frames)<=20 and len(enc['image_grid_thw'])==len(frames)
        assert p.shape==(3,1,enc['input_ids'].shape[1]) and sum(j.img_tokens)==int((enc['input_ids']==j.image_token_id).sum())
        result.append(dict(dataset=row['dataset'],video_id=row['video_id'],raw=raw,native_frames=len(frames),prefix_tokens=enc['input_ids'].shape[1],grid=enc['image_grid_thw'].tolist(),logical_offset=int(p.max())+1))
        if i%25==0:print(i,'/333',flush=True)
    (out/'summary.json').write_text(json.dumps(dict(host=socket.gethostname(),GT_read=False,coverage=len(result),PASS=True,rows=result),indent=2)+'\n');print('PREFLIGHT_PASS',flush=True)


if __name__=='__main__':main()
