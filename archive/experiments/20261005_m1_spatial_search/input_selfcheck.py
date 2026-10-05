"""Real processor/token seams and native mRoPE geometry for appended source images."""
import json
import os
import socket
import numpy as np
from PIL import Image
import torch
from inputs import ROOT,SPEC
from geometry import children,map_box
from src.mllm_renderer import cpu_position_renderer
from src.source_image_branch import encode_branch
from src.stance_cache import positions
from src.mllm_judge import VIDEO_QUESTION


def main():
    print('host',socket.gethostname(),flush=True);torch.set_num_threads(1)
    out=ROOT/'runs/20261005_m1_spatial_search/input_cpu_checks';out.mkdir(parents=True,exist_ok=True);(out/'run.pid').write_text(str(os.getpid()))
    rng=np.random.default_rng(0);checks=[]
    for width,height in [(5,5),(601,137),(137,601),(602,301),(301,602),(3,50)]:
        root=[0,0,width,height];parts=children(root)
        if min(width,height)<4:assert not parts;continue
        canvas=np.zeros((height,width),dtype=int)
        for x0,y0,x1,y1 in parts:canvas[y0:y1,x0:x1]+=1
        assert np.array_equal(canvas,np.ones_like(canvas))
        for parent in parts:
            full=map_box(parent,[0,0,1000,1000]);assert full==parent
            inner=map_box(parent,[123,234,765,876]);assert parent[0]<=inner[0]<inner[2]<=parent[2] and parent[1]<=inner[1]<inner[3]<=parent[3]
        checks.append(dict(shape=[width,height],partition_pixel_cover_exact=True,nested_coordinates_valid=True))
    image=Image.fromarray(rng.integers(0,256,(97,151,3),dtype=np.uint8));original=out/'original.png';image.save(original)
    box=[17,19,124,91];crop=image.crop(tuple(box));cropped=out/'crop.png';crop.save(cropped)
    assert np.array_equal(np.asarray(crop),np.asarray(image)[19:91,17:124]);crop.close();image.close()
    j=cpu_position_renderer()
    for count in (18,20):
        frames=[(float(i),original) for i in range(count)];segments=[(0.,18.,'An observable sign and hand gesture appear.')]
        msgs,files=j.prefix_messages(frames,segments);text,base=j.encode_prefix(msgs,files)
        qid,qtext=j.branch_ids(msgs,VIDEO_QUESTION,head_text=text)
        for stance in ('Yes','No'):
            aid,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance);ended=torch.cat([base['input_ids'],torch.tensor([qid+aid])],1)
            p,delta=positions(j,ended,base['image_grid_thw'])
            ctx=dict(msgs=msgs,history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',stance)],head=text+qtext+atext,stance_cache_logical_start=int(p.max())+1)
            content=[dict(type='text',text='Source original at actual time3.0 and pixelbox[17,19,124,91].\n'),dict(type='image'),dict(type='text',text='Actual source crop.\n'),dict(type='image')]
            encoded,new_positions,evidence=encode_branch(j,ctx,'Is this current window visually hateful? Answer Yes or No.',content,[original,cropped])
            joined=torch.cat([ended,encoded['input_ids']],1);grids=torch.cat([base['image_grid_thw'],encoded['image_grid_thw']],0)
            fullp,_=positions(j,joined,grids)
            assert torch.equal(new_positions,fullp[:,:,ended.shape[1]:]),'source suffix positions differ from original full conversation'
            rendered=j.render(msgs+ctx['history']+[dict(role='user',content=[*content,dict(type='text',text='Is this current window visually hateful? Answer Yes or No.')])],True)
            with Image.open(original) as one,Image.open(cropped) as two:
                complete=j.encode(rendered,[one.convert('RGB') for _ in range(count)]+[one.convert('RGB'),two.convert('RGB')])
            assert torch.equal(joined,complete['input_ids']),'source token seam differs from full native conversation'
            checks.append(dict(native_frames=count,stance=stance,actual_processor_full_token_seam_exact=True,full_native_mRoPE_suffix_exact=True,source_grid=encoded['image_grid_thw'].tolist()))
    (out/'summary.json').write_text(json.dumps(dict(host=socket.gethostname(),GT_read=False,PASS=True,checks=checks),indent=2)+'\n');print(json.dumps(checks,indent=2));print('INPUT_CPU_PASS',flush=True)


if __name__=='__main__':main()
