"""VFR/gap/half-open acquisition and actual spatial-quadrant software checks."""
import json
import socket
from fractions import Fraction
import av
import numpy as np
import torch
from inputs import ROOT,SPEC,quadrants
from src.fractional_window_frames import acquire,validate,selected


def main():
    print(socket.gethostname(),flush=True);out=ROOT/'runs/20261007_m1_mukv/source_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    path=out/'fixture.mkv';times=[0,1000,7000,8000,8150,12000,16000,17000,20000,23000]
    with av.open(str(path),'w') as container:
        stream=container.add_stream('ffv1',rate=1000);stream.width=16;stream.height=16;stream.pix_fmt='bgr0'
        stream.time_base=Fraction(1,1000);stream.codec_context.time_base=Fraction(1,1000)
        for i,t in enumerate(times):
            array=np.full((16,16,3),i*20,np.uint8);frame=av.VideoFrame.from_ndarray(array,format='rgb24');frame.pts=t;frame.time_base=Fraction(1,1000)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    row=dict(dataset='HateMM',video_id='fixture',video_path=str(path),duration=24.)
    meta=acquire(row,out/'frames',SPEC['source_fractions'],8);validate(meta,row,out/'frames',SPEC['source_fractions'],8)
    assert meta['target_indices']==[[1,2,2,2],[5,5,None,None],[7,8,9,9]]
    assert meta['selected_indices']==[[1,2],[5],[7,8,9]]
    # Actual sparse coordinates: groups have unequal widths and a nonzero
    # logical origin, so flattened quarter strips cannot pass this oracle.
    pos=torch.tensor([[[40]*6],[[50,50,50,51,51,51]],[[60,61,62,60,61,62]]]);assert quadrants(pos,list(range(6)))==[[0],[1,2],[3],[4,5]]
    singleton=torch.tensor([[[10]],[[20]],[[30]]]);assert quadrants(singleton,[0])==[[],[],[],[0]]
    missing=selected([],[(0,8)],[.125,.375,.625,.875]);assert missing==[[None]*4]
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,actual_VFR_rawRGB_replay=True,duplicate_missing_targets_explicit=True,
        no_crosswindow_or_lastframe_fallback=True,spatial_quadrant_oracle=True,source=meta,
        scope='actual synthetic video input/coordinate pipeline only; no complete sourceLM/3grain/8B/model/performance'),indent=2)+'\n');print('SOURCE_CPU_PASS',flush=True)


if __name__=='__main__':main()
