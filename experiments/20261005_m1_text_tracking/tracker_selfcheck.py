"""Known synthetic pixel transforms, not hate labels or a performance dataset."""
import json
from pathlib import Path
import socket
import sys
import numpy as np
import cv2
from tracking import advance,appearance,cut,crop,iou,SPEC

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
out=ROOT/'runs/20261005_m1_text_tracking/tracker_cpu_checks'
out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(0)
original=np.full((160,224,3),40,dtype=np.uint8)
box=[60,50,148,100]
texture=rng.integers(0,256,(50,88,3),dtype=np.uint8)
original[50:100,60:148]=texture
transformed=np.full_like(original,40)
shift=[3,2];expected=[63,52,151,102]
transformed[52:102,63:151]=texture
predicted,evidence=advance(original,transformed,box,crop(original,box))
assert predicted==expected,(predicted,evidence)
assert np.array_equal(crop(transformed,predicted),texture)
assert evidence['translation_original_pixels']==shift and evidence['reason']=='accepted'
assert not cut(original,transformed)[0]
accepted,visual=appearance(texture,texture.copy());assert accepted and visual['ncc']>=1-1e-12 and visual['mae']==0
altered=texture.copy();altered[:]=128
accepted,_=appearance(texture,altered);assert not accepted
assert cut(original,np.full_like(original,255))[0]
assert advance(original,np.zeros((10,20,3),np.uint8),box,texture)[0] is None
assert advance(np.full_like(original,40),np.full_like(original,40),box,texture)[0] is None
assert iou(box,box)==1 and iou(box,[0,0,10,10])==0
summary=dict(host=socket.gethostname(),scope='synthetic pixel transforms only; no OCR/model/GT/metrics',
    GT_read=False,PASS=True,opencv=cv2.__version__,expected_shift=shift,actual_box=predicted,
    initial_points=evidence['initial_points'],valid_points=evidence['valid_points'],
    appearance=evidence['appearance'],checks=['known actual RGB translation','exact supported crop pixels',
        'same-image appearance','changed-texture rejection','cut termination','shape rejection','no-corner rejection','region IoU'])
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2),flush=True)
