#!/usr/bin/env python3
"""No-GT tests of the pixel intervention and binary contrast definition."""
import json
from pathlib import Path
import torch
from visual_contrast import corrupt_pixels,class_contrast,schedule
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


def main():
    torch.manual_seed(0);torch.set_num_threads(1)
    checks=[]
    for patch in (2,16):
        x=torch.randn(12,3,1,patch,patch).expand(-1,-1,2,-1,-1).reshape(12,-1).clone()
        saved=x.clone();z,meta=corrupt_pixels(x,patch,2);again,_=corrupt_pixels(x,patch,2)
        identity,_=corrupt_pixels(x,patch,2,abar=1.)
        assert torch.equal(z,again) and torch.equal(identity,x) and torch.equal(x,saved)
        zz=z.reshape(-1,3,2,patch,patch);assert torch.equal(zz[:,:,0],zz[:,:,1])
        residual=(z-schedule()**.5*x).reshape(-1,3,2,patch,patch)
        assert not torch.equal(residual[0],residual[1]) and not torch.equal(z,x)
        checks.append({'patch_size':patch,'identity_exact':True,'deterministic':True,'original_unchanged':True,
            'temporal_copies_equal':True,'spatial_patch_noise_differs':True,**meta})
    grid=torch.linspace(-6,6,1000,dtype=torch.float64)
    double=float(torch.exp(torch.log1p(-(1e-5+(.005-1e-5)*torch.sigmoid(grid)))[:501].sum()))
    assert abs(schedule()-double)<1e-6 and schedule(0)<1.
    clean=torch.tensor([-20.,-1.,0.,5.,20.],dtype=torch.float64)
    corrupted=torch.tensor([3.,-2.,0.,4.,-10.],dtype=torch.float64)
    assert torch.equal(class_contrast(clean,corrupted,0.),clean)
    assert torch.equal(class_contrast(clean,clean),clean)
    # Class log-probability contrast, then difference, equals2a-b.
    lp_a=torch.stack((torch.nn.functional.logsigmoid(clean),torch.nn.functional.logsigmoid(-clean)),1)
    lp_b=torch.stack((torch.nn.functional.logsigmoid(corrupted),torch.nn.functional.logsigmoid(-corrupted)),1)
    d=2*lp_a-lp_b;torch.testing.assert_close(d[:,0]-d[:,1],class_contrast(clean,corrupted),rtol=0,atol=1e-12)
    out=ROOT/'runs/20261003_m1_visual_contrast/selfcheck';out.mkdir(parents=True,exist_ok=True)
    result={'pixel_tests':checks,'source_schedule_double_abs_error':abs(schedule()-double),
        'step0_is_not_identity':True,'class_probability_algebra_exact':True,'GT_read':False}
    (out/'noise.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('SELFCHECK_DONE')

if __name__=='__main__':main()
