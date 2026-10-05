"""Actual-pixel region partition and nested normalized-box source mapping."""
import json
import sys
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
SPEC=json.loads((Path(__file__).parent/'spec.json').read_text())


def validate_box(box,shape):
    assert len(box)==4 and all(type(v) is int for v in box)
    x0,y0,x1,y1=box;w,h=shape
    assert 0<=x0<x1<=w and 0<=y0<y1<=h


def children(box):
    x0,y0,x1,y1=box;w,h=x1-x0,y1-y0
    assert w>0 and h>0
    if min(w,h)<SPEC['minimum_partition_axis_pixels']:return []
    nx,ny=(4,1) if w>=2*h else ((1,4) if h>=2*w else (2,2))
    sx,sy=w//nx,h//ny
    result=[]
    for j in range(ny):
        for i in range(nx):
            result.append([x0+i*sx,y0+j*sy,x1 if i==nx-1 else x0+(i+1)*sx,
                y1 if j==ny-1 else y0+(j+1)*sy])
    assert len(result)==4 and sum((b[2]-b[0])*(b[3]-b[1]) for b in result)==w*h
    return result


def map_box(parent,normalized):
    scale=SPEC['coordinate_scale'];validate_box(normalized,(scale,scale))
    px,py,qx,qy=parent;w,h=qx-px,qy-py
    assert w>0 and h>0
    x0,y0,x1,y1=normalized
    result=[px+x0*w//scale,py+y0*h//scale,px+(x1*w+scale-1)//scale,py+(y1*h+scale-1)//scale]
    validate_box(result,(qx,qy));assert result[0]>=px and result[1]>=py
    return result


def priority_children(box,order,serial_start):
    pieces=children(box)
    assert sorted(order)==list(range(4))
    rank={index:r for r,index in enumerate(order)}
    return [dict(box=p,priority=rank[i],serial=serial_start+i) for i,p in enumerate(pieces)]
