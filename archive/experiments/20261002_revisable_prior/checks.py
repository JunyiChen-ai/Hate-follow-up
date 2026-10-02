#!/usr/bin/env python3
"""Independent path enumeration and parity checks, no labels or evaluation."""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

from model import ROOT, terms, quadrature, normal_log, fit, load_videos
from src.temporal import build_chain, fb_grid, bma_fb


def toy():
    v={"n":3,"L":48,"x":-.4,"cells":np.array([1,2]),"pair":np.array([True,False]),
       "obs":[{"use":np.array([0,1]),"y":np.array([.9,-.3])},
              {"use":np.array([0,1]),"y":np.array([-.2,1.1])}],
       "logw":np.log([.4,.6]),"chains":[]}
    for m in ("z_visual","z_speech"):
        v["chains"].append([build_chain(1,6,9,.5,mods=(m,)),build_chain(1,12,7,.5,mods=(m,))])
    p={"mu":np.array([-1.,1.,-.8,.8,-.7,.7]),"var":np.array([.8,.7,.9]),
       "tau":.6,"pi":.45,"iota":np.array([[.6,.4],[.4,.6]]),"arm":"full"}
    return v,p


def brute(v,p,nodes):
    u,lw=quadrature(nodes); logs=[]; hs=[]; modes=[]; offsets=[]
    for q in range(nodes):
        b=p["tau"]*u[q]
        lp=math.log1p(-p["pi"])+lw[q]+normal_log(v["x"],p["mu"][0]+b,p["var"][0])
        lp+=sum(float(normal_log(o["y"],p["mu"][2+2*j]+b,p["var"][j+1]).sum()) for j,o in enumerate(v["obs"]))
        logs.append(lp);hs.append(np.zeros(v["n"]));modes.append(0);offsets.append(b)
        options=[]
        for j in range(2):
            opts=[]
            for ci,ch in enumerate(v["chains"][j]):
                for path in itertools.product(range(ch["S"]),repeat=v["n"]):
                    pr=p["iota"][j,ch["ph"][path[0]]] / ch["k"]
                    for t in range(1,v["n"]):pr*=ch["A"][path[t-1],path[t]]
                    if not pr:continue
                    h=ch["hb"][list(path)]
                    l=math.log(pr)+v["logw"][ci]
                    for i,y in zip(v["obs"][j]["use"],v["obs"][j]["y"]):
                        c=v["cells"][i];on=max(h[c],h[c-1] if v["pair"][i] else 0)
                        l+=normal_log(y,p["mu"][2+2*j+on]+b,p["var"][j+1])
                    opts.append((l,h))
            options.append(opts)
        for left,right in itertools.product(*options):
            lp=math.log(p["pi"])+lw[q]+normal_log(v["x"],p["mu"][1]+b,p["var"][0])+left[0]+right[0]
            logs.append(lp);hs.append(np.maximum(left[1],right[1]));modes.append(1);offsets.append(b)
    logs=np.array(logs);LL=logsumexp(logs);w=np.exp(logs-LL);active=np.array(modes)==1
    uncond=w@np.array(hs);conditional=uncond/w[active].sum()
    return {"ll":float(LL),"conditional":conditional,"unconditional":uncond,
            "lo":float(logsumexp(logs[active])-logsumexp(logs[~active])),"offset_mean":float(w@offsets)}


def main():
    out=ROOT/"runs/20261002_revisable_prior/checks";out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(0);worst=0.
    for kind in ("two","three_nested","joint"):
        for _ in range(5):
            n=5;phase={"two":2,"three_nested":3,"joint":4}[kind]
            io=rng.dirichlet(np.ones(phase))
            chs=[build_chain(1,float(a),float(b),io,kind=kind) for a,b in rng.uniform(5,30,(3,2))]
            E=rng.normal(0,2,(3,n,len(chs[0]["pi0"])));lw=np.log(rng.dirichlet(np.ones(3)))
            ll,g,w=fb_grid(E,chs,lw)
            for i in range(3):
                ll0,g0,w0=bma_fb(E[i],chs,lw)
                worst=max(worst,abs(ll[i]-ll0),float(np.abs(g[i]-g0).max()),float(np.abs(w[i]-w0).max()))
    assert worst<1e-10,worst
    v,p=toy();fast=terms(v,p,3);slow=brute(v,p,3)
    brute_error=max(float(np.max(np.abs(np.asarray(fast[k])-slow[k]))) for k in slow)
    assert brute_error<1e-10,brute_error
    p["tau"]=0;fast=terms(v,p,7);p["arm"]="independent";ind=terms(v,p,1)
    independence_error=max(float(np.max(np.abs(np.asarray(fast[k])-ind[k]))) for k in ("ll","lo","conditional","unconditional"))
    assert independence_error<1e-10,independence_error
    v["x"]=-1
    for o in v["obs"]:o["y"][:]=3
    over=terms(v,p,1)
    assert over["lo"]>0,over["lo"]
    data=load_videos(ROOT/"runs/20260910_spvl/mllm/q3vl-8b/nostance",["HateMM"])["HateMM"]
    # Numerical smoke only; complete-corpus transforms, no GT, no selection by outcomes.
    fitted=fit(data[:3],"full",3,3)
    info={"seed":0,"batch_vs_scalar_max_error":worst,"joint_path_enumeration_max_error":brute_error,
          "tau_zero_vs_independent_max_error":independence_error,"wrong_global_prior_overturned_toy_logodds":over["lo"],
          "EM_smoke_loglik":fitted["loglik_history"],"GT_read":False}
    (out/"checks.json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info,indent=2));print("CHECKS_DONE")


if __name__=="__main__":main()
