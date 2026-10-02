#!/usr/bin/env python3
"""Frozen-parameter intervention on global observations; no labels in scoring."""
import argparse
import json
import math
import socket
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import norm, rankdata, spearmanr

from model import ROOT, load_videos, normal_log, quadrature, terms


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--tag",required=True);a=ap.parse_args()
    run=ROOT/"runs/20261002_revisable_prior"/a.tag
    cfg=json.load((run/"config.json").open());saved=json.load((run/"params.json").open())
    out=run/"diagnostics";out.mkdir(exist_ok=True)
    print("host",socket.gethostname(),flush=True)
    videos=load_videos(ROOT/cfg["run"],cfg["datasets"])
    rows=[];summary={}
    for ds,vs in videos.items():
        p={**saved[ds]}
        for k in ("mu","var","iota"):p[k]=np.asarray(p[k],float)
        for v in vs:
            t=terms(v,p,cfg["nodes"],grid_step=cfg.get("grid_step"))
            glob=normal_log(v["x"],p["mu"][:2],p["var"][0]+p["tau"]**2)
            prior=math.log(p["pi"])-math.log1p(-p["pi"])+glob[1]-glob[0]
            lo=terms(v,p,cfg["nodes"],x_override=float(norm.ppf(.1)),grid_step=cfg.get("grid_step"))
            hi=terms(v,p,cfg["nodes"],x_override=float(norm.ppf(.9)),grid_step=cfg.get("grid_step"))
            aa,bb=lo["conditional"],hi["conditional"]
            r=float(spearmanr(aa,bb).statistic) if np.ptp(aa)>0 and np.ptp(bb)>0 else None
            change=float(np.max(np.abs(aa-bb)))
            rows.append({"dataset":ds,"video_id":v["rec"]["video_id"],"raw_z":v["rec"]["extra"]["z_video"],
                         "prior_logodds":float(prior),"posterior_logodds":t["lo"],"revised_prior_sign":bool((prior>0)!=(t["lo"]>0)),
                         "offset_mean":t["offset_mean"],"force_low_logodds":lo["lo"],"force_high_logodds":hi["lo"],
                         "max_conditional_change":change,"rank_spearman":r})
        rr=[r for r in rows if r["dataset"]==ds]
        summary[ds]={"n":len(rr),"prior_sign_revisions":sum(r["revised_prior_sign"] for r in rr),
                     "max_conditional_change_median":float(np.median([r["max_conditional_change"] for r in rr])),
                     "rank_spearman_median":float(np.median([r["rank_spearman"] for r in rr if r["rank_spearman"] is not None])),
                     "conditional_changes_over_1e8":sum(r["max_conditional_change"]>1e-8 for r in rr)}
        print(ds,json.dumps(summary[ds]),flush=True)
    (out/"global_interventions.json").write_text(json.dumps({"rows":rows,"summary":summary,
        "intervention":"global transformed observation forced to N(0,1) quantile .1/.9, local reads/params fixed",
        "GT_used":False,"interpretation":"sensitivity and revisability check; not evidence that revisions are correct"},indent=2)+"\n")
    print("DIAGNOSTICS_DONE",flush=True)


if __name__=="__main__":main()
