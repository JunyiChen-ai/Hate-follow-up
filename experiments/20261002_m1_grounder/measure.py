#!/usr/bin/env python3
"""Paired, label-free M1 attention intervention; all downstream fitting is separate."""
import argparse
import json
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.mllm_judge import Judge, MODEL, VIDEO_QUESTION, yesno_question
from src.video_inputs import FPS, frame_paths, load_asr, load_manifest, fixed_windows, window_text
from grounder import ARMS, QueryAccess, token_regions, window_keep


def read_video(j,access,row,segments,arms,verify=False):
    started=time.perf_counter()
    ds,vid,dur=row["dataset"],row["video_id"],float(row["duration"])
    wins=fixed_windows(dur,8);texts=[window_text(segments,a,b) for a,b in wins]
    frames=frame_paths(ds,vid,20,"k20");assert frames,(ds,vid,"missing frames")
    msgs,files=j.prefix_messages(frames,segments)
    prefix,enc=j.encode_prefix(msgs,files)
    regions=token_regions(j,prefix,enc,frames,segments,wins)
    cache=j.prefix_cache(enc)
    b0,b0text=j.branch_ids(msgs,VIDEO_QUESTION)
    if verify:j.seam_check_tokens(msgs,files,enc["input_ids"][0].tolist(),VIDEO_QUESTION,b0)
    zv=j.cached_margin(cache,b0,in_place=True)
    stance="Yes" if zv>0 else "No"
    a0,a0text=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,a0)
    history=[{"role":"user","content":[{"type":"text","text":VIDEO_QUESTION}]},j.turn("assistant",stance)]
    head=prefix+b0text+a0text
    torch.cuda.synchronize();prefix_seconds=time.perf_counter()-started
    ncache=cache.get_seq_length()
    per={arm:[{} for _ in wins] for arm in arms};timing={arm:0. for arm in arms}
    verified={};branches=0
    for kind in ("visual","speech"):
        for i,((a,b),txt) in enumerate(zip(wins,texts)):
            if kind=="speech" and not txt.strip():continue
            q=yesno_question(i,len(wins),a,b,txt,kind)
            bids,_=j.branch_ids(msgs,q,history,head_text=head)
            for arm in arms:
                keep=window_keep(regions,i,ncache,arm)
                torch.cuda.synchronize();tick=time.perf_counter()
                z=access.margin(cache,bids,keep,arm)
                torch.cuda.synchronize();timing[arm]+=time.perf_counter()-tick
                per[arm][i]["z_"+kind]=float(z)
                if verify and branches==0:
                    ref=access.margin(cache,bids,keep,arm,copy_cache=True)
                    assert abs(z-ref)<1e-2,(arm,z,ref,"crop/copy mismatch")
                    verified[arm+"_crop_copy_abs_diff"]=abs(z-ref)
                    if arm=="base":
                        plain=j.cached_margin(cache,bids,in_place=False)
                        verified["base_hook_vs_ordinary_abs_diff"]=abs(z-plain)
                        assert abs(z-plain)<.25,(z,plain,"mask plumbing changes baseline")
            branches+=1
    P=regions["prefix_len"]
    stats=[]
    for i,(a,b) in enumerate(wins):
        m=regions["local"][i];shifted=regions["shifted"][i]
        stats.append({"start":a,"end":b,"kept_visual_tokens":int((m&regions["visual"]).sum()),
                      "kept_speech_tokens":int((m&regions["speech"]).sum()),
                      "kept_media_tokens":int(m.sum()),"shift_overlap":int((m&shifted).sum()),
                      "prefix_media_tokens":int((regions["visual"]|regions["speech"]).sum())})
    L=int(math.ceil(dur*FPS));index=np.clip(((np.arange(L)+.5)/FPS//8).astype(int),0,len(wins)-1)
    records={}
    for arm in arms:
        z=np.array([max(w.values()) for w in per[arm]])
        records[arm]={"schema_version":1,"method":"m1_grounder_"+arm,"dataset":ds,"video_id":vid,
            "duration":dur,"native_rate":FPS,"score_curve":z[index].tolist(),"intervals":[],"error":None,
            "seed":0,"code_path":"experiments/20261002_m1_grounder/measure.py",
            "calls":2,"extra":{"z_video":zv,"stance":stance,"n_branches":branches,
                "prefix_tokens":P,"cached_tokens":ncache,"text_layers":len(access.layers),"late_start_layer":access.cut,
                "prefix_seconds":prefix_seconds,"branch_seconds":timing[arm],
                "windows":[{"i":i,"start":a,"end":b,"z":float(z[i]),**per[arm][i],**stats[i]}
                           for i,(a,b) in enumerate(wins)]}}
    del cache
    return records,{"dataset":ds,"video_id":vid,"checks":verified,"windows":stats,
                    "prefix_seconds":prefix_seconds,"branch_seconds":timing}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-name",required=True)
    ap.add_argument("--datasets",nargs="+",default=["HateMM","HateClipSeg"])
    ap.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    ap.add_argument("--model",default=MODEL)
    ap.add_argument("--smoke",action="store_true",help="first two manifest records per corpus; plumbing only")
    a=ap.parse_args();torch.manual_seed(0)
    out=ROOT/"runs/20261002_m1_grounder"/a.run_name;out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out/"run.log"),logging.StreamHandler(sys.stdout)])
    logging.info("host %s",socket.gethostname());(out/"run.pid").write_text(str(os.getpid()))
    manifest=ROOT/"data/omsl_v6_inputs/manifests/all_test.jsonl"
    rows=load_manifest(manifest,a.datasets)
    if a.smoke:rows=[r for ds in a.datasets for r in [x for x in rows if x["dataset"]==ds][:2]]
    asr={ds:load_asr(ds) for ds in a.datasets}
    j=Judge(a.model);access=QueryAccess(j)
    import transformers
    config={**vars(a),"date":time.strftime("%Y-%m-%d"),"host":socket.gethostname(),"seed":0,
            "torch":torch.__version__,"transformers":transformers.__version__,"GT_in_reader":False,
            "code":"experiments/20261002_m1_grounder/{measure,grounder}.py; local sources 2026-10-02",
            "text_layers":len(access.layers),"late_start_layer":access.cut,"mask_plumbing":"explicit bf16 causal in every query layer",
            "video_question":VIDEO_QUESTION,"visual_question":yesno_question(0,1,0,8,"","visual"),
            "speech_question":yesno_question(0,1,0,8,"<text>","speech"),"calls_note":"same prefix and global read shared among experimental arms, not extra deployment calls"}
    (out/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    handles={};done={}
    for arm in a.arms:
        d=out/arm;d.mkdir(exist_ok=True)
        (d/"config.json").write_text(json.dumps({**config,"arm":arm},indent=2)+"\n")
        path=d/"predictions.jsonl"
        done[arm]={(r["dataset"],r["video_id"]) for r in map(json.loads,path.open())} if path.exists() else set()
        handles[arm]=path.open("a")
    assert all(v==done[a.arms[0]] for v in done.values()),"partial per-arm record; inspect and repair before resume"
    logchecks=(out/"checks.jsonl").open("a");start=time.time()
    for k,row in enumerate(rows):
        key=row["dataset"],row["video_id"]
        if key in done[a.arms[0]]:continue
        recs,checks=read_video(j,access,row,asr[key[0]].get(key[1],[]),a.arms,verify=a.smoke or k==0)
        for arm,r in recs.items():handles[arm].write(json.dumps(r)+"\n");handles[arm].flush()
        logchecks.write(json.dumps(checks)+"\n");logchecks.flush()
        logging.info("progress %d/%d %s %s elapsed=%.1f",k+1,len(rows),*key,time.time()-start)
    for f in handles.values():f.close()
    logchecks.close();access.close()
    logging.info("RUN_DONE videos=%d elapsed=%.1f",len(rows),time.time()-start)


if __name__=="__main__":main()
