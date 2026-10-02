#!/usr/bin/env python3
"""Paired label-free M1 head routing; no evaluation/GT dependency."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"CLAUDE.md").is_file())
sys.path.insert(0,str(ROOT))
from src.mllm_judge import Judge, MODEL, VIDEO_QUESTION, yesno_question
from src.video_inputs import FPS, frame_paths, load_asr, load_manifest, fixed_windows, window_text
from src.window_token_regions import token_regions
from selector import ARMS, HeadSelector


def unpack_heads(indices,nheads,device):
    result=torch.zeros((len(indices),nheads),dtype=torch.bool)
    for layer,heads in enumerate(indices):result[layer,heads]=True
    return result.to(device)


def read_video(j,router,row,segments,arms,verify=False,replay_record=None):
    started=time.perf_counter();ds,vid,dur=row["dataset"],row["video_id"],float(row["duration"])
    wins=fixed_windows(dur,8);texts=[window_text(segments,a,b) for a,b in wins]
    frames=frame_paths(ds,vid,20,"k20");assert frames,(ds,vid)
    msgs,files=j.prefix_messages(frames,segments);prefix,enc=j.encode_prefix(msgs,files)
    regions=token_regions(j,prefix,enc,frames,segments,wins)
    cache=j.prefix_cache(enc);b0,b0text=j.branch_ids(msgs,VIDEO_QUESTION)
    zv=j.cached_margin(cache,b0,in_place=True);stance="Yes" if zv>0 else "No"
    a0,a0text=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,a0)
    history=[{"role":"user","content":[{"type":"text","text":VIDEO_QUESTION}]},j.turn("assistant",stance)]
    head=prefix+b0text+a0text;ncache=cache.get_seq_length()
    if replay_record is not None:
        assert replay_record["extra"]["z_video"]==zv
        assert replay_record["extra"]["prefix_tokens"]==regions["prefix_len"]
        assert [(w["start"],w["end"]) for w in replay_record["extra"]["windows"]]==wins
    torch.cuda.synchronize();prefix_seconds=time.perf_counter()-started
    per={arm:[{} for _ in wins] for arm in arms};timing={arm:0. for arm in arms}
    checks={};branches=0;ordinary_seconds=0.;ordinary_maxdiff=0.
    for kind in ("visual","speech"):
        for i,((a,b),txt) in enumerate(zip(wins,texts)):
            if kind=="speech" and not txt.strip():continue
            q=yesno_question(i,len(wins),a,b,txt,kind);bids,_=j.branch_ids(msgs,q,history,head_text=head)
            replay=None
            if replay_record is not None:
                replay=unpack_heads(replay_record["extra"]["windows"][i]["heads_"+kind],router.layers[0].self_attn.config.num_attention_heads,j.device)
            for arm in arms:
                torch.cuda.synchronize();tick=time.perf_counter()
                z,trajectory=router.margin(cache,bids,regions,i,arm,replay=replay)
                torch.cuda.synchronize();timing[arm]+=time.perf_counter()-tick
                per[arm][i]["z_"+kind]=float(z)
                if arm!="base":
                    per[arm][i]["heads_"+kind]=[np.flatnonzero(v).tolist() for v in trajectory.cpu().numpy()]
                if arm=="select":replay=trajectory
                if verify and branches==0:
                    ref,reftrace=router.margin(cache,bids,regions,i,arm,replay=replay,copy_cache=True)
                    checks[arm+"_crop_copy_abs_diff"]=abs(z-ref)
                    assert abs(z-ref)<1e-2 and torch.equal(trajectory,reftrace),(arm,z,ref)
            if verify and "base" in arms:
                torch.cuda.synchronize();tick=time.perf_counter()
                ordinary=j.cached_margin(cache,bids,in_place=True);cache.crop(ncache)
                torch.cuda.synchronize();ordinary_seconds+=time.perf_counter()-tick
                diff=abs(ordinary-per["base"][i]["z_"+kind]);ordinary_maxdiff=max(ordinary_maxdiff,diff)
                assert diff<1e-2,(ds,vid,kind,i,diff,"wrapper changed base")
            branches+=1
    records={};stats=[]
    for i,(a,b) in enumerate(wins):
        m=regions["local"][i]
        stats.append({"start":a,"end":b,"kept_visual_tokens":int((m&regions["visual"]).sum()),
                      "kept_speech_tokens":int((m&regions["speech"]).sum()),"kept_media_tokens":int(m.sum()),
                      "shift_overlap":int((m&regions["shifted"][i]).sum()),
                      "prefix_media_tokens":int((regions["visual"]|regions["speech"]).sum())})
    L=int(math.ceil(dur*FPS));index=np.clip(((np.arange(L)+.5)/FPS//8).astype(int),0,len(wins)-1)
    for arm in arms:
        z=np.array([max(w[k] for k in ("z_visual","z_speech") if k in w) for w in per[arm]])
        records[arm]={"schema_version":1,"method":"m1_selector_"+arm,"dataset":ds,"video_id":vid,
            "duration":dur,"native_rate":FPS,"score_curve":z[index].tolist(),"intervals":[],"error":None,
            "seed":0,"code_path":str(Path(__file__).relative_to(ROOT)),"calls":3+branches,
            "extra":{"z_video":zv,"stance":stance,"n_branches":branches,"prefix_tokens":regions["prefix_len"],
                "cached_tokens":ncache,"text_layers":len(router.layers),"n_heads":trajectory.shape[-1],
                "prefix_seconds":prefix_seconds,"branch_seconds":timing[arm],
                "windows":[{"i":i,"z":float(z[i]),**per[arm][i],**stats[i]} for i in range(len(wins))]}}
    if verify:checks.update({"ordinary_max_abs_diff":ordinary_maxdiff,"ordinary_branch_seconds":ordinary_seconds})
    del cache
    return records,{"dataset":ds,"video_id":vid,"checks":checks,"prefix_seconds":prefix_seconds,"branch_seconds":timing}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-name",required=True)
    ap.add_argument("--arms",nargs="+",choices=ARMS,default=["base","select"])
    ap.add_argument("--replay-run",help="completed select-arm directory containing predictions.jsonl")
    ap.add_argument("--smoke",action="store_true");a=ap.parse_args()
    if any(x in a.arms for x in ("permuted_heads","shifted_support")) and not ("select" in a.arms or a.replay_run):
        ap.error("replay controls need select earlier in arm order or --replay-run")
    if "select" in a.arms:
        assert all(a.arms.index("select")<a.arms.index(x) for x in ("permuted_heads","shifted_support") if x in a.arms)
    torch.manual_seed(0);out=ROOT/"runs/20261002_m1_selector"/a.run_name;out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",handlers=[logging.FileHandler(out/"run.log"),logging.StreamHandler(sys.stdout)])
    logging.info("host %s",socket.gethostname());(out/"run.pid").write_text(str(os.getpid()))
    rows=load_manifest(ROOT/"data/omsl_v6_inputs/manifests/all_test.jsonl",["HateMM","HateClipSeg"])
    if a.smoke:rows=[r for ds in ("HateMM","HateClipSeg") for r in [x for x in rows if x["dataset"]==ds][:2]]
    asr={ds:load_asr(ds) for ds in ("HateMM","HateClipSeg")}
    replay={}
    if a.replay_run:
        replay={(r["dataset"],r["video_id"]):r for r in map(json.loads,(Path(a.replay_run)/"predictions.jsonl").open())}
    j=Judge(MODEL);router=HeadSelector(j)
    import transformers
    config={**vars(a),"date":time.strftime("%Y-%m-%d"),"host":socket.gethostname(),"seed":0,
        "model":MODEL,"torch":torch.__version__,"transformers":transformers.__version__,"GT_in_reader":False,
        "code":"experiments/20261002_m1_selector/{measure,selector}.py; src/window_token_regions.py; current local sources 2026-10-02",
        "routing":"per-token local mean affinity > remote mean; current path, each layer, last query position; route entire observed query",
        "video_question":VIDEO_QUESTION,"visual_question":yesno_question(0,1,0,8,"","visual"),
        "speech_question":yesno_question(0,1,0,8,"<text>","speech"),"calls_note":"3+n_branches per deployed arm; shared prefix/global in paired run; verification calls extra"}
    (out/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    handles={};done={}
    for arm in a.arms:
        d=out/arm;d.mkdir(exist_ok=True);(d/"config.json").write_text(json.dumps({**config,"arm":arm},indent=2)+"\n")
        path=d/"predictions.jsonl"
        done[arm]={(r["dataset"],r["video_id"]) for r in map(json.loads,path.open())} if path.exists() else set()
        handles[arm]=path.open("a")
    assert all(v==done[a.arms[0]] for v in done.values()),"partial per-arm records need repair before resume"
    checks=(out/"checks.jsonl").open("a");start=time.time()
    for k,row in enumerate(rows):
        key=row["dataset"],row["video_id"]
        if key in done[a.arms[0]]:continue
        recs,checked=read_video(j,router,row,asr[key[0]].get(key[1],[]),a.arms,verify=a.smoke,replay_record=replay.get(key))
        if a.replay_run:assert key in replay,key
        for arm,r in recs.items():handles[arm].write(json.dumps(r)+"\n");handles[arm].flush()
        checks.write(json.dumps(checked)+"\n");checks.flush()
        logging.info("progress %d/%d %s %s elapsed=%.1f",k+1,len(rows),*key,time.time()-start)
    for f in handles.values():f.close()
    checks.close();router.close();logging.info("RUN_DONE videos=%d elapsed=%.1f",len(rows),time.time()-start)


if __name__=="__main__":main()
