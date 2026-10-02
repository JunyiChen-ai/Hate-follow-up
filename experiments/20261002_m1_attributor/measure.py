#!/usr/bin/env python3
"""Paired native window judgments and global cached-value attribution."""
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
from scipy.stats import spearmanr

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"CLAUDE.md").is_file())
sys.path.insert(0,str(ROOT))
from src.mllm_judge import Judge,MODEL,VIDEO_QUESTION,yesno_question
from src.video_inputs import FPS,frame_paths,load_asr,load_manifest,fixed_windows,window_text
from src.window_token_regions import token_regions
from attributor import Attributor,cache_branch,window_contributions


def read_video(j,attributor,row,segments,smoke=False):
    started=time.perf_counter();ds,vid,dur=row["dataset"],row["video_id"],float(row["duration"])
    wins=fixed_windows(dur,8);texts=[window_text(segments,a,b) for a,b in wins]
    frames=frame_paths(ds,vid,20,"k20");assert frames,(ds,vid)
    msgs,files=j.prefix_messages(frames,segments);prefix,enc=j.encode_prefix(msgs,files)
    regions=token_regions(j,prefix,enc,frames,segments,wins)
    cache=j.prefix_cache(enc);P=cache.get_seq_length();b0,b0text=j.branch_ids(msgs,VIDEO_QUESTION)
    torch.cuda.synchronize();prefix_seconds=time.perf_counter()-started
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    base_cache=cache_branch(cache)
    global_tick=time.perf_counter()
    zv=j.cached_margin(base_cache,b0,in_place=True);torch.cuda.synchronize()
    native_global_seconds=time.perf_counter()-global_tick;stance="Yes" if zv>0 else "No"
    a0,a0text=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(base_cache,a0)
    history=[{"role":"user","content":[{"type":"text","text":VIDEO_QUESTION}]},j.turn("assistant",stance)]
    head=prefix+b0text+a0text;B=base_cache.get_seq_length();base=[{} for _ in wins];branches=0
    for kind in ("visual","speech"):
        for i,((a,b),txt) in enumerate(zip(wins,texts)):
            if kind=="speech" and not txt.strip():continue
            question=yesno_question(i,len(wins),a,b,txt,kind)
            bids,_=j.branch_ids(msgs,question,history,head_text=head)
            base[i]["z_"+kind]=j.cached_margin(base_cache,bids,in_place=True)
            base_cache.crop(B);branches+=1
    torch.cuda.synchronize();base_seconds=time.perf_counter()-tick
    del base_cache
    before=[(l.keys.detach().cpu().clone(),l.values.detach().cpu().clone()) for l in cache.layers] if smoke else None
    media=regions["visual"]|regions["speech"]
    torch.cuda.synchronize();tick=time.perf_counter()
    initial_recomputed=attributor.recomputed_layers
    with attributor.query_precision(b0,fp32=getattr(attributor,"fp32",False)):
        torch.cuda.synchronize();ref_tick=time.perf_counter()
        native_precision=attributor.native_margin(cache,b0)
        torch.cuda.synchronize();reference_seconds=time.perf_counter()-ref_tick
        attribution,endpoint,numeric,solutions=attributor.integrate(cache,b0,media,smoke=smoke)
    torch.cuda.synchronize();attr_seconds=time.perf_counter()-tick
    numeric["actual_recomputed_layers"]=attributor.recomputed_layers-initial_recomputed
    numeric["deployed_recomputed_layers"]=numeric["deployed_backwards"]*len(attributor.layers) if attributor.checkpoint_query else 0
    actual_integration_seconds=numeric["deployed_seconds"] if numeric.get("adaptive",False) else numeric["endpoint_seconds"]+sum(t["seconds"] for t in numeric["trials"])
    precision_overhead_seconds=attr_seconds-reference_seconds-actual_integration_seconds
    deployed_attribute_seconds=precision_overhead_seconds+numeric["deployed_seconds"]+(native_global_seconds if attributor.fp32 else 0.)
    peak=torch.cuda.max_memory_allocated()/2**30
    assert abs(numeric["f1"]-native_precision)<.01,(ds,vid,"endpoint/native",numeric["f1"],native_precision)
    if not getattr(attributor,"fp32",False):assert native_precision==zv
    restored_margin=None
    if smoke and getattr(attributor,"fp32",False):
        restored_margin=j.cached_margin(cache,b0)
        assert restored_margin==zv,("precision restoration changed native result",restored_margin,zv)
    assert np.max(np.abs(attribution[~media]),initial=0.)==0.
    if before:
        assert all(torch.equal(k,l.keys.cpu()) and torch.equal(v,l.values.cpu()) for (k,v),l in zip(before,cache.layers))
        del before
    attributed,density,mapping=window_contributions(attribution,regions,texts)
    endpoint_windows,_,_=window_contributions(endpoint,regions,texts)
    # Stored token contributions permit CPU-only signed/density/time controls.
    payload={"attribution":attribution,"endpoint":endpoint,"media":media,
        "visual":regions["visual"],"speech":regions["speech"],"local":np.stack(regions["local"])}
    stability=[]
    if smoke and not numeric.get("adaptive",False):
        for n in (16,32,64,128):
            u,_,_=window_contributions(solutions[n],regions,texts)
            v,_,_=window_contributions(solutions[256],regions,texts)
            x=np.array([max(w.values()) for w in u]);y=np.array([max(w.values()) for w in v])
            stability.append({"nodes":n,"reference":256,"window_max_abs_diff":float(np.max(abs(x-y))),
                "window_mean_abs_diff":float(np.mean(abs(x-y))),
                "window_spearman":float(spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else None,
                "token_relative_l1":float(np.abs(solutions[n]-solutions[256]).sum()/max(np.abs(solutions[256]).sum(),1e-12))})
    L=int(math.ceil(dur*FPS));index=np.clip(((np.arange(L)+.5)/FPS//8).astype(int),0,len(wins)-1)
    records={}
    for arm,values,calls,seconds in (("base",base,3+branches,base_seconds),
                                   ("attribute",attributed,1+int(attributor.fp32)+numeric["deployed_forwards"],deployed_attribute_seconds)):
        z=np.array([max(w.values()) for w in values])
        records[arm]={"schema_version":1,"method":"m1_attributor_"+arm,"dataset":ds,"video_id":vid,
            "duration":dur,"native_rate":FPS,"score_curve":z[index].tolist(),"intervals":[],"error":None,
            "seed":0,"code_path":str(Path(__file__).relative_to(ROOT)),"calls":calls,
            "extra":{"z_video":zv,"stance":stance,"prefix_tokens":P,"prefix_seconds":prefix_seconds,
                "branch_seconds":seconds,"backward_calls":0 if arm=="base" else numeric["deployed_backwards"],
                "windows":[{"i":i,"start":a,"end":b,"z":float(z[i]),**values[i]} for i,(a,b) in enumerate(wins)]}}
    diagnostic={"dataset":ds,"video_id":vid,"numeric":numeric,"mapping":mapping,"stability":stability,
        "global_native":zv,"native_attribution_precision":native_precision,"query_precision":"fp32" if getattr(attributor,"fp32",False) else "bf16",
        "endpoint_abs_diff":abs(native_precision-numeric["f1"]),"precision_margin_drift":native_precision-zv,
        "restored_native_margin":restored_margin,"diagnostic_reference_forwards":1,"diagnostic_reference_seconds":reference_seconds,
        "diagnostic_restoration_forwards":int(restored_margin is not None),
        "cache_immutable":True if smoke else None,
        "native_global_seconds":native_global_seconds,"precision_and_wrapper_seconds":precision_overhead_seconds,
        "deployed_attribute_seconds":deployed_attribute_seconds,
        "prefix_seconds":prefix_seconds,"base_seconds":base_seconds,"attribute_seconds":attr_seconds,
        "peak_allocated_GiB":peak,"endpoint_windows":endpoint_windows,"density_windows":density}
    del cache
    return records,diagnostic,payload


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-name",required=True);ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--numeric-video",help="diagnostic smoke only; no performance evaluation")
    ap.add_argument("--adaptive",action="store_true",help="numerical feasibility diagnostic")
    ap.add_argument("--fp32-query",action="store_true",help="promote language query computation, keep native prefix")
    ap.add_argument("--checkpoint-query",action="store_true",help="recompute decoder activations during query backward")
    a=ap.parse_args()
    if a.numeric_video and not a.smoke:ap.error("numeric-video requires smoke")
    torch.manual_seed(0);out=ROOT/"runs/20261002_m1_attributor"/a.run_name;out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",handlers=[logging.FileHandler(out/"run.log"),logging.StreamHandler(sys.stdout)])
    logging.info("host %s",socket.gethostname());(out/"run.pid").write_text(str(os.getpid()))
    rows=load_manifest(ROOT/"data/omsl_v6_inputs/manifests/all_test.jsonl",["HateMM","HateClipSeg"])
    if a.numeric_video:
        rows=[r for r in rows if r["video_id"]==a.numeric_video];assert len(rows)==1
    elif a.smoke:rows=[r for ds in ("HateMM","HateClipSeg") for r in [v for v in rows if v["dataset"]==ds][:2]]
    asr={ds:load_asr(ds) for ds in ("HateMM","HateClipSeg")}
    j=Judge(MODEL);engine=Attributor(j);engine.adaptive=a.adaptive;engine.fp32=a.fp32_query
    engine.checkpoint_query=a.checkpoint_query
    import transformers
    config={**vars(a),"date":time.strftime("%Y-%m-%d"),"host":socket.gethostname(),"seed":0,"model":MODEL,
        "torch":torch.__version__,"transformers":transformers.__version__,"GT_in_reader":False,
        "code":"experiments/20261002_m1_attributor/{measure,attributor}.py; src/window_token_regions.py; local sources 2026-10-03, numerical repairs",
        "nodes":[16,32,64,128,256],"convergence_relative_l1":.05,"completeness_atol":.25,"completeness_rtol":.05,
        "video_question":VIDEO_QUESTION,"visual_question":yesno_question(0,1,0,8,"","visual"),
        "speech_question":yesno_question(0,1,0,8,"<text>","speech"),"frames":20,"window_seconds":8}
    (out/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    handles={};done={}
    for arm in ("base","attribute"):
        d=out/arm;d.mkdir(exist_ok=True);(d/"config.json").write_text(json.dumps({**config,"arm":arm},indent=2)+"\n")
        p=d/"predictions.jsonl";done[arm]={(r["dataset"],r["video_id"]) for r in map(json.loads,p.open())} if p.exists() else set()
        handles[arm]=p.open("a")
    assert done["base"]==done["attribute"],"partial arm records must be repaired before resume"
    checked=(out/"checks.jsonl").open("a");started=time.time()
    for k,row in enumerate(rows):
        key=row["dataset"],row["video_id"]
        if key in done["base"]:continue
        recs,check,payload=read_video(j,engine,row,asr[key[0]].get(key[1],[]),smoke=a.smoke)
        d=out/"tokens"/key[0];d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/(key[1]+".npz"),**payload)
        checked.write(json.dumps(check)+"\n");checked.flush()
        # Smoke keeps failed numerical trials for inspection. Full run stops before
        # admitting an unreliable score; it never drops an inconvenient video.
        if not a.smoke:assert check["numeric"]["numerical_pass"],(key,"quadrature failed",check["numeric"])
        for arm,r in recs.items():handles[arm].write(json.dumps(r)+"\n");handles[arm].flush()
        logging.info("progress %d/%d %s %s nodes=%s elapsed=%.1f peak_GiB=%.2f",k+1,len(rows),*key,
            check["numeric"]["accepted_nodes"],time.time()-started,check["peak_allocated_GiB"])
    for f in handles.values():f.close()
    checked.close();engine.close();logging.info("RUN_DONE videos=%d elapsed=%.1f",len(rows),time.time()-started)


if __name__=="__main__":main()
