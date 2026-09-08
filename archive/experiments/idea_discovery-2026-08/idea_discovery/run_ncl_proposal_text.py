#!/usr/bin/env python3
"""Proposal-conditioned semantic evidence for bicameral NCL (GT blind)."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_cwa_text_warrants import POLICY, QUESTION
from scripts.idea_discovery.run_melt import MLLM, sanitized_cohort
from scripts.idea_discovery.run_ncl_tribunal import load, overlap_text
from scripts.idea_discovery.run_transcript_existence import make_prompt, score
from scripts.idea_discovery.run_visual_temporal_canvas import valid_chunks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',type=Path,required=True)
    ap.add_argument('--proposals',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct');ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--limit',type=int,default=0);ap.add_argument('--char-cap',type=int,default=1600);a=ap.parse_args()
    if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    cohort=sanitized_cohort(a.cohort);cohort=cohort[:a.limit] if a.limit else cohort
    proposals,all_chunks=load(a.proposals),transcript_rows();model=MLLM(a.model)
    null=float(score(model,[make_prompt('',POLICY,QUESTION,a.char_cap)],1)[0])
    config={'version':'ncl_proposal_text_v1','model':a.model,'null_log_odds':null,
            'decision_threshold':0.0,'cohort_sha256':hashlib.sha256(a.cohort.read_bytes()).hexdigest(),
            'proposals_sha256':hashlib.sha256(a.proposals.read_bytes()).hexdigest(),
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'gt_access':False}
    with a.out.open('x',encoding='utf8') as f:
      for i,row in enumerate(cohort,1):
       key=(row['dataset'],row['video_id']);duration=float(row['duration'])
       chunks,rejected=valid_chunks(all_chunks.get(key,[]),duration);examples=[];meta=[]
       for p in proposals[key]['proposals']:
        start,end=float(p['start']),float(p['end']);local=overlap_text(chunks,start,end,a.char_cap)
        examples.append(make_prompt(local,POLICY,QUESTION,a.char_cap));meta.append({'rank':int(p['rank']),'start':start,'end':end,'text_chars':len(local)})
       values=score(model,examples,a.batch_size)
       observations=[{**m,'log_odds':float(z),'supported':float(z)>0} for m,z in zip(meta,values)]
       f.write(json.dumps({'dataset':key[0],'video_id':key[1],'duration':duration,
                           'observations':observations,'invalid_asr_spans_rejected':rejected,
                           'config':config},ensure_ascii=False)+'\n');f.flush()
       print(json.dumps({'i':i,'n':len(cohort),'key':key,'supported':sum(x['supported'] for x in observations)}),flush=True)
    print(json.dumps({'videos':len(cohort),'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),'calls':model.calls}))

if __name__=='__main__':main()
