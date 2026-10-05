#!/usr/bin/env python3
"""NoGT binding and R1 compatibility checks against complete actual B records."""
import json
import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.mllm_renderer import cpu_renderer
from src.mllm_judge import yesno_question
from handle_inputs import CACHE,selected_rows
from handle_measure import question,usable_record


def main():
    j=cpu_renderer();count={};videos=0
    for row in selected_rows(False):
        ds=row['dataset'];vid=row['video_id'];videos+=1
        metadata=json.loads((CACHE/ds/(vid+'.json')).read_text())
        detail=json.loads((ROOT/'runs/20261005_m1_program/r1_handles_full_main/details'/ds/(vid+'.json')).read_text())
        c=detail['native_conversation'];n=len(metadata['windows'])
        totals=count.setdefault(ds,dict(windows=0,visual_applied=0,speech_applied=0,visual_fallback=0,speech_fallback=0,R1_queries_exact=0))
        for i,(w,t) in enumerate(zip(metadata['windows'],detail['traces'])):
            totals['windows']+=1
            for kind in ('visual','speech'):
                old=t['branches'][kind]
                if not old['available']:continue
                args=(w['execution'],i,n,t['start'],t['end'],t['native_body'],kind)
                original,rec=question(*args,1)
                assert original==old['question'] and rec==old['record']
                ids,text=j.branch_ids(c['msgs'],original,c['history'],head_text=c['head'])
                assert ids==old['suffix_ids'] and text==old['suffix_text'];totals['R1_queries_exact']+=1
                revised,rec2=question(*args,2);assert rec2==rec
                active=usable_record(rec2,kind)
                if active:assert revised==original;totals[kind+'_applied']+=1
                else:
                    assert revised==yesno_question(i,n,t['start'],t['end'],t['native_body'],kind)
                    totals[kind+'_fallback']+=1
                ids,text=j.branch_ids(c['msgs'],revised,c['history'],head_text=c['head']);assert ids and text
    assert videos==333 and all(v['visual_applied']>0 and v['visual_fallback']>0 and v['speech_fallback']>0 for v in count.values())
    out=ROOT/'runs/20261005_m1_program/available_cpu_checks';out.mkdir(exist_ok=True)
    result=dict(PASS=True,GT_read=False,videos=videos,datasets=count,scope='source/query/token binding only; no model forwards or performance')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
