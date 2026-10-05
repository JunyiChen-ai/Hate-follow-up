#!/usr/bin/env python3
"""Complete immutable-source/query replay; R1/R2 compatibility and R3 availability."""
import copy
import json
from handle_measure import available_field,question,usable_record
from handle_inputs import ROOT,CACHE,selected_rows
from src.mllm_renderer import cpu_renderer
from src.mllm_judge import yesno_question


def main():
    tests={'UNKNOWN':False,' UNKNOWN action:UNKNOWN':False,'unknown, vehicle visible':False,'Unknown: no target':False,'UNKNOWN.target':False,'UNKNOWNish action':True,'Person holding a sign':True,'not UNKNOWN':True}
    for value,expected in tests.items():assert available_field(value)==expected
    j=cpu_renderer();totals={};videos=0
    for row in selected_rows(False):
        ds,vid=row['dataset'],row['video_id'];videos+=1
        m=json.loads((CACHE/ds/(vid+'.json')).read_text());snapshot=copy.deepcopy(m)
        dd={r:json.loads((ROOT/f'runs/20261005_m1_program/r{r}_handles_full_main/details'/ds/(vid+'.json')).read_text()) for r in (1,2)}
        c=dd[1]['native_conversation'];assert c==dd[2]['native_conversation'];n=len(m['windows'])
        t=totals.setdefault(ds,dict(windows=0,R1_R2_exact_queries=0,visual_applied=0,visual_fallback=0,speech_applied=0,speech_fallback=0,normalized_fields=0))
        for i,w in enumerate(m['windows']):
            a=dd[1]['traces'][i];t['windows']+=1
            for kind in ('visual','speech'):
                if not a['branches'][kind]['available']:continue
                args=(w['execution'],i,n,a['start'],a['end'],a['native_body'],kind)
                for r in (1,2):
                    q,rec=question(*args,r);old=dd[r]['traces'][i]['branches'][kind]
                    assert q==old['question'] and rec==old['record']
                    ids,text=j.branch_ids(c['msgs'],q,c['history'],head_text=c['head']);assert ids==old['suffix_ids'] and text==old['suffix_text']
                    t['R1_R2_exact_queries']+=1
                q,rec=question(*args,3);active=usable_record(rec,kind,3)
                t[kind+('_applied' if active else '_fallback')]+=1
                if not active:assert q==yesno_question(i,n,a['start'],a['end'],a['native_body'],kind)
                for original,new in zip(question(*args,1)[1]['evidence'],rec['evidence']):
                    if new['value']['kind']=='action':
                        for field in ('actor','action','target'):
                            changed=original['value'][field]!=new['value'][field];t['normalized_fields']+=changed
                            assert new['value'][field]==(original['value'][field] if available_field(original['value'][field]) else 'UNKNOWN')
                ids,text=j.branch_ids(c['msgs'],q,c['history'],head_text=c['head']);assert ids and text
        assert m==snapshot,'source mutated'
    assert videos==333 and all(t['visual_applied']>0 and t['visual_fallback']>0 and t['normalized_fields']>0 for t in totals.values())
    out=ROOT/'runs/20261005_m1_program/resolved_cpu_checks';out.mkdir(exist_ok=True)
    result=dict(PASS=True,GT_read=False,videos=videos,predicate_cases=len(tests),datasets=totals,source_immutable=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
