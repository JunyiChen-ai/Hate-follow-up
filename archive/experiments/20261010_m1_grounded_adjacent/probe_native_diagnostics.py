"""Candidate 41 revision 1, test error analysis (rule 10, development): does a flag from probe A or B mark the
windows where the extra read moves toward the label? Per rule x arm, on all windows and on the videos the within
metric uses (both labels present): flagged vs unflagged counts, positive share, helpful rate by the visual
direction of (extra read - native read), mean |dz|. Also: how many flag_a values change when numbers within
tolerance of a window bound are dropped (review 2026-10-11, finding 3), and what the flags correlate with
(number of in-window frames, sign of the native read). Files read: probe_main records, placement_controls_main
records, data/gt_4fps. Labels are used only here and in the evaluator."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from probe_native import SPEC
EXP=ROOT/'runs/20261010_m1_grounded_adjacent';STORED=ROOT/'runs/20261007_m1_streamingtom/placement_controls_main'
DATASETS=('HateMM','HateClipSeg');ARMS=('a','l','E');TOL=SPEC['match_tolerance']+1e-9


def main():
    gt={};mixed={}
    for ds in DATASETS:
        z=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True);gt[ds]={v:y for v,y in zip(z['video_ids'],z['y4'])};mixed[ds]={v for v,y in gt[ds].items() if 0<y.sum()<len(y)}
    rows=[]
    for ds in DATASETS:
        for f in sorted((EXP/'probe_main/records'/ds).glob('*.json')):
            b=json.loads(f.read_text());vid=b['predictions']['base']['video_id'];y=gt[ds][vid];st=json.loads((STORED/'records'/ds/f.name).read_text());ft=b['native']['frame_times']
            for t,w,s in zip(b['traces'],b['predictions']['base']['extra']['windows'],st['traces']):
                a0,a1=int(round(w['start']*4)),int(round(w['end']*4));lab=int(y[a0:a1].max()) if a1>a0 and a0<len(y) else 0
                # flag_a without numbers within tolerance of a bound
                nums=[x for x in t['numbers_a'] if not (abs(x-w['start'])<=TOL or abs(x-w['end'])<=TOL)]
                matched=sorted({u for u in ft if any(abs(x-u)<=TOL for x in nums)});truth=sorted(t['inside_times'])
                correct2=(matched==truth) if truth else (t['none_a'] and not matched)
                rows.append(dict(ds=ds,mixed=vid in mixed[ds],lab=lab,v=w['z_visual'],n_inside=len(truth),flag_A=bool(t['flag_a']),flag_B=bool(t['flag_b']),flag_A_nobound=not correct2,
                    none_a=t['none_a'],missed=t['missed_a'],extra=t['extra_a'],
                    a=s['adjacent_native']['z'] if s['adjacent_native'] else None,l=s['adjacent_local']['z'] if s['adjacent_local'] else None,E=s['stored_local_clean']))
    out={}
    for ds in DATASETS:
        R=[r for r in rows if r['ds']==ds];o=out[ds]={}
        o['flag_a_changes_without_bound_numbers']=sum(r['flag_A']!=r['flag_A_nobound'] for r in R);o['windows']=len(R)
        o['flag_a_rate_by_n_inside']={str(k):(float(np.mean([r['flag_A'] for r in R if r['n_inside']==k])),sum(r['n_inside']==k for r in R)) for k in sorted({r['n_inside'] for r in R})}
        o['flag_b_rate_by_sign_v']={'v>0':float(np.mean([r['flag_B'] for r in R if r['v']>0])),'v<=0':float(np.mean([r['flag_B'] for r in R if r['v']<=0]))}
        o['flag_a_rate_by_sign_v']={'v>0':float(np.mean([r['flag_A'] for r in R if r['v']>0])),'v<=0':float(np.mean([r['flag_A'] for r in R if r['v']<=0]))}
        o['probe_a_error_types_covered']={'none_but_frames':sum(r['none_a'] and r['n_inside']>0 for r in R),'missed_some':sum((not r['none_a']) and r['missed']>0 for r in R),'extra_frames':sum(r['extra']>0 for r in R)}
        for scope in ('all','mixed'):
            S=[r for r in R if (scope=='all' or r['mixed'])]
            for rule in ('A','B'):
                for arm in ARMS:
                    el=[r for r in S if r[arm] is not None and r[arm]!=r['v']]
                    cell={}
                    for fl in (True,False):
                        c=[r for r in el if r['flag_'+rule]==fl]
                        if not c:cell['flagged' if fl else 'unflagged']=dict(n=0);continue
                        helpful=[((r[arm]-r['v'])>0)==bool(r['lab']) for r in c]
                        cell['flagged' if fl else 'unflagged']=dict(n=len(c),positive_share=float(np.mean([r['lab'] for r in c])),helpful_rate=float(np.mean(helpful)),
                            mean_abs_dz=float(np.mean([abs(r[arm]-r['v']) for r in c])),helpful_abs_dz=float(sum(abs(r[arm]-r['v']) for r,h in zip(c,helpful) if h)),harmful_abs_dz=float(sum(abs(r[arm]-r['v']) for r,h in zip(c,helpful) if not h)))
                    o[f'{scope}:{rule}:{arm}']=cell
    (EXP/'probe_analysis/diagnostics.json').write_text(json.dumps(dict(scope='development-selected test error analysis (rule 10); labels only here and in the evaluator',results=out),indent=2)+'\n')
    for ds in DATASETS:
        o=out[ds];print(ds,'windows',o['windows'],'flag_a flips without bound numbers',o['flag_a_changes_without_bound_numbers'])
        print('  flag_a by n_inside',{k:(round(v[0],2),v[1]) for k,v in o['flag_a_rate_by_n_inside'].items()});print('  flag_a by sign v',{k:round(v,2) for k,v in o['flag_a_rate_by_sign_v'].items()},' flag_b by sign v',{k:round(v,2) for k,v in o['flag_b_rate_by_sign_v'].items()})
        print('  probe A error types',o['probe_a_error_types_covered'])
        for scope in ('all','mixed'):
            for rule in ('A','B'):
                for arm in ARMS:
                    c=o[f'{scope}:{rule}:{arm}'];f=c['flagged'];u=c['unflagged']
                    if f.get('n') and u.get('n'):print(f"  {scope:5s} {rule} {arm}: flagged n={f['n']:4d} pos={f['positive_share']:.2f} helpful={f['helpful_rate']:.3f} |dz|={f['mean_abs_dz']:.2f} | unflagged n={u['n']:4d} pos={u['positive_share']:.2f} helpful={u['helpful_rate']:.3f} |dz|={u['mean_abs_dz']:.2f}")
    print('C41R1_DIAGNOSTICS_DONE')


if __name__=='__main__':main()
