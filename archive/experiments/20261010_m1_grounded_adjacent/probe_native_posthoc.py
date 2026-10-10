"""Post-hoc diagnostic (rule 10; NOT a declared selection): the diagnostics showed that windows where probe B
cites an in-window frame (rule B unflagged, about 20 % of windows) have a higher helpful rate than the flagged
ones. This evaluates the inverse of rule B (take the extra read only where probe B cites an in-window frame) on the
same stored reads, with six rate-matched random controls, to see whether even this label-informed choice of rule
would pass the declared gate. Reported as development-selected and post hoc."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from probe_native_analyze import load,hybrid,evaluate,metric,ROOT,EXP,DATASETS,ARMS,SEEDS,METRICS,W,FLOOR

def main():
    videos,_=load(False);out=EXP/'probe_posthoc';out.mkdir(exist_ok=True);preds={};rate={}
    for arm in ARMS:
        for ds in DATASETS:
            el=[w for vid in videos if vid['ds']==ds for w in vid['win'] if w[arm] is not None];rate[(ds,arm)]=sum(not w['flag_B'] for w in el)/max(1,len(el))
    for vi,vid in enumerate(videos):
        for arm in ARMS:
            preds.setdefault(f'inv_B_{arm}',[]).append(hybrid(vid['base'],[w[arm] if (w[arm] is not None and not w['flag_B']) else w['v'] for w in vid['win']],f'm1_c41r1_posthoc_inv_B_{arm}'))
            for seed in SEEDS:
                rng=np.random.default_rng([seed,vi]);r=rate[(vid['ds'],arm)]
                preds.setdefault(f'random_inv_B_{arm}_{seed}',[]).append(hybrid(vid['base'],[w[arm] if (w[arm] is not None and rng.random()<r) else w['v'] for w in vid['win']],f'm1_c41r1_posthoc_random_{arm}_{seed}'))
    for name,rr in preds.items():
        folder=out/name;folder.mkdir(exist_ok=True);(folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
        (folder/'config.json').write_text(json.dumps(dict(posthoc=name,built_from='probe_main records + placement_controls_main records',labels_used=False,declared=False),indent=2)+'\n')
    with ThreadPoolExecutor(8) as ex:
        for name in ex.map(lambda n:evaluate(n,out/n,out/'decoded'),sorted(preds)):print('EVALUATED',name,flush=True)
    r6=metric(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json');m={n:metric(out/'decoded'/n/'metrics.json') for n in preds};res={}
    for arm in ARMS:
        n=f'inv_B_{arm}';g=lambda x,ds,k:m[x][ds][k]-r6[ds][k];band={ds:[g(f'random_inv_B_{arm}_{s}',ds,W) for s in SEEDS] for ds in DATASETS}
        res[n]=dict(rate={ds:rate[(ds,arm)] for ds in DATASETS},gain_vs_r6={ds:{k:g(n,ds,k) for k in METRICS} for ds in DATASETS},random_band_within=band,
            above_every_random={ds:g(n,ds,W)-max(band[ds]) for ds in DATASETS},supported=bool(all(g(n,ds,W)-max(band[ds])>FLOOR[W] for ds in DATASETS) and all(g(n,ds,k)>-FLOOR[k] for ds in DATASETS for k in METRICS[:2])))
        print(n,'rate',{ds:round(res[n]['rate'][ds],2) for ds in DATASETS},'within vs r6',' / '.join(f"{res[n]['gain_vs_r6'][ds][W]:+.4f}" for ds in DATASETS),'above every random',' / '.join(f"{res[n]['above_every_random'][ds]:+.4f}" for ds in DATASETS),'supported',res[n]['supported'])
    (out/'summary.json').write_text(json.dumps(dict(scope='post-hoc, development-selected, not declared; rule chosen after seeing the label diagnostics',results=res),indent=2)+'\n');print('POSTHOC_DONE')

if __name__=='__main__':main()
