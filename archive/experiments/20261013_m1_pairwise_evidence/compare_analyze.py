"""Candidate 42 analysis: binding checks, aggregation of the pairwise comparisons into per-window values, the declared
compositions and controls, the sole evaluator with the fixed r6 decoder, the declared gates, and the rule-10 diagnostics
(the only place where test labels are read, together with the evaluator)."""
import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compare import ROOT,EXP,PLACEMENT,DATASETS,SPEC,IMPLEMENTATION,CHOICES,selected_rows,windows_for,edges_for,d_from_logprobs

R6=ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl'
R6_METRICS=ROOT/'runs/20260926_twolevel/r6_bma/metrics.json'
REFERENCE_METRICS={'adjacent_local':ROOT/'runs/20261007_m1_streamingtom/placement_controls_main_decoded/adjacent_local/metrics.json',
    'adjacent_native':ROOT/'runs/20261007_m1_streamingtom/placement_controls_main_decoded/adjacent_native/metrics.json',
    'local_clean':ROOT/'runs/20261007_m1_streamingtom/local_controls_main_decoded/local_clean/metrics.json'}
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')
W='within_video_macro_ROC_AUC'
FLOOR={'frame_ROC_AUC':.005,'frame_PR_AUC':.005,W:.01}
SEEDS=6
SELECTION=EXP/'selection';DECODED=EXP/'selection_decoded';OUT=EXP/'analysis'


def solve(n,edges,values):
    """u = argmin_{sum u = 0} sum_(i,j) (u_i - u_j - l_ij)^2 through the graph Laplacian pseudo-inverse."""
    if n==0:return []
    L=np.zeros((n,n));b=np.zeros(n)
    for (i,j),l in zip(edges,values):
        L[i,i]+=1;L[j,j]+=1;L[i,j]-=1;L[j,i]-=1;b[i]+=l;b[j]-=l
    u=np.linalg.pinv(L)@b;return (u-u.mean()).tolist()


def records(path):
    rows={}
    for line in path.open():
        r=json.loads(line);key=r['dataset'],r['video_id'];assert key not in rows;rows[key]=r
    return rows


def metric(path):return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def selfcheck():
    rng=np.random.default_rng(0);checks={}
    for n in list(range(1,61))+[90,120]:
        edges=edges_for(n,7);assert all(i<j for i,j in edges) and len(edges)==len(set(edges))
        expected=0 if n<=1 else n*(n-1)//2 if n<=SPEC['complete_graph_max_windows'] else min(2*n,n*(n-1)//2)
        assert len(edges)==expected,(n,len(edges),expected)
        if n>=2:
            adj=np.zeros((n,n));
            for i,j in edges:adj[i,j]=adj[j,i]=1
            reach={0};frontier=[0]
            while frontier:
                k=frontier.pop();new=[m for m in range(n) if adj[k,m] and m not in reach];reach.update(new);frontier+=new
            assert len(reach)==n,'graph not connected'
            truth=rng.normal(size=n);truth-=truth.mean();u=solve(n,edges,[truth[i]-truth[j] for i,j in edges])
            assert np.allclose(u,truth,atol=1e-8),('exact recovery',n)
        checks[n]=len(edges)
    assert edges_for(10,3)!=edges_for(10,4) and edges_for(10,3)==edges_for(10,3)
    d,pc=d_from_logprobs(np.log([.6,.3,.1]));assert abs(d-math.log(.65/.35))<1e-12 and abs(pc-.1)<1e-12
    assert abs(d_from_logprobs(np.log([.3,.6,.1]))[0]+d)<1e-12
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'selfcheck.json').write_text(json.dumps(dict(PASS=True,edges_by_windows=checks),indent=2)+'\n');print('SELFCHECK_PASS')


def load_bundles(smoke):
    root=EXP/('smoke' if smoke else 'main');cfg=json.loads((root/'config.json').read_text())
    assert cfg['spec']==SPEC and cfg['implementation']==IMPLEMENTATION and cfg['GT_read'] is False and cfg['smoke']==smoke
    out=[]
    for vi,row in selected_rows(smoke):
        b=json.loads((root/'records'/row['dataset']/(row['video_id']+'.json')).read_text());assert b['version']==SPEC['version'] and b['implementation']==IMPLEMENTATION and b['spec']==SPEC
        out.append((vi,row,b))
    return root,out


def prepare(smoke):
    root,bundles=load_bundles(smoke);r6=records(R6)
    summary={ds:dict(videos=0,windows=0,edges=0,comparisons_with_images=0,covered_local=0,swap_agree=0,swap_defined=0,pC=[],pA=[],pB=[],abs_l=[],s=[],seconds=dict(prefix=0.,native_visual=0.,native_speech=0.,compare=0.),peak_GiB=0.,
        nonzero_l=0) for ds in DATASETS}
    for vi,row,b in bundles:
        key=row['dataset'],row['video_id'];s=summary[key[0]];windows=windows_for(row,[]);n=len(windows);assert b['checks']['video_index']==vi and b['checks']['windows']==n
        base=b['predictions']['base'];assert base['extra']['z_video']==r6[key]['extra']['z_video'] and base['extra']['windows']==r6[key]['extra']['windows'] and base['score_curve']==r6[key]['score_curve']
        place=json.loads((PLACEMENT/'records'/key[0]/(key[1]+'.json')).read_text());frames=place['frames']
        for w,pw,bw,t in zip(windows,base['extra']['windows'],b['windows'],place['traces']):
            assert bw['i']==w['i']==t['i'] and bw['bounds']==[w['start'],w['end']]==t['bounds'] and bw['native_visual']==pw['z_visual']==t['native_visual'] and bw.get('native_speech')==pw.get('z_speech')
            assert bw['LOCAL']==[frames[i]['time'] for i in t['LOCAL']]
        edges=edges_for(n,vi);assert [(e['i'],e['j']) for e in b['edges']]==edges
        for e in b['edges']:
            d1,p1=d_from_logprobs(e['lp_ij']);d2,p2=d_from_logprobs(e['lp_ji']);assert e['d_ij']==d1 and e['d_ji']==d2 and e['pC_ij']==p1 and e['pC_ji']==p2 and e['l_ij']==(d1-d2)/2 and e['s_ij']==(d1+d2)/2
            assert e['images']==[len(b['windows'][e['i']]['LOCAL'])+len(b['windows'][e['j']]['LOCAL'])]*2
            s['edges']+=1;s['pC']+=[p1,p2];s['abs_l'].append(abs(e['l_ij']));s['s'].append(e['s_ij']);s['nonzero_l']+=e['l_ij']!=0
            for lp in (e['lp_ij'],e['lp_ji']):
                p=np.exp(np.asarray(lp));p=p/p.sum();s['pA'].append(p[0]);s['pB'].append(p[1])
            if d1!=0 and d2!=0:s['swap_defined']+=1;s['swap_agree']+=int(np.sign(d1)==-np.sign(d2))
        c=b['checks'];assert c['edges']==len(edges) and c['actual_forwards']==base['calls']+2*len(edges) and c['actual_vision']==1+c['comparisons_with_images']
        s['videos']+=1;s['windows']+=n;s['comparisons_with_images']+=c['comparisons_with_images'];s['covered_local']+=c['covered_local'];s['peak_GiB']=max(s['peak_GiB'],c['peak_GiB'])
        for k in s['seconds']:s['seconds'][k]+=c['times'][k]
    for ds,s in summary.items():
        s['swap_agreement']=s['swap_agree']/s['swap_defined'] if s['swap_defined'] else None;s['mean_pC']=float(np.mean(s.pop('pC'))) if s['edges'] else None
        s['letter_marginals']=dict(A=float(np.mean(s.pop('pA'))),B=float(np.mean(s.pop('pB')))) if s['edges'] else None
        s['mean_abs_l']=float(np.mean(s.pop('abs_l'))) if s['edges'] else None;s['mean_position_preference_s']=float(np.mean(s.pop('s'))) if s['edges'] else None
    out=root.parent/(root.name+'_analysis');out.mkdir(exist_ok=True)
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(dict(PASS=True,GT_read=False,coverage=len(bundles),base_equals_r6=True,datasets=summary),indent=2)+'\n')
    print(json.dumps(summary,indent=1));print('PREPARE_PASS',flush=True)


def compose(base,visual,name):
    p=json.loads(json.dumps(base));ww=p['extra']['windows'];assert len(ww)==len(visual)
    for w,v in zip(ww,visual):
        w['z_visual']=float(v);w['z']=max(w['z_visual'],w['z_speech']) if 'z_speech' in w else w['z_visual']
    idx=np.clip(((np.arange(math.ceil(float(p['duration'])*4))+.5)/4//SPEC['window_seconds']).astype(int),0,len(ww)-1)
    p['score_curve']=np.asarray([w['z'] for w in ww])[idx].tolist();p['method']=name;p['code_path']='experiments/20261013_m1_pairwise_evidence/compare_analyze.py';return p


def per_video(bundles):
    """Per video: v, s, l (adjacent_local or v), E, a, edges, l_ij, u, u_perm[k]."""
    out=[]
    for vi,row,b in bundles:
        key=row['dataset'],row['video_id'];place=json.loads((PLACEMENT/'records'/key[0]/(key[1]+'.json')).read_text());n=len(b['windows'])
        v=[w['native_visual'] for w in b['windows']];s=[w.get('native_speech') for w in b['windows']]
        l=[t['adjacent_local']['z'] if t['adjacent_local'] else t['native_visual'] for t in place['traces']]
        E=[t['stored_local_clean'] if t.get('stored_local_clean') is not None else t['native_visual'] for t in place['traces']]
        a=[t['adjacent_native']['z'] if t['adjacent_native'] else t['native_visual'] for t in place['traces']]
        edges=[(e['i'],e['j']) for e in b['edges']];lij=[e['l_ij'] for e in b['edges']];u=solve(n,edges,lij) if n else []
        perms=[]
        for k in range(SEEDS):
            rng=np.random.default_rng([100+k,vi]);perms.append(solve(n,edges,[lij[x] for x in rng.permutation(len(lij))]) if n else [])
        out.append(dict(vi=vi,row=row,key=key,base=b['predictions']['base'],v=v,s=s,l=l,E=E,a=a,edges=edges,lij=lij,u=u,perms=perms,
            fused=[w['z'] for w in b['predictions']['base']['extra']['windows']]))
    return out


def compositions(video):
    l=np.asarray(video['l']);u=np.asarray(video['u']);out={'full_l':l.tolist(),'cmp_order':(l.mean()+u).tolist() if len(l) else [],'cmp_sum':(l+u).tolist()}
    for k,up in enumerate(video['perms']):
        up=np.asarray(up);out[f'perm_order_{k}']=(l.mean()+up).tolist() if len(l) else [];out[f'perm_sum_{k}']=(l+up).tolist()
    return out


def build():
    root,bundles=load_bundles(False);assert json.loads((root.parent/(root.name+'_analysis')/'alignment.json').read_text())['PASS'] is True
    videos=per_video(bundles);names=list(compositions(videos[0]).keys());rows={name:[] for name in names}
    for video in videos:
        for name,values in compositions(video).items():rows[name].append(compose(video['base'],values,'m1_c42_'+name))
    for name,rr in rows.items():
        folder=SELECTION/name;folder.mkdir(parents=True,exist_ok=True);(folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
        (folder/'config.json').write_text(json.dumps(dict(selection=name,spec=SPEC,implementation=IMPLEMENTATION,GT_read=False,source='experiments/20261013_m1_pairwise_evidence/compare_analyze.py;2026-10-13'),indent=2)+'\n')
    (SELECTION/'names.json').write_text(json.dumps(names)+'\n');print('BUILD_DONE',names,flush=True)


def evaluate_one(name):
    folder=SELECTION/name
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(folder/'predictions.jsonl'),'--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(folder/'metrics.json')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(folder),'--datasets',*DATASETS,'--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2',
        '--out-root',str(DECODED),'--tag',name],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    return name


def evaluate(threads):
    names=json.loads((SELECTION/'names.json').read_text())
    with ThreadPoolExecutor(threads) as pool:
        for name in pool.map(evaluate_one,names):print('EVALUATED',name,flush=True)
    print('EVALUATE_DONE',flush=True)


def window_labels(y4,windows):
    out=[]
    for w in windows:
        a,b=int(round(w['start']*4)),min(int(round(w['end']*4)),len(y4));seg=y4[a:b];out.append(1 if seg.size and float(seg.mean())>.5 else 0)
    return out


def boot_gain(video_ids,hit_new,hit_base,n_boot=2000):
    vids=np.asarray(video_ids);uniq=np.unique(vids);idx={v:np.where(vids==v)[0] for v in uniq};rng=np.random.default_rng(0);diffs=[]
    for _ in range(n_boot):
        pick=rng.choice(len(uniq),size=len(uniq),replace=True);sel=np.concatenate([idx[uniq[p]] for p in pick]);diffs.append(hit_new[sel].mean()-hit_base[sel].mean())
    lo,hi=np.percentile(diffs,[2.5,97.5]);return [float(lo),float(hi)]


def diagnostics(videos):
    """Rule 10: test labels read here only. Pair accuracy of the comparison against the GT label difference on graph edges
    whose windows carry different labels, versus the differences of the pointwise reads on the same pairs."""
    from scipy.stats import spearmanr
    gt={}
    for ds in DATASETS:
        z=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True)
        for vid,y in zip(z['video_ids'],z['y4']):gt[(ds,str(vid))]=np.asarray(y,dtype=float)
    out={}
    for ds in DATASETS:
        pairs=[];rho=[];sd=[]
        for video in videos:
            if video['key'][0]!=ds or video['key'] not in gt:continue
            labels=window_labels(gt[video['key']],windows_for(video['row'],[]));n=len(labels)
            if n>=3 and np.std(video['l'])>0 and np.std(video['u'])>0:rho.append(spearmanr(video['u'],video['l']).correlation);sd.append((float(np.std(video['u'])),float(np.std(video['l']))))
            for (i,j),lij in zip(video['edges'],video['lij']):
                if labels[i]==labels[j]:continue
                truth=1. if labels[i]>labels[j] else -1.
                pairs.append(dict(video_id=video['key'][1],truth=truth,cmp=lij,l=video['l'][i]-video['l'][j],v=video['v'][i]-video['v'][j],E=video['E'][i]-video['E'][j],a=video['a'][i]-video['a'][j],
                    both_positive=video['fused'][i]>0 and video['fused'][j]>0))
        e={'n_pairs':len(pairs),'n_videos':len({p['video_id'] for p in pairs}),'spearman_u_vs_l_mean':float(np.mean(rho)) if rho else None,'spearman_videos':len(rho),
            'sd_u_mean':float(np.mean([x for x,_ in sd])) if sd else None,'sd_l_within_video_mean':float(np.mean([y for _,y in sd])) if sd else None}
        for subset,P in (('all',pairs),('both_fused_positive',[p for p in pairs if p['both_positive']])):
            if not P:e[subset]=dict(n_pairs=0);continue
            truth=np.array([p['truth'] for p in P]);acc={k:float(np.mean(np.sign([p[k] for p in P])==truth)) for k in ('cmp','l','v','E','a')}
            hit_c=(np.sign([p['cmp'] for p in P])==truth).astype(float);hit_l=(np.sign([p['l'] for p in P])==truth).astype(float)
            e[subset]=dict(n_pairs=len(P),n_videos=len({p['video_id'] for p in P}),accuracy=acc,gain_cmp_over_l=acc['cmp']-acc['l'],gain_cmp_over_l_ci95_videos=boot_gain([p['video_id'] for p in P],hit_c,hit_l),
                sign_agreement_cmp_vs_l=float(np.mean(np.sign([p['cmp'] for p in P])==np.sign([p['l'] for p in P]))),ties_cmp=int(np.sum(np.array([p['cmp'] for p in P])==0)))
        out[ds]=e
    return out


def report():
    root,bundles=load_bundles(False);videos=per_video(bundles);names=json.loads((SELECTION/'names.json').read_text())
    m={'r6':metric(R6_METRICS)};m.update({k:metric(p) for k,p in REFERENCE_METRICS.items()});m.update({name:metric(DECODED/name/'metrics.json') for name in names})
    def diff(x,y):return {ds:{k:m[x][ds][k]-m[y][ds][k] for k in METRICS} for ds in DATASETS}
    gain={name:diff(name,'r6') for name in names}
    binding=dict(full_l_equals_adjacent_local_decoded=all(abs(m['full_l'][ds][k]-m['adjacent_local'][ds][k])<1e-9 for ds in DATASETS for k in METRICS))
    def rule8(d):
        no_loss=all(d[ds][k]>-FLOOR[k] for ds in DATASETS for k in METRICS);common=[k for k in METRICS if all(d[ds][k]>=.01 for ds in DATASETS)]
        return dict(no_loss_beyond_noise=no_loss,common_gain_metrics=common,passed=bool(common and no_loss))
    verdicts={}
    for name,kind in (('cmp_order','order'),('cmp_sum','sum')):
        vs_l=diff(name,'full_l');band={ds:[gain[f'perm_{kind}_{k}'][ds][W] for k in range(SEEDS)] for ds in DATASETS}
        mech=dict(within_vs_full_l={ds:vs_l[ds][W] for ds in DATASETS},pooled_vs_full_l={ds:{k:vs_l[ds][k] for k in METRICS[:2]} for ds in DATASETS},
            supported=all(vs_l[ds][W]>=.01 for ds in DATASETS) and all(vs_l[ds][k]>-FLOOR[k] for ds in DATASETS for k in METRICS[:2]))
        verdicts[name]=dict(performance_rule8_vs_r6=rule8(gain[name]),mechanism_vs_full_l=mech,permutation_band_within_vs_r6=band,
            within_above_every_permutation={ds:gain[name][ds][W]-max(band[ds]) for ds in DATASETS},vs_full_E={ds:diff(name,'local_clean')[ds][W] for ds in DATASETS},vs_adjacent_native_within={ds:diff(name,'adjacent_native')[ds][W] for ds in DATASETS})
    alignment=json.loads((root.parent/(root.name+'_analysis')/'alignment.json').read_text())['datasets']
    result=dict(scope='development-selected; declared in README before the run; sole evaluator, fixed r6 decoder; primary cmp_order, secondary cmp_sum; GT read only in the evaluator and in diagnostics',
        r6={ds:{k:m['r6'][ds][k] for k in METRICS} for ds in DATASETS},metrics={name:{ds:{k:m[name][ds][k] for k in METRICS} for ds in DATASETS} for name in m if name!='r6'},gain_vs_r6=gain,binding=binding,
        comparisons={ds:{k:alignment[ds][k] for k in ('edges','comparisons_with_images','swap_agreement','mean_pC','letter_marginals','mean_abs_l','mean_position_preference_s','seconds','peak_GiB')} for ds in DATASETS},
        verdicts=verdicts,primary=dict(name='cmp_order',performance=verdicts['cmp_order']['performance_rule8_vs_r6']['passed'],mechanism=verdicts['cmp_order']['mechanism_vs_full_l']['supported']),
        diagnostics=diagnostics(videos),sources={'r6':str(R6_METRICS.relative_to(ROOT)),**{k:str(p.relative_to(ROOT)) for k,p in REFERENCE_METRICS.items()},**{name:str((DECODED/name/'metrics.json').relative_to(ROOT)) for name in names}})
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdicts=verdicts,diagnostics=result['diagnostics'],comparisons=result['comparisons']),indent=1));print('REPORT_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('selfcheck','prepare','build','evaluate','report'),required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--threads',type=int,default=8);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();done=False
    try:
        if a.stage=='selfcheck':selfcheck()
        elif a.stage=='prepare':prepare(a.smoke)
        elif a.stage=='build':build()
        elif a.stage=='evaluate':evaluate(a.threads)
        else:report()
        done=True
    finally:
        n=1;tag=a.stage+('_smoke' if a.smoke else '')
        while (OUT/f'{tag}_attempt_{n:04d}.json').exists():n+=1
        (OUT/f'{tag}_attempt_{n:04d}.json').write_text(json.dumps(dict(completed=done,elapsed_seconds=time.perf_counter()-start,stage=a.stage),indent=2)+'\n')


if __name__=='__main__':main()
