"""Candidate 42 rule-10 test error analysis (2026-10-13, after the main run): where does the comparison lose against the
pointwise read with the same frames? (1) decoded per-video within AUC of full_l / cmp_order / cmp_sum on the within
videos, split by the video's swap agreement; (2) pair accuracy on label-different edges split by the edge's own
consistency (swap agreement, pC); (3) ordering accuracy over ALL label-different window pairs of the within videos for
u, l, v and l+u (the aggregated ordering, before the decoder); (4) graph density: u from the distance-1 ring edges only
(half the edges), composed as cmp_order_d1 and decoded. Labels are read here and in the evaluator only."""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compare import ROOT,EXP,DATASETS,SPEC
from compare_analyze import load_bundles,per_video,solve,compose,window_labels,windows_for,evaluate_one,SELECTION,DECODED,metric

OUT=EXP/'error_analysis'


def gt_all():
    gt={}
    for ds in DATASETS:
        z=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True)
        for vid,y in zip(z['video_ids'],z['y4']):gt[(ds,str(vid))]=np.asarray(y,dtype=float)
    return gt


def decoded_curves(name):
    out={}
    for line in (DECODED/name/'predictions.jsonl').open():
        r=json.loads(line);out[(r['dataset'],r['video_id'])]=np.asarray(r['score_curve'],dtype=float)
    return out


def distance1_edges(n,vi):
    if n<=1:return []
    if n<=SPEC['complete_graph_max_windows']:return [(i,j) for i in range(n) for j in range(i+1,n)]
    perm=np.random.default_rng([SPEC['seed'],vi]).permutation(n).tolist();edges=set()
    for k in range(n):
        a,b=perm[k],perm[(k+1)%n];edges.add((min(a,b),max(a,b)))
    return sorted(edges)


def main():
    OUT.mkdir(parents=True,exist_ok=True);root,bundles=load_bundles(False);videos=per_video(bundles);gt=gt_all()
    by_key={v['key']:v for v in videos};edge_detail={b['predictions']['base']['video_id']:b['edges'] for _,_,b in bundles}
    # (4) density: cmp_order from distance-1 edges only
    rows=[]
    for video in videos:
        n=len(video['l']);lmap={e:x for e,x in zip(video['edges'],video['lij'])};e1=distance1_edges(n,video['vi']);assert all(e in lmap for e in e1)
        u1=solve(n,e1,[lmap[e] for e in e1]) if n else [];l=np.asarray(video['l']);rows.append(compose(video['base'],(l.mean()+np.asarray(u1)).tolist() if n else [],'m1_c42_cmp_order_d1'))
        video['u1']=u1
    folder=SELECTION/'cmp_order_d1';folder.mkdir(parents=True,exist_ok=True);(folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (folder/'config.json').write_text(json.dumps(dict(selection='cmp_order_d1',note='error analysis: distance-1 ring edges only (half the edges)',GT_read=False),indent=2)+'\n')
    evaluate_one('cmp_order_d1')
    m={name:metric(DECODED/name/'metrics.json') for name in ('full_l','cmp_order','cmp_sum','cmp_order_d1')};r6=metric(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    density={ds:{name:m[name][ds]['within_video_macro_ROC_AUC']-r6[ds]['within_video_macro_ROC_AUC'] for name in m} for ds in DATASETS}
    # (1) per-video decoded within by swap agreement; (2) edge accuracy by consistency; (3) all-pairs ordering accuracy
    curves={name:decoded_curves(name) for name in ('full_l','cmp_order','cmp_sum')};result={}
    for ds in DATASETS:
        pv=[];edge_rows=[];pair_rows=[]
        for video in videos:
            if video['key'][0]!=ds or video['key'] not in gt:continue
            y=gt[video['key']];n_frames=min(len(y),*(len(curves[k][video['key']]) for k in curves))
            if y[:n_frames].min()==y[:n_frames].max():continue
            E=edge_detail[video['key'][1]];cons=[(np.sign(e['d_ij'])==-np.sign(e['d_ji'])) and e['d_ij']!=0 and e['d_ji']!=0 for e in E];agree=float(np.mean(cons)) if cons else None
            w={k:float(roc_auc_score(y[:n_frames],curves[k][video['key']][:n_frames])) for k in curves}
            pv.append(dict(video_id=video['key'][1],swap_agreement=agree,windows=len(video['l']),**w))
            labels=window_labels(y,windows_for(video['row'],[]));l=np.asarray(video['l']);u=np.asarray(video['u']);v=np.asarray(video['v']);lu=l+u;u1=np.asarray(video['u1'])
            for e,c in zip(E,cons):
                i,j=e['i'],e['j']
                if labels[i]==labels[j]:continue
                truth=1. if labels[i]>labels[j] else -1.;pc=(e['pC_ij']+e['pC_ji'])/2
                edge_rows.append(dict(video_id=video['key'][1],consistent=bool(c),low_pC=pc<.5,hit_cmp=float(np.sign(e['l_ij'])==truth),hit_l=float(np.sign(l[i]-l[j])==truth)))
            n=len(labels);acc=dict(u=[],l=[],v=[],lu=[],u1=[])
            for i in range(n):
                for j in range(i+1,n):
                    if labels[i]==labels[j]:continue
                    truth=1. if labels[i]>labels[j] else -1.
                    for k,arr in (('u',u),('l',l),('v',v),('lu',lu),('u1',u1)):acc[k].append(float(np.sign(arr[i]-arr[j])==truth))
            if acc['u']:pair_rows.append({k:float(np.mean(x)) for k,x in acc.items()}|dict(video_id=video['key'][1],n_pairs=len(acc['u'])))
        med=float(np.median([p['swap_agreement'] for p in pv if p['swap_agreement'] is not None]))
        groups={}
        for gname,sel in (('all',pv),('swap_agreement_above_median',[p for p in pv if p['swap_agreement'] is not None and p['swap_agreement']>med]),('swap_agreement_at_or_below_median',[p for p in pv if p['swap_agreement'] is None or p['swap_agreement']<=med])):
            groups[gname]=dict(videos=len(sel),mean_within={k:float(np.mean([p[k] for p in sel])) for k in curves},mean_cmp_order_minus_full_l=float(np.mean([p['cmp_order']-p['full_l'] for p in sel])),
                mean_cmp_sum_minus_full_l=float(np.mean([p['cmp_sum']-p['full_l'] for p in sel])),videos_cmp_order_better=int(sum(p['cmp_order']>p['full_l'] for p in sel)),mean_swap_agreement=float(np.mean([p['swap_agreement'] for p in sel if p['swap_agreement'] is not None])))
        edges={}
        for gname,sel in (('all',edge_rows),('consistent',[r for r in edge_rows if r['consistent']]),('inconsistent',[r for r in edge_rows if not r['consistent']]),('low_pC',[r for r in edge_rows if r['low_pC']]),('high_pC',[r for r in edge_rows if not r['low_pC']])):
            edges[gname]=dict(n_pairs=len(sel),acc_cmp=float(np.mean([r['hit_cmp'] for r in sel])) if sel else None,acc_l=float(np.mean([r['hit_l'] for r in sel])) if sel else None)
        pairs=dict(videos=len(pair_rows),pairs=int(sum(p['n_pairs'] for p in pair_rows)),macro_accuracy={k:float(np.mean([p[k] for p in pair_rows])) for k in ('u','l','v','lu','u1')},
            pooled_accuracy={k:float(np.sum([p[k]*p['n_pairs'] for p in pair_rows])/sum(p['n_pairs'] for p in pair_rows)) for k in ('u','l','v','lu','u1')})
        result[ds]=dict(swap_agreement_median=med,per_video_within=groups,edge_accuracy_by_consistency=edges,all_pairs_ordering=pairs,density_within_vs_r6=density[ds])
    res=dict(scope='development-selected test error analysis (rule 10); labels read here and in the evaluator only',files_read=['runs/20261013_m1_pairwise_evidence/main/records','runs/20261013_m1_pairwise_evidence/selection_decoded/{full_l,cmp_order,cmp_sum,cmp_order_d1}/predictions.jsonl','data/gt_4fps/*.npz','runs/20261007_m1_streamingtom/placement_controls_main/records'],results=result)
    (OUT/'summary.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(result,indent=1));print('ERROR_ANALYSIS_DONE')


if __name__=='__main__':main()
