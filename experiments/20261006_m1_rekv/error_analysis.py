"""Post-whole raw V/S/max/final ordering diagnostics using canonical helpers."""
import argparse
import json
import os
import socket
import numpy as np
from inputs import ROOT,DATASETS
from src.eval.evaluate import pooled,within_video_macro


def read(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    result={(r['dataset'],r['video_id']):r for r in rows};assert len(rows)==len(result)==333
    return result


def ordering(before,after,frame_counts):
    before=np.asarray(before,np.float64);after=np.asarray(after,np.float64);counts=np.asarray(frame_counts,np.int64)
    valid=np.isfinite(before)&np.isfinite(after)
    pairs=np.triu(np.ones((len(before),len(before)),bool),1)&valid[:,None]&valid[None,:]
    old=np.sign(before[:,None]-before[None,:]);new=np.sign(after[:,None]-after[None,:]);weights=counts[:,None]*counts[None,:]
    return dict(window_pairs=int(pairs.sum()),changed_window_pairs=int(((old!=new)&pairs).sum()),
        weighted_frame_pairs=int(weights[pairs].sum()),changed_weighted_frame_pairs=int(weights[(old!=new)&pairs].sum()),
        before_tied_pairs=int(((old==0)&pairs).sum()),after_tied_pairs=int(((new==0)&pairs).sum()),
        equal_all_margins=bool(np.array_equal(before,after,equal_nan=True)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--controls',action='store_true');a=ap.parse_args()
    root=ROOT/'runs/20261006_m1_rekv';run=root/'r1_full_main';decoded=root/'r1_full_main_decoded'
    paths={'native':(run/'base/predictions.jsonl',decoded/'base/predictions.jsonl'),
        'main':(run/'optimized/predictions.jsonl',decoded/'optimized/predictions.jsonl')}
    if a.controls:
        for arm in ('R0','L0','C0','L1','C1','L2','D0','T0','H0'):
            mode='selection' if arm not in ('T0','H0') else 'time' if arm=='T0' else 'history'
            paths[arm]=(root/('controls_main_'+mode)/arm/'predictions.jsonl',root/'controls_main_decoded'/arm/'predictions.jsonl')
    out=root/('controls_main_error_analysis' if a.controls else 'r1_full_main_error_analysis');out.mkdir(parents=True,exist_ok=True)
    print(socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    raw={name:read(pair[0]) for name,pair in paths.items()};final={name:read(pair[1]) for name,pair in paths.items()}
    result=dict(GT_read=True,scope='development-selected descriptive post-whole analysis only; canonical helpers; no scoring/fitting/threshold route',
        sources=[str(p.relative_to(ROOT)) for pair in paths.values() for p in pair],datasets={})
    videos=[];rng=np.random.default_rng(0)
    for ds in DATASETS:
        path=ROOT/'data/gt_4fps'/(ds+'.npz');result['sources'].append(str(path.relative_to(ROOT)))
        with np.load(path,allow_pickle=True) as gt:y={str(v):np.asarray(gt['y4'][i],np.int8) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        arrays={name:{k:{} for k in ('V','S','max','final')} for name in paths};differences={name:{k:[] for k in arrays[name]} for name in paths if name!='main'}
        for key in raw['main']:
            if key[0]!=ds:continue
            video=key[1];n=min(len(y[video]),len(raw['main'][key]['score_curve']));yy=y[video][:n]
            idx=np.clip(((np.arange(n)+.5)/4//8).astype(int),0,len(raw['main'][key]['extra']['windows'])-1)
            one=dict(dataset=ds,video_id=video,within={},ordering={})
            for name in paths:
                windows=raw[name][key]['extra']['windows']
                for kind,field in [('V','z_visual'),('S','z_speech'),('max','z')]:arrays[name][kind][video]=np.asarray([w.get(field,np.nan) for w in windows])[idx]
                arrays[name]['final'][video]=np.asarray(final[name][key]['score_curve'])[:n]
                one['within'][name]={kind:within_video_macro({'v':yy},{'v':scores[video]})['within_video_macro_ROC_AUC'] for kind,scores in arrays[name].items()}
            frame_counts=np.bincount(idx,minlength=len(raw['main'][key]['extra']['windows']))
            for name in differences:
                bw=raw[name][key]['extra']['windows'];mw=raw['main'][key]['extra']['windows']
                one['ordering'][name]={kind:ordering([w.get(field,np.nan) for w in bw],[w.get(field,np.nan) for w in mw],frame_counts)
                    for kind,field in [('V','z_visual'),('S','z_speech'),('max','z')]}
                vb=np.asarray([w['z_visual'] for w in bw]);vm=np.asarray([w['z_visual'] for w in mw]);speech=np.asarray([w.get('z_speech',np.nan) for w in bw])
                assert np.array_equal(speech,np.asarray([w.get('z_speech',np.nan) for w in mw]),equal_nan=True),'S must remain exact'
                valid=np.isfinite(speech);old=np.sign(vb-speech);new=np.sign(vm-speech)
                one['ordering'][name]['max_branch']=dict(speech_windows=int(valid.sum()),changed_winner_windows=int(((old!=new)&valid).sum()),
                    changed_winner_frames=int(frame_counts[(old!=new)&valid].sum()),native_tied_windows=int(((old==0)&valid).sum()),main_tied_windows=int(((new==0)&valid).sum()))
                for kind in differences[name]:
                    main_value=one['within']['main'][kind];value=one['within'][name][kind]
                    if main_value is not None and value is not None:differences[name][kind].append(main_value-value)
            videos.append(one)
        table={name:{kind:{**pooled(y,scores),**within_video_macro(y,scores)} for kind,scores in arrays[name].items()} for name in arrays}
        paired={}
        for name,rows in differences.items():
            paired[name]={}
            for kind,delta in rows.items():
                values=np.asarray(delta,np.float64);assert len(values)>0
                samples=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
                paired[name][kind]=dict(n=len(values),main_minus_arm_mean=float(values.mean()),paired_bootstrap95=np.quantile(samples,[.025,.975]).tolist(),seed=0,draws=10000)
        result['datasets'][ds]=dict(canonical_table=table,paired_within=paired,
            speech_scope='shared finite speech-frame subset; may have fewer eligible videos than standard full within')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(out/'per_video.json').write_text(json.dumps(videos,indent=2)+'\n')
    print(json.dumps({ds:r['paired_within'] for ds,r in result['datasets'].items()},indent=2));print('ERROR_ANALYSIS_DONE',flush=True)


if __name__=='__main__':main()
