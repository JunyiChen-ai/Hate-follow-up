"""Which channel's window mean orders HateClipSeg hateful videos by GT share: Omni (cand. 43 records) vs Qwen3-VL (r6). Diagnostic, rule 10."""
import json, glob, os
import numpy as np
from scipy.stats import spearmanr
ROOT = '/home/jehc223/Hate-follow-up'
vl = {}
for line in open(f'{ROOT}/runs/20260926_glr/base_gridA/predictions.jsonl'):
    d = json.loads(line); vl[(d['dataset'], d['video_id'])] = d
for ds in ['HateMM', 'HateClipSeg']:
    gt = np.load(f'{ROOT}/data/gt_4fps/{ds}.npz', allow_pickle=True)
    ys = {v: np.asarray(y, float) for v, sp, y in zip(gt['video_ids'], gt['split'], gt['y4']) if sp == 'test'}
    cols = {k: [] for k in ['share', 'O_vis_native', 'O_vis_local', 'O_sp_prop', 'O_sp_audio', 'VL_vis', 'VL_sp']}
    for f in sorted(glob.glob(f'{ROOT}/runs/20261014_m1_audio_bound/main/records/{ds}/*.json')):
        vid = os.path.basename(f)[:-5]; d = vl.get((ds, vid))
        if d is None or vid not in ys or ys[vid].max() == 0:
            continue
        r = json.load(open(f)); T = r['traces']
        mean = lambda xs: float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else np.nan
        cols['share'].append(ys[vid].mean())
        cols['O_vis_native'].append(mean([(t.get('visual_native') or {}).get('z') for t in T]))
        cols['O_vis_local'].append(mean([(t.get('visual_local') or {}).get('z') for t in T]))
        cols['O_sp_prop'].append(mean([(t.get('speech_prop') or {}).get('z') for t in T]))
        cols['O_sp_audio'].append(mean([(t.get('speech_audio') or {}).get('z') for t in T]))
        cols['VL_vis'].append(mean([w['z_visual'] for w in d['extra']['windows']]))
        cols['VL_sp'].append(mean([w.get('z_speech') for w in d['extra']['windows']]))
    sh = np.array(cols['share']); print(f'== {ds}: {len(sh)} hateful videos; Spearman(channel window mean, GT share)')
    for k in cols:
        if k == 'share':
            continue
        x = np.array(cols[k]); ok = ~np.isnan(x)
        print(f'   {k:13s} {spearmanr(x[ok], sh[ok]).correlation:+.3f}  (n={ok.sum()})')
