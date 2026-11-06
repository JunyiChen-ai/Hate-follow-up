"""Video-level comparison of Qwen3-Omni (candidate 43 records) and Qwen3-VL-8B (r6 base_gridA) on the same r6 inputs.
Diagnostic only (rule 10): reads test GT to explain the HateClipSeg pooled loss of the Omni reader."""
import json, glob, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = '/home/jehc223/Hate-follow-up'
vl = {}
for line in open(f'{ROOT}/runs/20260926_glr/base_gridA/predictions.jsonl'):
    d = json.loads(line)
    vl[(d['dataset'], d['video_id'])] = d


def wmax(v, s):
    return v if s is None else max(v, s)


for ds in ['HateMM', 'HateClipSeg']:
    gt = np.load(f'{ROOT}/data/gt_4fps/{ds}.npz', allow_pickle=True)
    lab = {v: int(np.asarray(y).max() > 0) for v, sp, y in zip(gt['video_ids'], gt['split'], gt['y4']) if sp == 'test'}
    rows = []
    for f in sorted(glob.glob(f'{ROOT}/runs/20261014_m1_audio_bound/main/records/{ds}/*.json')):
        vid = os.path.basename(f)[:-5]
        r = json.load(open(f))
        d = vl.get((ds, vid))
        if d is None or vid not in lab:
            continue
        ow = [wmax(t['visual_native']['z'], (t.get('speech_prop') or {}).get('z')) for t in r['traces']]
        vw = [w['z'] for w in d['extra']['windows']]
        ov = [t['visual_native']['z'] for t in r['traces']]
        vv = [w['z_visual'] for w in d['extra']['windows']]
        rows.append(dict(vid=vid, y=lab[vid], o_z=r['native']['global_margin'], v_z=d['extra']['z_video'],
                         o_m=float(np.mean(ow)), v_m=float(np.mean(vw)), o_vis=float(np.mean(ov)), v_vis=float(np.mean(vv))))
    y = np.array([r['y'] for r in rows])
    g = lambda k: np.array([r[k] for r in rows])
    print(f'== {ds}: {len(rows)} videos, {y.sum()} hateful / {(1 - y).sum()} non-hateful')
    for name, a, b in [('verdict z_video', 'o_z', 'v_z'), ('window mean (max of channels)', 'o_m', 'v_m'),
                       ('visual window mean', 'o_vis', 'v_vis')]:
        print(f'  video AUC {name}: Omni {roc_auc_score(y, g(a)):.3f}  Qwen3-VL {roc_auc_score(y, g(b)):.3f}')
    for k_o, k_v, nm in [('o_z', 'v_z', 'key = verdict + window mean')]:
        print(f'  video AUC {nm}: Omni {roc_auc_score(y, g(k_o) + g("o_m")):.3f}  Qwen3-VL {roc_auc_score(y, g(k_v) + g("v_m")):.3f}')
    for who, k in [('Omni', 'o_z'), ('Qwen3-VL', 'v_z')]:
        z = g(k)
        print(f'  {who}: verdict Yes rate hateful {np.mean(z[y == 1] > 0):.2f}, non-hateful {np.mean(z[y == 0] > 0):.2f}; '
              f'margin median hateful {np.median(z[y == 1]):+.1f}, non-hateful {np.median(z[y == 0]):+.1f}')
    oz, vz = g('o_z') > 0, g('v_z') > 0
    print(f'  verdict agreement {np.mean(oz == vz):.2f}; Omni Yes & VL No: hateful {np.sum(oz & ~vz & (y == 1))}, non-hateful {np.sum(oz & ~vz & (y == 0))}; '
          f'Omni No & VL Yes: hateful {np.sum(~oz & vz & (y == 1))}, non-hateful {np.sum(~oz & vz & (y == 0))}')


# Part 2: frame score = key + centred rank (|rank| <= .5), so pooled ROC is set by how the key orders videos.
# Proxy: every frame gets its video's key; pooled frame AUC of that, plus the key's order of hateful videos by share.
from scipy.stats import spearmanr
print('\n== Part 2: key only, pooled frame AUC proxy; Spearman of key with GT share among hateful videos')
for ds in ['HateMM', 'HateClipSeg']:
    gt = np.load(f'{ROOT}/data/gt_4fps/{ds}.npz', allow_pickle=True)
    ys = {v: np.asarray(y, float) for v, sp, y in zip(gt['video_ids'], gt['split'], gt['y4']) if sp == 'test'}
    R = []
    for f in sorted(glob.glob(f'{ROOT}/runs/20261014_m1_audio_bound/main/records/{ds}/*.json')):
        vid = os.path.basename(f)[:-5]
        d = vl.get((ds, vid))
        if d is None or vid not in ys:
            continue
        r = json.load(open(f))
        ow = np.mean([wmax(t['visual_native']['z'], (t.get('speech_prop') or {}).get('z')) for t in r['traces']])
        vw = np.mean([w['z'] for w in d['extra']['windows']])
        R.append((vid, ys[vid], r['native']['global_margin'], d['extra']['z_video'], ow, vw))
    keys = {'VL verdict + VL mean (r6)': lambda o_z, v_z, o_m, v_m: v_z + v_m,
            'Omni verdict + Omni mean': lambda o_z, v_z, o_m, v_m: o_z + o_m,
            'Omni verdict + VL mean': lambda o_z, v_z, o_m, v_m: o_z + v_m,
            'VL verdict + Omni mean': lambda o_z, v_z, o_m, v_m: v_z + o_m,
            'VL verdict only': lambda o_z, v_z, o_m, v_m: v_z,
            'Omni verdict only': lambda o_z, v_z, o_m, v_m: o_z,
            'VL mean only': lambda o_z, v_z, o_m, v_m: v_m,
            'Omni mean only': lambda o_z, v_z, o_m, v_m: o_m}
    print(f'-- {ds}')
    sd = lambda i: np.std([x[i] for x in R])
    print(f'   spread (sd over videos): Omni verdict {sd(2):.1f}, VL verdict {sd(3):.1f}, Omni window mean {sd(4):.1f}, VL window mean {sd(5):.1f}')
    for nm, fn in keys.items():
        k = np.array([fn(*x[2:]) for x in R])
        yy = np.concatenate([x[1] for x in R]); kk = np.concatenate([np.full(len(x[1]), ki) for x, ki in zip(R, k)])
        hate = [(ki, x[1].mean()) for x, ki in zip(R, k) if x[1].max() > 0]
        rho = spearmanr([h[0] for h in hate], [h[1] for h in hate]).correlation
        print(f'   {nm:28s} pooled-frame AUC {roc_auc_score(yy, kk):.3f}   Spearman(key, share | hateful) {rho:+.3f}')
