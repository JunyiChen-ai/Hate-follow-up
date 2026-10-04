#!/usr/bin/env python3
"""Integrity checks and canonical-evaluator orchestration; GT only after scoring."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'CLAUDE.md').is_file())
sys.path.insert(0, str(ROOT))
from src.video_inputs import fixed_windows, load_manifest
from src.eval.evaluate import within_video_macro
from measure import existing_records, validate_pair
from extract import selected_rows, validate_cache, CACHE
from src.video_inputs import load_asr
from reader import arm_support

DATASETS = ('HateMM', 'HateClipSeg')
METRICS = ('frame_ROC_AUC', 'frame_PR_AUC', 'within_video_macro_ROC_AUC')


def read(path):
    records = {}
    for line in path.open():
        row = json.loads(line)
        key = row['dataset'], row['video_id']
        assert key not in records
        records[key] = row
    return records


def prepare(root, output, smoke, arm):
    rows = selected_rows(smoke)
    manifest = {(r['dataset'], r['video_id']): r for r in rows}
    config = json.loads((root / 'config.json').read_text())
    assert config['model'] == 'Qwen/Qwen3-VL-8B-Instruct' and config['GT_in_reader'] is False
    assert config['arm'] == arm and config['smoke'] == smoke
    raw = {a: existing_records(root / a / 'predictions.jsonl', manifest) for a in ('base', 'optimized')}
    checks = existing_records(root / 'checks.jsonl', manifest)
    assert all(r.keys() == manifest.keys() for r in [*raw.values(), checks])
    old = read(ROOT / 'runs/20260926_glr/base_gridA/predictions.jsonl')
    asr = {ds: load_asr(ds) for ds in DATASETS}
    changed = interventions = applied = 0
    costs = []
    for key, base in raw['base'].items():
        new, check = raw['optimized'][key], checks[key]
        detail = json.loads((root / 'details' / key[0] / (key[1]+'.json')).read_text())
        validate_pair(base, new, check, detail)
        native = old[key]
        assert base['extra']['z_video'] == native['extra']['z_video']
        assert np.array_equal(base['score_curve'], native['score_curve'])
        assert base['extra']['windows'] == native['extra']['windows']
        cached = json.loads((CACHE / key[0] / (key[1]+'.json')).read_text())
        validate_cache(cached, manifest[key], asr[key[0]].get(key[1],[]))
        assert detail['words'] == cached['words']
        p = arm_support(cached, arm)
        np.testing.assert_array_equal(np.asarray(detail['support']).reshape(p.shape), p)
        assert check['actual_forwards'] == 3+check['native_branches']+check['new_speech_calls']+check['diagnostic_calls']
        for trace, oldw in zip(detail['traces'], base['extra']['windows']):
            i = trace['window']
            chosen = np.flatnonzero(p.argmax(1)==i).tolist() if len(p) else []
            assert trace['selected_word_ids'] == chosen
            assert trace['body'] == ' '.join(cached['words'][wid]['text'] for wid in chosen)
            assert abs(trace['word_mass']-(p[:,i].sum() if len(p) else 0)) < 1e-10
            if trace['margin'] is not None: assert trace['attention_layers'] == 36
            changed += trace['margin'] != oldw.get('z_speech')
            if smoke and trace.get('cloned_cache_exact') is not None:
                assert trace['cloned_cache_exact'] is True
                interventions += bool(trace['intervention_changes_margin'])
                applied += bool(trace['intervention_changes_support'])
        costs.append(check)
    assert changed > 0
    if smoke: assert applied > 0 and interventions > 0
    result = dict(coverage=len(manifest), native_exact=True, changed_speech_windows=changed,
                  intervention_support_changes=applied if smoke else None,
                  intervention_margin_changes=interventions if smoke else None,
                  GT_read=False, mechanism_supported=False, cost={})
    for ds in DATASETS:
        records = [r for r in costs if r['dataset']==ds]
        result['cost'][ds] = dict(standalone_seconds={a:sum(r['standalone_seconds'][a] for r in records)
                for a in ('base','optimized')}, peak_GiB=max(max(r['peak_GiB'],r['align_peak_GiB']) for r in records),
                **{k:sum(r[k] for r in records) for k in ('align_seconds','decode_seconds','encoder_calls','decoder_calls','actual_forwards','new_speech_calls')})
        if smoke:
            representative = [r for r in records if r['video_id'] != 'hate_video_114']
            result['cost'][ds]['rough_full_seconds'] = (215 if ds=='HateMM' else 118)*np.mean([r['standalone_seconds']['optimized'] for r in representative])
    (output / ('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return raw


def evaluate(root, decoded, name):
    subprocess.run([sys.executable, '-m', 'src.eval.evaluate_four_datasets', '--predictions',
        str(root / name / 'predictions.jsonl'), '--gt-dir', 'data/gt_4fps', '--datasets', *DATASETS,
        '--out', str(root / name / 'metrics.json')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'experiments/20260926_twolevel/twolevel_r2.py', '--run', str(root / name),
        '--datasets', *DATASETS, '--noleak', '--transform', 'nscore', '--key', 'calib', '--duration', 'bma',
        '--bma-prior', 'length', '--min-windows', '2', '--bma-grid', '6', '--arm', 'm2',
        '--out-root', str(decoded), '--tag', name], cwd=ROOT, check=True)


def metrics(path):
    return {r['dataset']: r for r in json.loads(path.read_text())['per_dataset']}


def bootstrap(values):
    values = np.asarray(values)
    rng = np.random.default_rng(0)
    if not len(values): return dict(n=0,mean=None,CI95=None)
    means = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    return dict(n=len(values), mean=float(values.mean()), CI95=np.quantile(means, [.025, .975]).tolist())


def report(root, decoded, output):
    mm = {a: metrics(decoded / a / 'metrics.json') for a in ('base', 'optimized')}
    current = metrics(ROOT / 'runs/20260926_twolevel/r6_bma/metrics.json')
    raw = {a: read(root / a / 'predictions.jsonl') for a in mm}
    final = {a: read(decoded / a / 'predictions.jsonl') for a in mm}
    result = dict(scope='development-selected; unchanged r6 independently applied to each arm',
                  metric_sources={a: str((decoded / a / 'metrics.json').relative_to(ROOT)) for a in mm},
                  datasets={}, mechanism_supported=False)
    per_video = []
    for ds in DATASETS:
        assert all(mm['base'][ds][m] == current[ds][m] for m in METRICS), 'native six metrics mismatch'
        delta = {m: mm['optimized'][ds][m] - current[ds][m] for m in METRICS}
        gt = np.load(ROOT / f'data/gt_4fps/{ds}.npz', allow_pickle=True)
        ys = {str(v): np.asarray(gt['y4'][i]) for i, v in enumerate(gt['video_ids']) if str(gt['split'][i]) == 'test'}
        paired = {'final': [], 'raw_max': [], 'raw_speech_shared': []}
        for key in [k for k in raw['base'] if k[0] == ds]:
            row = dict(dataset=ds, video_id=key[1])
            y = ys[key[1]]
            for name in mm:
                row[name] = {}
                scores = dict(final=np.asarray(final[name][key]['score_curve']), raw_max=np.asarray(raw[name][key]['score_curve']))
                w = raw[name][key]['extra']['windows']
                idx = np.clip(((np.arange(len(scores['raw_max'])) + .5) / 4 // 8).astype(int), 0, len(w) - 1)
                for kind, curve in scores.items():
                    row[name][kind] = within_video_macro({key[1]: y}, {key[1]: curve})[METRICS[-1]]
            # Raw speech is measured on frames with a speech read in BOTH arms only.
            shared = np.asarray([all('z_speech' in raw[a][key]['extra']['windows'][i] for a in mm)
                                for i in range(len(w))])[idx]
            for name in mm:
                sw = raw[name][key]['extra']['windows']
                curve = np.asarray([item.get('z_speech',0.) for item in sw])[idx]
                n = min(len(y),len(curve)); use = shared[:n]
                row[name]['raw_speech_shared'] = within_video_macro({key[1]:y[:n][use]},
                    {key[1]:curve[:n][use]})[METRICS[-1]]
            if row['base']['final'] is not None:
                per_video.append(row)
                for kind in paired:
                    if row['base'][kind] is not None and row['optimized'][kind] is not None:
                        paired[kind].append(row['optimized'][kind]-row['base'][kind])
        result['datasets'][ds] = dict(final={a: {m: mm[a][ds][m] for m in METRICS} for a in mm},
            delta=delta, within_paired={kind: bootstrap(v) for kind, v in paired.items()},
            n_eligible=len(paired['final']))
    common = [m for m in METRICS if all(result['datasets'][ds]['delta'][m] >= .01 for ds in DATASETS)]
    no_losses = all(result['datasets'][ds]['delta'][m] >= (-.01 if m == METRICS[-1] else -.005)
                    for ds in DATASETS for m in METRICS)
    result['gates'] = dict(common_gain_metrics=common, performance_pass=bool(common and no_losses),
        any_qualifying_gain=any(result['datasets'][ds]['delta'][m] >= .01 for ds in DATASETS for m in METRICS),
        no_losses_beyond_noise=no_losses)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'per_video.json').write_text(json.dumps(per_video, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    print('ANALYSIS_DONE', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('prepare', 'evaluate', 'report'), required=True)
    ap.add_argument('--name', choices=('base', 'optimized'))
    ap.add_argument('--arm', choices=('soft', 'hard', 'proportional', 'unweighted', 'shifted'), default='soft')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    stem = 'r1_' + a.arm + ('_smoke' if a.smoke else '_main')
    root = ROOT / 'runs/20261004_m1_acoustic' / stem
    decoded = root.parent / (stem + '_decoded')
    output = root.parent / (stem + '_analysis')
    output.mkdir(parents=True, exist_ok=True)
    if a.stage == 'prepare':
        prepare(root, output, a.smoke, a.arm)
    elif a.stage == 'evaluate':
        assert not a.smoke and a.name
        evaluate(root, decoded, a.name)
    else:
        assert not a.smoke
        report(root, decoded, output)


if __name__ == '__main__':
    main()
