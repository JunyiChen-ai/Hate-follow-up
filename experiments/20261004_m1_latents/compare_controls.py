#!/usr/bin/env python3
"""Post-score mechanism comparisons; requires all predeclared complete controls."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'CLAUDE.md').is_file())
sys.path.insert(0, str(ROOT))
from analyze import DATASETS, METRICS, metrics, read, bootstrap
from src.eval.evaluate import within_video_macro
from src.video_inputs import load_manifest, fixed_windows

ARMS = ('full', 'initial', 'warmup', 'search_only', 'wrong_support')
EXP = ROOT / 'runs/20261004_m1_latents'


def main():
    primary = json.loads((EXP / 'r1_full_main_analysis/summary.json').read_text())
    assert primary['gates']['performance_pass'] is True, 'main performance gate required before control analysis'
    expected = {(r['dataset'], r['video_id']): r for r in load_manifest(
        ROOT / 'data/omsl_v6_inputs/manifests/all_test.jsonl', DATASETS)}
    assert len(expected) == 333 and sum(k[0] == 'HateMM' for k in expected) == 215
    mm, raw, decoded, checks, detail = {}, {}, {}, {}, {}
    configs = {}
    for arm in ARMS:
        stem = 'r1_' + arm + '_main'
        run = EXP / stem
        cfg = json.loads((run / 'config.json').read_text())
        assert cfg['arm'] == arm and cfg['smoke'] is False and cfg['GT_in_reader'] is False
        configs[arm] = {k: v for k, v in cfg.items() if k not in ('arm', 'host', 'date')}
        if arm != 'full':
            assert configs[arm] == configs['full']
        alignment = json.loads((EXP / (stem + '_analysis') / 'alignment.json').read_text())
        assert alignment['coverage'] == 333 and alignment['native_exact'] is True
        raw[arm] = read(run / 'optimized/predictions.jsonl')
        decoded[arm] = read(EXP / (stem + '_decoded') / 'optimized/predictions.jsonl')
        mm[arm] = metrics(EXP / (stem + '_decoded') / 'optimized/metrics.json')
        checks[arm] = read(run / 'checks.jsonl')
        assert raw[arm].keys() == decoded[arm].keys() == checks[arm].keys() == expected.keys()
        if arm != 'full':
            assert raw[arm].keys() == raw['full'].keys()
        detail[arm] = {key: json.loads((run / 'details' / key[0] / (key[1] + '.json')).read_text()) for key in raw[arm]}
    differences, per_video, interventions, costs = {}, [], {}, {}
    for ds in DATASETS:
        gt = np.load(ROOT / f'data/gt_4fps/{ds}.npz', allow_pickle=True)
        ys = {str(v): np.asarray(gt['y4'][i]) for i, v in enumerate(gt['video_ids']) if str(gt['split'][i]) == 'test'}
        eligible = []
        intervention = dict(total_windows=0, multiple_frames=0, changed_positive_support=0,
                            unchanged_positive_support_windows=[], single_frame_windows=[])
        for key in [k for k in raw['full'] if k[0] == ds]:
            full = raw['full'][key]
            row = dict(dataset=ds, video_id=key[1], within={})
            for arm in ARMS:
                r = raw[arm][key]
                assert r['extra']['z_video'] == full['extra']['z_video'] and r['extra']['stance'] == full['extra']['stance']
                nw = len(fixed_windows(float(expected[key]['duration']), 8))
                assert len(r['extra']['windows']) == len(full['extra']['windows']) == nw
                assert len(detail['full'][key]['traces']) == len(detail[arm][key]['traces']) == nw
                for field in ('image_counts', 'image_positions', 'original_frame_times', 'merger_shape'):
                    assert detail[arm][key][field] == detail['full'][key][field]
                for f, w, tf, tc in zip(full['extra']['windows'], r['extra']['windows'],
                                       detail['full'][key]['traces'], detail[arm][key]['traces']):
                    assert f['i'] == w['i'] and f['start'] == w['start'] and f['end'] == w['end']
                    assert f.get('z_speech') == w.get('z_speech')
                    assert tf['positive'] == tc['positive'] and tf['negative'] == tc['negative']
                raw_curve = np.asarray(r['score_curve'])
                windows = r['extra']['windows']
                idx = np.clip(((np.arange(len(raw_curve)) + .5) / 4 // 8).astype(int), 0, len(windows) - 1)
                curves = dict(final=np.asarray(decoded[arm][key]['score_curve']), raw_max=raw_curve,
                              raw_visual=np.asarray([w['z_visual'] for w in windows])[idx])
                row['within'][arm] = {kind: within_video_macro({key[1]: ys[key[1]]}, {key[1]: s})[METRICS[-1]]
                                     for kind, s in curves.items()}
            if row['within']['full']['final'] is not None:
                eligible.append(row)
                per_video.append(row)
            wrong = detail['wrong_support'][key]
            counts = wrong['image_counts']
            cumulative = np.cumsum([0] + counts).tolist()
            for trace in wrong['traces']:
                intervention['total_windows'] += 1
                intervention['multiple_frames'] += len(counts) > 1
                intervention['changed_positive_support'] += bool(trace['actual_content_changed'])
                marker = dict(video_id=key[1], window=trace['window'])
                if not trace['actual_content_changed']:
                    intervention['unchanged_positive_support_windows'].append(marker)
                if len(counts) == 1:
                    intervention['single_frame_windows'].append(marker)
                for field, index_field in (('wrong_positive', 'positive'), ('wrong_negative', 'negative')):
                    original = [i for group in trace[index_field] for i in group]
                    bindings = trace[field]
                    assert len(bindings) == len(original)
                    for index, binding in zip(original, bindings):
                        frame = next(i for i in range(len(counts)) if cumulative[i] <= index < cumulative[i + 1])
                        patch = index - cumulative[frame]
                        target = (frame + len(counts) // 2) % len(counts)
                        target_patch = patch * counts[target] // counts[frame]
                        assert binding == dict(frame=frame, patch=patch, target_frame=target, target_patch=target_patch)
                        assert 0 <= target_patch < counts[target]
            if len(wrong['image_counts']) == 1:
                assert not any(t['actual_content_changed'] for t in wrong['traces'])
        interventions[ds] = intervention
        differences[ds] = {}
        costs[ds] = {}
        for arm in ARMS:
            costs[ds][arm] = dict(standalone_seconds=sum(r['extra']['standalone_seconds'] for k, r in raw[arm].items() if k[0] == ds),
                forwards=sum(r['calls'] for k, r in raw[arm].items() if k[0] == ds),
                peak_GiB=max(c['peak_GiB'] for k, c in checks[arm].items() if k[0] == ds))
            if arm == 'full':
                continue
            delta = {m: mm['full'][ds][m] - mm[arm][ds][m] for m in METRICS}
            paired = {kind: bootstrap([r['within']['full'][kind] - r['within'][arm][kind] for r in eligible])
                      for kind in ('final', 'raw_max', 'raw_visual')}
            differences[ds][arm] = dict(full_minus_control=delta, paired=paired)
    common = {arm: [m for m in METRICS if all(differences[ds][arm]['full_minus_control'][m] >= .01 for ds in DATASETS)]
              for arm in ARMS if arm != 'full'}
    # This gate only supports the whole-optimization claim and content-binding check.
    # Stage-specific necessity is reported independently; no automatic paper claim.
    result = dict(scope='development-selected complete predeclared controls; full minus control',
                  metrics={arm: {ds: {m: mm[arm][ds][m] for m in METRICS} for ds in DATASETS} for arm in ARMS},
                  metric_sources={arm: str((EXP / ('r1_' + arm + '_main_decoded') / 'optimized/metrics.json').relative_to(ROOT)) for arm in ARMS},
                  comparisons=differences, common_gain_metrics=common, intervention=interventions, costs=costs,
                  whole_optimization_threshold_pass=bool(common['initial']),
                  content_binding_threshold_pass=bool(common['wrong_support']),
                  StageI_necessity_threshold_pass=bool(common['search_only']),
                  StageII_necessity_threshold_pass=bool(common['warmup']),
                  mechanism_supported=False,
                  note='Threshold comparisons are evidence for independent interpretation; raw ordering and actual intervention coverage must also support the stated explanation. Content-change coverage measures positive-support embeddings; negative bindings are checked as index mappings, not claimed as a stored content-equality audit.')
    output = EXP / 'r1_mechanism_analysis'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'per_video.json').write_text(json.dumps(per_video, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    print('CONTROL_ANALYSIS_DONE', flush=True)


if __name__ == '__main__':
    main()
