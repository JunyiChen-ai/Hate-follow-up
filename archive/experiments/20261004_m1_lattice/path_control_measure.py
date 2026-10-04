#!/usr/bin/env python3
"""Actual paired measurements for the declared lattice controls, no GT access."""
import argparse
import copy
import json
import logging
import socket
import sys
import time
import numpy as np
import torch

from extract import ROOT, CACHE, DATASETS, selected_rows, validate
sys.path.insert(0, str(ROOT))
from src.mllm_judge import Judge, MODEL, VIDEO_QUESTION, yesno_question
from src.stance_cache import build, margin
from src.video_inputs import frame_paths, fixed_windows, window_text, load_asr
from measure import record, tick
from lattice import CACHE_VERSION, CONSTANTS
from path_control_reader import graph_positions
from path_control_reader import ARMS, CONTROL_VERSION, compile_control, control_margin, donor_map


def current_context(renderer, row, segments, stance):
    frames = frame_paths(row['dataset'], row['video_id'], 20)
    msgs, files = renderer.prefix_messages(frames, segments)
    head, enc = renderer.encode_prefix(msgs, files)
    qids, qtext = renderer.branch_ids(msgs, VIDEO_QUESTION)
    aids, atext = renderer.answer_ids(msgs, VIDEO_QUESTION, stance)
    history = [renderer.turn('user', VIDEO_QUESTION), renderer.turn('assistant', stance)]
    ids = enc['input_ids'][0].tolist() + qids + aids
    return dict(msgs=msgs, files=files, history=history, head=head + qtext + atext), dict(
        files=[str(p) for p in files], token_ids=ids, image_counts=list(renderer.img_tokens),
        prefix_head=head + qtext + atext, prefix_tokens=enc['input_ids'].shape[1])


def equivalent(a, b):
    for k in ('dataset', 'video_id', 'duration', 'native_rate', 'score_curve', 'error', 'calls'):
        assert a[k] == b[k], k
    for k in ('z_video', 'stance', 'prefix_tokens', 'stance_cache_tokens',
              'stance_cache_logical_start', 'windows'):
        assert a['extra'][k] == b['extra'][k], k


def validate_bundle(row, bundle, metadata, segments, renderer, smoke):
    validate(metadata, row, segments)
    assert bundle['control_version'] == CONTROL_VERSION and bundle['GT_read'] is False
    assert bundle['segments'] == [list(s) for s in segments]
    assert list(bundle['arms']) == list(ARMS)
    base = bundle['base']; wins = fixed_windows(float(row['duration']), 8)
    ctx, current = current_context(renderer, row, segments, base['extra']['stance'])
    assert current == bundle['native_input'], 'current native conversation/input differs'
    n = len(current['token_ids']); start = base['extra']['stance_cache_logical_start']
    assert n == base['extra']['stance_cache_tokens']
    assert current['prefix_tokens'] == base['extra']['prefix_tokens']
    donors = donor_map(metadata['windows']); count = len(donors)
    diagnostics = 0
    for name, r in [('base', base), *bundle['arms'].items()]:
        assert (r['dataset'], r['video_id']) == (row['dataset'], row['video_id'])
        assert r['duration'] == float(row['duration']) and r['error'] is None and r['native_rate'] == 4
        assert r['extra']['stance'] == ('Yes' if r['extra']['z_video'] > 0 else 'No')
        assert len(r['extra']['windows']) == len(wins)
        for k in ('z_video', 'stance', 'prefix_tokens', 'stance_cache_tokens', 'stance_cache_logical_start'):
            assert r['extra'][k] == base['extra'][k]
        ww = r['extra']['windows']
        for i, (w, (a, b)) in enumerate(zip(ww, wins)):
            assert (w['i'], w['start'], w['end']) == (i, a, b)
            assert w['z'] == max(w['z_visual'], w.get('z_speech', float('-inf')))
            assert w['z_visual'] == base['extra']['windows'][i]['z_visual']
        idx = np.clip(((np.arange(len(r['score_curve'])) + .5) / 4 // 8).astype(int), 0, len(ww) - 1)
        assert len(r['score_curve']) == int(np.ceil(r['duration'] * 4))
        assert np.array_equal(r['score_curve'], np.asarray([w['z'] for w in ww])[idx])
        assert np.isfinite(r['score_curve']).all() and np.isfinite(r['extra']['z_video'])
        assert r['calls'] == 3 + len(wins) + sum('z_speech' in w for w in ww)
        if name == 'base':
            for w, m in zip(ww, metadata['windows']):
                assert ('z_speech' in w) == bool(m['native_body'].strip())
            continue
        traces = bundle['traces'][name]
        assert len(traces) == len(wins)
        for w, m, t in zip(ww, metadata['windows'], traces):
            assert ('z_speech' in w) == t['available'] == m['available']
            if not m['available']:
                assert t['reason'] == m['reason']
                continue
            donor = metadata['windows'][donors[m['i']]] if name == 'wrong_audio_window' else None
            ids, graph, h, tail, expected = compile_control(renderer, ctx, m, len(wins), name, donor)
            assert all(t[k] == v for k, v in expected.items()), 'current graph/heads/source differs'
            assert t['prefix_tokens'] == n and t['prefix_logical_start'] == start
            assert t['physical_tokens'] == len(ids) and t['graph_tokens'] == len(graph['ids'])
            assert t['path_count'] == len(graph['paths']) and t['margin'] == w['z_speech']
            pos_arm = 'flat' if name in ('flat', 'onebest') else 'full'
            bias_arm = name if name in ('flat', 'binary') else 'full'
            assert t['position_arm'] == pos_arm and t['bias_arm'] == bias_arm
            assert t['ordinary_serial'] == (name == 'onebest') and t['attention_dtype'] == 'torch.bfloat16'
            assert t['logical_positions'] == graph_positions(graph, h, tail, start, pos_arm)
        clones = [t for t in traces if 'clone_margin' in t]
        assert len(clones) == int(smoke and count > 0)
        assert all(t['clone_margin'] == t['margin'] for t in clones)
        diagnostics += len(clones)
        cost = bundle['checks']['arms'][name]
        assert cost['diagnostic_forwards'] == len(clones) and cost['actual_forwards'] == count
    c = bundle['checks']
    assert c['diagnostic_forwards'] == diagnostics
    assert c['actual_forwards'] == base['calls'] + len(ARMS) * count + diagnostics
    assert c['preprocessing_seconds'] == metadata['standalone_seconds']
    assert c['input_actual_forwards'] == metadata['actual_forwards']
    assert c['peak_GiB'] >= metadata['peak_GiB']
    timing_keys = ('prefix_seconds', 'visual_seconds', 'reference_speech_seconds',
                   'physical_read_seconds', 'preprocessing_seconds', 'peak_GiB')
    assert all(np.isfinite(c[k]) and c[k] >= 0 for k in timing_keys)
    assert base['extra']['standalone_seconds'] == (
        c['prefix_seconds'] + c['visual_seconds'] + c['reference_speech_seconds'])
    total_read = c['prefix_seconds'] + c['visual_seconds'] + c['reference_speech_seconds']
    for name in ARMS:
        cost = c['arms'][name]
        assert all(np.isfinite(cost[k]) and cost[k] >= 0 for k in
                   ('speech_seconds', 'diagnostic_seconds', 'standalone_seconds'))
        expected_time = metadata['standalone_seconds'] + c['prefix_seconds'] + c['visual_seconds'] + cost['speech_seconds']
        assert cost['standalone_seconds'] == bundle['arms'][name]['extra']['standalone_seconds'] == expected_time
        total_read += cost['speech_seconds'] + cost['diagnostic_seconds']
    assert c['physical_read_seconds'] >= total_read


@torch.no_grad()
def read_video(j, row, segments, metadata, smoke):
    first = j.forward_calls; torch.cuda.reset_peak_memory_stats(); joint_start = tick()
    frames = frame_paths(row['dataset'], row['video_id'], 20)
    start = tick(); cache, ctx = build(j, frames, segments); prefix_seconds = tick() - start
    # The rendered chat contains one image placeholder, while its cache contains
    # the actual processor grid's expanded image tokens.
    native_ids = []; image_index = 0
    for token in j.tok.encode(ctx['head'], add_special_tokens=False):
        if token == j.image_token_id:
            native_ids.extend([token] * ctx['image_counts'][image_index]); image_index += 1
        else:
            native_ids.append(token)
    assert image_index == len(ctx['image_counts']) and len(native_ids) == cache.get_seq_length()
    native_input = dict(files=[str(p) for p in ctx['files']],
        token_ids=native_ids, image_counts=ctx['image_counts'],
        prefix_head=ctx['head'], prefix_tokens=ctx['prefix_tokens'])
    wins = fixed_windows(float(row['duration']), 8)
    visual, speech = [], []; visual_seconds = reference_seconds = 0.
    for i, (a, b) in enumerate(wins):
        body = window_text(segments, a, b)
        start = tick(); visual.append(margin(j, cache, ctx, yesno_question(i, len(wins), a, b, body, 'visual')))
        visual_seconds += tick() - start
        if body.strip():
            start = tick(); speech.append(margin(j, cache, ctx, yesno_question(i, len(wins), a, b, body, 'speech')))
            reference_seconds += tick() - start
        else:
            speech.append(None)
    base = record(row, ctx, visual, speech, prefix_seconds + visual_seconds + reference_seconds, 'm1_native')
    arms, traces, costs = {}, {}, {}; donors = donor_map(metadata['windows'])
    diagnostics = 0
    for name in ARMS:
        zz, tt = [], []; seconds = diagnostic_seconds = 0.; checked = False
        for m in metadata['windows']:
            if not m['available']:
                zz.append(None); tt.append(dict(available=False, reason=m['reason'])); continue
            donor = metadata['windows'][donors[m['i']]] if name == 'wrong_audio_window' else None
            start = tick(); z, trace = control_margin(j, cache, ctx, m, len(wins), name, donor); seconds += tick() - start
            trace['available'] = True
            if smoke and not checked:
                start = tick(); clone = copy.deepcopy(cache)
                try:
                    replay, _ = control_margin(j, clone, ctx, m, len(wins), name, donor)
                    assert replay == z
                finally:
                    del clone
                trace['clone_margin'] = replay; diagnostic_seconds += tick() - start
                diagnostics += 1; checked = True
            zz.append(z); tt.append(trace)
        traces[name] = tt
        arms[name] = record(row, ctx, visual, zz,
            metadata['standalone_seconds'] + prefix_seconds + visual_seconds + seconds, 'm1_path_lattice_' + name)
        costs[name] = dict(speech_seconds=seconds, diagnostic_seconds=diagnostic_seconds,
            actual_forwards=len(donors), diagnostic_forwards=int(checked),
            standalone_seconds=arms[name]['extra']['standalone_seconds'])
    joint_seconds = tick() - joint_start; actual = j.forward_calls - first
    assert actual == base['calls'] + len(ARMS) * len(donors) + diagnostics
    checks = dict(actual_forwards=actual, diagnostic_forwards=diagnostics,
        physical_read_seconds=joint_seconds, preprocessing_seconds=metadata['standalone_seconds'],
        input_actual_forwards=metadata['actual_forwards'], peak_GiB=max(torch.cuda.max_memory_allocated()/2**30, metadata['peak_GiB']),
        prefix_seconds=prefix_seconds, visual_seconds=visual_seconds, reference_speech_seconds=reference_seconds, arms=costs)
    del cache
    for r in [base, *arms.values()]: r['code_path'] = 'experiments/20261004_m1_lattice/path_control_measure.py'
    return dict(control_version=CONTROL_VERSION, GT_read=False, segments=[list(s) for s in segments],
        native_input=native_input, base=base, arms=arms, traces=traces, checks=checks)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    out = ROOT/'runs/20261004_m1_lattice'/('r2_controls_' + ('smoke' if args.smoke else 'main'))
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'run.log'), logging.StreamHandler(sys.stdout)])
    logging.info('host %s', socket.gethostname()); (out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    config = dict(host=socket.gethostname(), date=time.strftime('%Y-%m-%d'), model=MODEL,
        constants=CONSTANTS, cache_version=CACHE_VERSION, control_version=CONTROL_VERSION, arms=list(ARMS),
        GT_read=False, smoke=args.smoke, torch=torch.__version__, transformers=transformers.__version__,
        command='python -u ' + ' '.join(sys.argv),
        code='experiments/20261004_m1_lattice/{path_control_reader,path_control_measure,path_reader,path_graph}.py + src/stance_cache.py; sources2026-10-05')
    cp = out/'config.json'
    if cp.exists():
        old = json.loads(cp.read_text()); assert {k:v for k,v in old.items() if k != 'date'} == {k:v for k,v in config.items() if k != 'date'}
    else:
        cp.write_text(json.dumps(config, indent=2) + '\n')
    rows = selected_rows(args.smoke); asr = {ds:load_asr(ds) for ds in DATASETS}
    from src.mllm_renderer import cpu_renderer
    renderer = cpu_renderer(); completed = {}
    for row in rows:
        p = out/'records'/row['dataset']/(row['video_id']+'.json')
        if p.exists():
            metadata = json.loads((CACHE/row['dataset']/(row['video_id']+'.json')).read_text())
            bundle = json.loads(p.read_text())
            validate_bundle(row, bundle, metadata, asr[row['dataset']].get(row['video_id'], []), renderer, args.smoke)
            completed[row['dataset'], row['video_id']] = bundle
    torch.manual_seed(0); j = Judge(MODEL); j.forward_calls = 0
    hook = j.model.model.register_forward_pre_hook(lambda *_:setattr(j, 'forward_calls', j.forward_calls + 1))
    for i, row in enumerate(rows, 1):
        key = row['dataset'], row['video_id']
        if key in completed:
            logging.info('%d/%d reuse %s/%s', i, len(rows), *key); continue
        segments = asr[key[0]].get(key[1], [])
        metadata = json.loads((CACHE/key[0]/(key[1]+'.json')).read_text()); validate(metadata, row, segments)
        bundle = read_video(j, row, segments, metadata, args.smoke)
        validate_bundle(row, bundle, metadata, segments, renderer, args.smoke)
        p = out/'records'/key[0]/(key[1]+'.json'); p.parent.mkdir(parents=True, exist_ok=True)
        temp = p.with_suffix('.partial'); temp.write_text(json.dumps(bundle)+'\n'); temp.replace(p)
        completed[key] = bundle; logging.info('%d/%d %s/%s %.2fs', i, len(rows), *key, bundle['checks']['physical_read_seconds'])
    hook.remove()
    for name in ('base',) + ARMS:
        d = out/name; d.mkdir(exist_ok=True); (d/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        with (d/'predictions.jsonl').open('w') as f:
            for row in rows:
                b = completed[row['dataset'], row['video_id']]
                f.write(json.dumps(b['base'] if name == 'base' else b['arms'][name])+'\n')
    logging.info('CONTROL_DONE coverage=%d', len(rows))


if __name__ == '__main__':
    main()
