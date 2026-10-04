#!/usr/bin/env python3
"""Paired native and optimized reads. This executable never loads ground truth."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np
import torch
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'CLAUDE.md').is_file())
sys.path.insert(0, str(ROOT))
from src.mllm_judge import Judge, MODEL, VIDEO_QUESTION, yesno_question
from src.video_inputs import FPS, frame_paths, load_asr, load_manifest, fixed_windows, window_text
from latents import LatentReader, QueryAttention, question_rows, K, POS, NEG, WARMUP, SEARCH, TOP
from latents import TAU, LR, SIGMA, DECAY, ALPHA


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def existing_records(path, expected):
    rows = {}
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            key = r['dataset'], r['video_id']
            assert key in expected and key not in rows, ('unexpected/duplicate record', path, key)
            rows[key] = r
    return rows


def validate_pair(base, optimized, check, detail):
    assert base['dataset'] == optimized['dataset'] == check['dataset']
    assert base['video_id'] == optimized['video_id'] == check['video_id']
    assert base['duration'] == optimized['duration'] and base['native_rate'] == optimized['native_rate'] == FPS
    assert base['error'] is optimized['error'] is None
    assert np.isfinite(base['score_curve']).all() and np.isfinite(optimized['score_curve']).all()
    assert len(base['score_curve']) == len(optimized['score_curve']) == math.ceil(base['duration'] * FPS)
    assert base['extra']['z_video'] == optimized['extra']['z_video']
    assert base['extra']['stance'] == optimized['extra']['stance']
    bw, ow = base['extra']['windows'], optimized['extra']['windows']
    assert len(bw) == len(ow) == len(detail['traces']) == check['visual_queries']
    for b, o, t in zip(bw, ow, detail['traces']):
        assert b['i'] == o['i'] == t['window']
        assert b['start'] == o['start'] and b['end'] == o['end']
        assert b.get('z_speech') == o.get('z_speech')
        assert o['z_visual'] == t['margin']
    assert sum(detail['image_counts']) == detail['merger_shape'][0] == len(detail['image_positions'])


@torch.no_grad()
def read_video(j, reader, row, segments, arm, smoke):
    ds, vid, duration = row['dataset'], row['video_id'], float(row['duration'])
    frames = frame_paths(ds, vid, 20)
    assert 1 <= len(frames) <= 20
    wins = fixed_windows(duration, 8)
    texts = [window_text(segments, a, b) for a, b in wins]
    available = [bool(t.strip()) for t in texts]
    B = len(wins) + sum(available)
    first_call = j.forward_calls
    torch.cuda.reset_peak_memory_stats()
    start = tick()
    msgs, files = j.prefix_messages(frames, segments)
    text, enc = j.encode_prefix(msgs, files)
    image_positions = (enc['input_ids'][0] == j.image_token_id).nonzero().flatten().to(j.device)
    captured = []

    def merger_hook(_module, _args, output):
        assert torch.is_tensor(output) and output.ndim == 2
        captured.append(output.detach().float())

    hook = j.model.model.visual.merger.register_forward_hook(merger_hook)
    try:
        cache = j.prefix_cache(enc)
    finally:
        hook.remove()
    assert len(captured) == 1
    embeddings = captured[0]
    counts = list(j.img_tokens)
    assert embeddings.shape[0] == image_positions.numel() == sum(counts)
    assert embeddings.shape[1] == j.model.config.text_config.hidden_size
    P = cache.get_seq_length()
    qid, qtext = j.branch_ids(msgs, VIDEO_QUESTION)
    zv = j.cached_margin(cache, qid, in_place=True)
    stance = 'Yes' if zv > 0 else 'No'
    aid, atext = j.answer_ids(msgs, VIDEO_QUESTION, stance)
    j.extend_cache(cache, aid)
    history = [{'role': 'user', 'content': [{'type': 'text', 'text': VIDEO_QUESTION}]}, j.turn('assistant', stance)]
    head = text + qtext + atext
    n = cache.get_seq_length()
    delta = j.model.model.rope_deltas.clone()
    prefix_seconds = tick() - start
    generator = torch.Generator(device=j.device).manual_seed(0)
    native, optimized, speech, traces = [], [], [], []
    query_seconds = speech_seconds = latent_seconds = diagnostic_seconds = 0.
    for i, ((a, b), t) in enumerate(zip(wins, texts)):
        question = yesno_question(i, len(wins), a, b, t, 'visual')
        bids, suffix = j.branch_ids(msgs, question, history, head_text=head)
        qrows = question_rows(j, question, suffix, bids)
        attention = QueryAttention(qrows, image_positions, len(bids), j.model.config.text_config.num_hidden_layers)
        j.model.model.rope_deltas = delta.clone()
        t0 = tick()
        with attention.capture():
            h = j._step(cache, bids)
        z = j.margins_fp32(h[None])[0]
        native.append(z)
        relevance = attention.relevance()
        query_seconds += tick() - t0
        qn = cache.get_seq_length()
        assert qn == n + len(bids)
        qdelta = j.model.model.rope_deltas.clone()
        t0 = tick()
        v, detail = reader.read(cache, embeddings, relevance, counts, qdelta, generator, arm=arm, smoke=smoke)
        diagnostic_seconds += detail.get('diagnostic_seconds', 0.)
        latent_seconds += tick() - t0 - detail.get('diagnostic_seconds', 0.)
        optimized.append(v)
        detail.update(window=i, question_rows=qrows, question_tokens=len(bids),
                      attention_layers=attention.layers, question_cache_length=qn)
        if smoke:
            detail['captured_attention_native_margin'] = z
        cache.crop(n)
        j.model.model.rope_deltas = delta.clone()
        if available[i]:
            bids_s, _ = j.branch_ids(msgs, yesno_question(i, len(wins), a, b, t, 'speech'), history, head_text=head)
            t0 = tick()
            s = j.cached_margin(cache, bids_s, in_place=True)
            cache.crop(n)
            j.model.model.rope_deltas = delta.clone()
            speech_seconds += tick() - t0
        else:
            s = None
        speech.append(s)
        if smoke:
            t0 = tick()
            replay = j.cached_margin(cache, bids, in_place=True)
            assert replay == z, ('native SDPA/cache changed', replay, z)
            detail['K0_native_replay_exact'] = True
            cache.crop(n)
            j.model.model.rope_deltas = delta.clone()
            diagnostic_seconds += tick() - t0
        assert cache.get_seq_length() == n
        traces.append(detail)
    extra_per_window = 1 if arm in ('initial', 'warmup') else SEARCH + 2
    assert j.forward_calls - first_call == 3 + B + extra_per_window * len(wins) + (3 * len(wins) if smoke else 0)
    standalone = dict(base=prefix_seconds + query_seconds + speech_seconds,
                      optimized=prefix_seconds + query_seconds + speech_seconds + latent_seconds)
    index = np.clip(((np.arange(math.ceil(duration * FPS)) + .5) / FPS // 8).astype(int), 0, len(wins) - 1)
    records = {}
    for name, values in (('base', native), ('optimized', optimized)):
        windows = []
        for i, ((a, b), v, s) in enumerate(zip(wins, values, speech)):
            w = dict(i=i, start=a, end=b, z_visual=v, z=max(v, s) if s is not None else v)
            if s is not None:
                w['z_speech'] = s
            windows.append(w)
        scores = np.asarray([w['z'] for w in windows])
        assert np.isfinite(scores).all()
        records[name] = dict(schema_version=1, method='m1_latents_' + name + '_' + arm,
                            dataset=ds, video_id=vid, duration=duration, native_rate=FPS,
                            score_curve=scores[index].tolist(), intervals=[], error=None, seed=0,
                            code_path=str(Path(__file__).relative_to(ROOT)),
                            calls=3 + B + (extra_per_window * len(wins) if name == 'optimized' else 0),
                            extra=dict(z_video=zv, stance=stance, prefix_tokens=P,
                                       standalone_seconds=standalone[name], n_branches=B, windows=windows))
    check = dict(dataset=ds, video_id=vid, visual_queries=len(wins), native_branches=B,
                 actual_forwards=j.forward_calls - first_call, prefix_tokens=P,
                 standalone_seconds=standalone, diagnostic_seconds=diagnostic_seconds,
                 peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                 native_replay_exact=True if smoke else None, paired_seconds=tick() - start)
    details = dict(traces=traces, image_counts=counts, original_frame_times=[ft for ft, _ in frames],
                   image_positions=image_positions.tolist(), merger_shape=list(embeddings.shape))
    del cache, embeddings, captured
    return records, check, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--arm', choices=('full', 'initial', 'warmup', 'search_only', 'wrong_support'), default='full')
    a = ap.parse_args()
    out = ROOT / 'runs/20261004_m1_latents' / ('r1_' + a.arm + ('_smoke' if a.smoke else '_main'))
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(out / 'run.log'), logging.StreamHandler(sys.stdout)])
    logging.info('host %s', socket.gethostname())
    (out / 'run.pid').write_text(str(os.getpid()))
    torch.manual_seed(0)
    rows = load_manifest(ROOT / 'data/omsl_v6_inputs/manifests/all_test.jsonl', ['HateMM', 'HateClipSeg'])
    if a.smoke:
        rows = [r for ds in ('HateMM', 'HateClipSeg') for r in [x for x in rows if x['dataset'] == ds][:2]] + [
            r for r in rows if r['dataset'] == 'HateMM' and r['video_id'] == 'hate_video_114']
    else:
        assert len(rows) == 333
        assert sum(r['dataset'] == 'HateMM' for r in rows) == 215
        assert len({(r['dataset'], r['video_id']) for r in rows}) == 333
    asr = {ds: load_asr(ds) for ds in ('HateMM', 'HateClipSeg')}
    j = Judge(MODEL)
    reader = LatentReader(j)
    j.forward_calls = 0
    def count(*_):
        j.forward_calls += 1
    counter = j.model.model.register_forward_pre_hook(count)
    import transformers
    config = dict(date=time.strftime('%Y-%m-%d'), host=socket.gethostname(), arm=a.arm, smoke=a.smoke,
                  model=MODEL, torch=torch.__version__, transformers=transformers.__version__, seed=0,
                  GT_in_reader=False, code='experiments/20261004_m1_latents/{measure,latents}.py; sources 2026-10-04',
                  source_definition='paper Eq2-6 adaptation; distinct from official contextual Stage I',
                  K=K, positive=POS, negative=NEG, warmup=WARMUP, search=SEARCH, top_delta=TOP,
                  tau=TAU, lr=LR, sigma=SIGMA, decay=DECAY, alpha=ALPHA, NES_index_origin=0,
                  attention='post-RoPE actual causal probabilities, all-visible-key denominator, question text rows',
                  best_state='actual evaluated candidate; tie retains earlier', window_seconds=8, fps=FPS,
                  global_and_speech='native unchanged', frame_limit=20,
                  timing_note='base includes read-only attention capture; optimized charges all latent calls')
    config_path = out / 'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        assert {k: v for k, v in previous.items() if k != 'date'} == {k: v for k, v in config.items() if k != 'date'}, 'incompatible existing run config'
    else:
        assert not any((out / name / 'predictions.jsonl').exists() for name in ('base', 'optimized'))
        config_path.write_text(json.dumps(config, indent=2) + '\n')
    expected = {(r['dataset'], r['video_id']) for r in rows}
    handles, done = {}, {}
    for name in ('base', 'optimized'):
        d = out / name
        d.mkdir(exist_ok=True)
        arm_config = {**json.loads(config_path.read_text()), 'prediction_arm': name}
        if (d / 'config.json').exists():
            assert json.loads((d / 'config.json').read_text()) == arm_config
        else:
            (d / 'config.json').write_text(json.dumps(arm_config, indent=2) + '\n')
        p = d / 'predictions.jsonl'
        done[name] = existing_records(p, expected)
    cp = out / 'checks.jsonl'
    seen = existing_records(cp, expected)
    assert done['base'].keys() == done['optimized'].keys() == seen.keys(), 'incomplete paired record'
    for key in seen:
        detail = json.loads((out / 'details' / key[0] / (key[1] + '.json')).read_text())
        validate_pair(done['base'][key], done['optimized'][key], seen[key], detail)
    for name in ('base', 'optimized'):
        handles[name] = (out / name / 'predictions.jsonl').open('a')
    checked = cp.open('a')
    started = time.time()
    for k, row in enumerate(rows):
        key = row['dataset'], row['video_id']
        if key in seen:
            continue
        recs, check, details = read_video(j, reader, row, asr[key[0]].get(key[1], []), a.arm, a.smoke)
        validate_pair(recs['base'], recs['optimized'], check, details)
        dest = out / 'details' / key[0]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / (key[1] + '.json')).write_text(json.dumps(details) + '\n')
        for name, r in recs.items():
            handles[name].write(json.dumps(r) + '\n')
            handles[name].flush()
        checked.write(json.dumps(check) + '\n')
        checked.flush()
        logging.info('progress %d/%d %s %s elapsed=%.1f peak_GiB=%.2f latent_calls=%d',
                     k + 1, len(rows), *key, time.time() - started, check['peak_GiB'],
                     (SEARCH + 2 if a.arm not in ('initial', 'warmup') else 1) * check['visual_queries'])
    for f in handles.values():
        f.close()
    checked.close()
    counter.remove()
    logging.info('RUN_DONE videos=%d elapsed=%.1f', len(rows), time.time() - started)


if __name__ == '__main__':
    main()
