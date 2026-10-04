#!/usr/bin/env python3
"""Acquire the complete draft/plan/independent-answer/replacement pipeline."""
import argparse
import json
import logging
import socket
import sys
import time
import torch
from inputs import ROOT, CACHE, DATASETS, selected_rows, windows_for, overview_for, original_content, planner_content, validate
from interface import SPEC, VERSION, CONSTANTS, catalog, write_draft, write_plan, write_field
from interface import compile_draft, compile_plan, compile_verification, replace_fields
from src.actual_video_frames import acquire_window_frames
from src.structured_source_generation import generate
from src.source_generation import clock
from src.mllm_judge import Judge, MODEL
from src.video_inputs import load_asr


@torch.no_grad()
def acquire(j, row, segments, folder):
    torch.cuda.reset_peak_memory_stats()
    start = clock(j)
    before, before_vision = j.forward_calls, j.vision_calls
    source = acquire_window_frames(row, folder / 'frames')
    windows = windows_for(row, segments, source, folder)
    overview = overview_for(row)
    records, generations = [], []
    for w in windows:
        sources = catalog(w)
        content, paths = original_content(overview, segments, w, 'draft')
        dg = generate(j, ROOT, SPEC['draft_system'], content, paths, CONSTANTS['draft_tokens'], lambda s: write_draft(s, sources))
        draft = compile_draft(dg, sources)
        available = bool(w['frames'] or w['body'].strip())
        pg = generate(j, ROOT, SPEC['planner_system'], planner_content(w, draft), [], CONSTANTS['planner_tokens'], lambda s: write_plan(s, available))
        plan = compile_plan(pg, available)
        verifications = []
        generations += [dg, pg]
        for q in plan['questions']:
            # Rebuild original input only: no draft record, prior answers or KV.
            content, paths = original_content(overview, segments, w, 'verification', q)
            vg = generate(j, ROOT, SPEC['verification_system'], content, paths, CONSTANTS['verification_tokens'], lambda s: write_field(s, q['field'], sources))
            observed = compile_verification(vg, q['field'], sources)
            verifications.append(dict(question=q, generation=vg, record=observed))
            generations.append(vg)
        compiled, changes = replace_fields(draft, plan, [v['record'] for v in verifications])
        records.append(dict(window=w['i'], draft=dict(generation=dg, record=draft), plan=dict(generation=pg, record=plan), verifications=verifications, compiled=compiled, changes=changes))
    return dict(version=VERSION, constants=CONSTANTS, spec=SPEC, model=MODEL, dataset=row['dataset'], video_id=row['video_id'], duration=float(row['duration']),
        GT_read=False, host=socket.gethostname(), date=time.strftime('%Y-%m-%d'), segments=[list(s) for s in segments], source=source, windows=windows, overview=overview,
        records=records, generation_seconds=sum(g['seconds'] for g in generations), standalone_seconds=clock(j) - start,
        actual_forwards=j.forward_calls - before, actual_vision_forwards=j.vision_calls - before_vision, peak_GiB=torch.cuda.max_memory_allocated() / 2**30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = ROOT / 'runs/20261005_m1_verification' / ('r1_extract_' + ('smoke' if a.smoke else 'main'))
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler(out / 'run.log'), logging.StreamHandler(sys.stdout)])
    logging.info('host %s', socket.gethostname())
    (out / 'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    cfg = dict(host=socket.gethostname(), date=time.strftime('%Y-%m-%d'), model=MODEL, version=VERSION, constants=CONSTANTS, spec=SPEC, GT_read=False, smoke=a.smoke,
        torch=torch.__version__, transformers=transformers.__version__, code='experiments/20261005_m1_verification/{interface,inputs,extract}.py + stable src; sources2026-10-05', command='python -u ' + ' '.join(sys.argv))
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    torch.manual_seed(0)
    j = Judge(MODEL)
    j.forward_calls = j.vision_calls = 0
    hooks = [j.model.model.register_forward_pre_hook(lambda *_: setattr(j, 'forward_calls', j.forward_calls + 1)), j.model.model.visual.register_forward_pre_hook(lambda *_: setattr(j, 'vision_calls', j.vision_calls + 1))]
    asr = {ds: load_asr(ds) for ds in DATASETS}
    rows = selected_rows(a.smoke)
    for number, row in enumerate(rows, 1):
        folder = CACHE / row['dataset'] / row['video_id']
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'metadata.json'
        segments = asr[row['dataset']].get(row['video_id'], [])
        if path.exists():
            m = json.loads(path.read_text())
            validate(m, row, segments, j)
            logging.info('%d/%d source reuse %s/%s', number, len(rows), row['dataset'], row['video_id'])
            continue
        m = acquire(j, row, segments, folder)
        validate(m, row, segments, j)
        temporary = folder / 'metadata.partial'
        temporary.write_text(json.dumps(m) + '\n')
        temporary.replace(path)
        logging.info('%d/%d %s/%s %.2fs verification_calls=%d changed=%d', number, len(rows), row['dataset'], row['video_id'], m['standalone_seconds'], sum(len(r['verifications']) for r in m['records']), sum(c['literal_changed'] for r in m['records'] for c in r['changes']))
    for hook in hooks:
        hook.remove()
    (CACHE / 'PROVENANCE.md').write_text('# Current local factual verification\n\nGenerated by experiments/20261005_m1_verification/extract.py + shared actual_video_frames/structured_source_generation; sources2026-10-05.\nFrozen Qwen/Qwen3-VL-8B-Instruct, no labels. Real raw video paths/PTS/PNG, proportional ASR crops, nominal overview times, all raw drafts/plans/independent observations/field replacements, tokens/grids/forwards/costs per metadata.json. Parser/ownership validation is not semantic proof.\nCommand ' + cfg['command'] + ' in allocated Slurm on ' + cfg['host'] + ', ' + cfg['date'] + '.\n')
    logging.info('EXTRACTION_DONE coverage=%d', len(rows))


if __name__ == '__main__':
    main()
