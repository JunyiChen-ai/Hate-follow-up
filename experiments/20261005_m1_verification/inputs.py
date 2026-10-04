"""Current source-bound prompts, independent execution and complete replay."""
from pathlib import Path
import math
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'CLAUDE.md').is_file())
sys.path.insert(0, str(ROOT))
from interface import SPEC, VERSION, CONSTANTS, FIELDS, canonical, model_visible, catalog, source_table
from interface import write_draft, write_plan, write_field, compile_draft, compile_plan, compile_verification, replace_fields
from src.video_inputs import load_manifest, frame_paths, fixed_windows, window_text
from src.actual_video_frames import validate_frames
from src.mllm_judge import MODEL

DATASETS = ('HateMM', 'HateClipSeg')
CACHE = ROOT / SPEC['input_cache']


def selected_rows(smoke=False):
    rows = load_manifest(ROOT / 'data/omsl_v6_inputs/manifests/all_test.jsonl', DATASETS)
    if smoke:
        rows = [r for ds in DATASETS for r in [x for x in rows if x['dataset'] == ds][:2]] + [r for r in rows if r['dataset'] == 'HateMM' and r['video_id'] == 'hate_video_114']
    assert len(rows) == (5 if smoke else 333)
    return rows


def windows_for(row, segments, source, folder):
    entries = {e['index']: e for e in source['entries']}
    windows = []
    for i, ((a, b), indices) in enumerate(zip(fixed_windows(float(row['duration']), 8), source['selected_indices'])):
        frames = [dict(id=f'p{k}', path=str((folder / 'frames' / f'frame_{index:08d}.png').relative_to(ROOT)), **entries[index]) for k, index in enumerate(indices)]
        windows.append(dict(i=i, start=a, end=b, body=window_text(segments, a, b), frames=frames))
    return windows


def overview_for(row):
    return [dict(id=f'overview{k}', time=t, path=str(p.relative_to(ROOT)), time_kind='nominal overview only') for k, (t, p) in enumerate(frame_paths(row['dataset'], row['video_id'], 20))]


def original_content(overview, segments, window, stage, question=None):
    assert stage in ('draft', 'verification') and ((question is not None) == (stage == 'verification'))
    content = [dict(type='text', text='Original native overview for interpretation; local sources below own occurrences.\n')]
    paths = []
    for frame in overview:
        content.extend([dict(type='text', text=canonical(model_visible(frame)) + '\n'), dict(type='image')])
        paths.append(frame['path'])
    content.append(dict(type='text', text='Original full ASR context:\n' + canonical(segments) + '\nActual LOCAL frames:\n'))
    for frame in window['frames']:
        content.extend([dict(type='text', text=canonical(model_visible(frame)) + '\n'), dict(type='image')])
        paths.append(frame['path'])
    content.append(dict(type='text', text='Complete LOCAL source catalog:\n' + canonical(source_table(window)) + '\n'))
    if question is not None:
        assert set(question) == {'field', 'question'} and question['field'] in FIELDS
        content.append(dict(type='text', text='This independent question:\n' + canonical(question) + '\n'))
    content.append(dict(type='text', text=SPEC[stage + '_instruction']))
    return content, paths


def planner_content(window, draft):
    return [dict(type='text', text='Current neutral draft:\n' + canonical(draft) + '\nCurrent source availability:\n' + canonical(source_table(window)) + '\n' + SPEC['planner_instruction'])]


def validate(m, row, segments, j=None):
    from src.structured_source_generation import validate_generation
    folder = CACHE / row['dataset'] / row['video_id']
    assert (m['version'], m['constants'], m['spec'], m['model']) == (VERSION, CONSTANTS, SPEC, MODEL)
    assert (m['dataset'], m['video_id'], m['duration']) == (row['dataset'], row['video_id'], float(row['duration']))
    assert m['GT_read'] is False and m['segments'] == [list(s) for s in segments]
    validate_frames(m['source'], row, folder / 'frames')
    windows = windows_for(row, segments, m['source'], folder)
    assert m['windows'] == windows and m['overview'] == overview_for(row) and len(m['records']) == len(windows)
    generations = []
    for w, r in zip(windows, m['records']):
        assert r['window'] == w['i']
        sources = catalog(w)
        available = bool(w['frames'] or w['body'].strip())
        draft = compile_draft(r['draft']['generation'], sources)
        assert draft == r['draft']['record']
        plan = compile_plan(r['plan']['generation'], available)
        assert plan == r['plan']['record'] and len(r['verifications']) == len(plan['questions'])
        generations += [r['draft']['generation'], r['plan']['generation']]
        if j:
            content, paths = original_content(m['overview'], m['segments'], w, 'draft')
            validate_generation(j, ROOT, r['draft']['generation'], SPEC['draft_system'], content, paths, CONSTANTS['draft_tokens'], lambda s: write_draft(s, sources))
            validate_generation(j, ROOT, r['plan']['generation'], SPEC['planner_system'], planner_content(w, draft), [], CONSTANTS['planner_tokens'], lambda s: write_plan(s, available))
        observed = []
        for q, v in zip(plan['questions'], r['verifications']):
            assert v['question'] == q
            field = compile_verification(v['generation'], q['field'], sources)
            assert field == v['record']
            observed.append(field)
            generations.append(v['generation'])
            if j:
                content, paths = original_content(m['overview'], m['segments'], w, 'verification', q)
                validate_generation(j, ROOT, v['generation'], SPEC['verification_system'], content, paths, CONSTANTS['verification_tokens'], lambda s: write_field(s, q['field'], sources))
        compiled, changes = replace_fields(draft, plan, observed)
        assert r['compiled'] == compiled and r['changes'] == changes
    assert m['actual_forwards'] == sum(g['actual_forwards'] for g in generations)
    assert m['actual_vision_forwards'] == sum(g['actual_vision_forwards'] for g in generations)
    assert abs(m['generation_seconds'] - sum(g['seconds'] for g in generations)) < 1e-6
    assert m['standalone_seconds'] >= m['generation_seconds'] + m['source']['decode_seconds']
    assert all(math.isfinite(m[k]) and m[k] >= 0 for k in ('standalone_seconds', 'generation_seconds', 'peak_GiB'))
    return m
