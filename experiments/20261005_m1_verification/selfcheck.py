#!/usr/bin/env python3
"""Meaningful replacement, independent input, ownership and actual-source checks."""
import copy
import json
import logging
import socket
import sys
from PIL import Image
from inputs import ROOT, selected_rows, windows_for, overview_for, original_content, planner_content
from interface import SPEC, FIELDS, CONSTANTS, canonical, model_visible, catalog, branch_record
from interface import write_draft, write_plan, write_field, compile_draft, compile_plan, compile_verification, replace_fields
from src.mllm_renderer import cpu_renderer
from src.video_inputs import load_asr, fixed_windows, window_text
from src.actual_video_frames import validate_frames
from src.structured_source_generation import Stream, image_rope_delta


def rejected(fn):
    try:
        fn()
    except (AssertionError, ValueError, KeyError, IndexError, TypeError):
        return True
    raise AssertionError('Corrupt record accepted')


def raw_generation(value):
    return dict(truncated=False, selection=value, text=canonical(value))


def grammar_fixture(j, writer, choices, descriptions):
    """Synthetic author choices; replay actual production grammar, no model claim."""
    class Tape:
        def __init__(self):
            self.tokens = []
            self.events = []
            self.choices = iter(choices)
            self.descriptions = iter(descriptions)

        def force(self, value):
            self.tokens += j.tok.encode(value, add_special_tokens=False)

        def choose(self, options):
            value = next(self.choices)
            assert value in options
            self.force(value)
            return value

        def description(self):
            value = next(self.descriptions)
            self.force(value)
            self.force('"')
            self.events.append(dict(text=value))

    tape = Tape()
    expected = writer(tape)
    stream = Stream(j, 1024, tokens=tape.tokens)
    actual = writer(stream)
    assert actual == expected and stream.tokens == tape.tokens
    return raw_generation(actual)


def logical_checks(j):
    count = 0
    w = dict(i=0, start=0., end=8., body='A person discusses a quoted target', frames=[dict(id='p0', path='internal/HateMM/hate_video_fixture.png', index=2, time=2.)])
    sources = catalog(w)
    assert sources and all(r['owner'] == 0 for r in sources.values())
    count += 1
    for body, bad in [('same words same words', 'same words'), ('a a a', 'a a')]:
        assert not any(r.get('quote') == bad for r in catalog(dict(w, body=body)).values())
        count += 1
    choices = ['true', canonical('w0000:p0'), ']', 'false', 'true', canonical('w0000:t0001_0002'), ']', 'false']
    dg = grammar_fixture(j, lambda s: write_draft(s, sources), choices, ['DRAFT_ACT_MARKER', 'DRAFT_TARGET_MARKER'])
    draft = compile_draft(dg, sources)
    assert [r['known'] for r in draft['fields']] == [True, False, True, False]
    count += 1
    pg = grammar_fixture(j, lambda s: write_plan(s, True), [canonical('act'), ',', canonical('target')], ['What local act occurs?', 'Who is actually addressed?'])
    plan = compile_plan(pg, True)
    assert [q['field'] for q in plan['questions']] == ['act', 'target']
    count += 1
    vg = grammar_fixture(j, lambda s: write_field(s, 'act', sources), ['true', canonical('w0000:p0'), ']'], ['ACTUAL_OBSERVATION_MARKER'])
    observation = compile_verification(vg, 'act', sources)
    unknown = compile_verification(dict(truncated=True, selection=None), 'target', sources)
    compiled, changes = replace_fields(draft, plan, [observation, unknown])
    assert compiled['fields'][0]['value'] == 'ACTUAL_OBSERVATION_MARKER' and not compiled['fields'][2]['known']
    assert compiled['fields'][1] == draft['fields'][1] and compiled['fields'][3] == draft['fields'][3]
    assert all(c['literal_changed'] for c in changes)
    count += 1
    no_plan = compile_plan(dict(truncated=True, selection=None), True)
    assert replace_fields(draft, no_plan, [])[0]['fields'] == draft['fields']
    count += 1
    assert all(not f['known'] for f in compile_draft(dict(truncated=True, selection=None), sources)['fields'])
    count += 1
    empty = dict(name='act', known=True, value='', witnesses=['w0000:p0'])
    assert not compile_verification(raw_generation(empty), 'act', sources)['known']
    count += 1
    for update in [dict(name='actor'), dict(known=1), dict(witnesses=['w0001:p0']), dict(witnesses=[]), dict(witnesses=['w0000:p0', 'w0000:p0']), dict(value='x ' * 25), dict(known=False, value='not UNKNOWN')]:
        raw = dict(name='act', known=True, value='Observed', witnesses=['w0000:p0'])
        raw.update(update)
        rejected(lambda: compile_verification(raw_generation(raw), 'act', sources))
        count += 1
    for questions in [[dict(field='act', question='a'), dict(field='act', question='b')], [dict(field='invalid', question='a')], []]:
        rejected(lambda: compile_plan(raw_generation(dict(questions=questions)), True))
        count += 1
    assert compile_plan(raw_generation(dict(questions=[dict(field='act', question='')])), True)['error'] == 'empty_question'
    count += 1
    assert compile_plan(grammar_fixture(j, lambda s: write_plan(s, False), [], []), False)['questions'] == []
    count += 1
    current = dict(windows=[w], records=[dict(compiled=compiled)])
    branch = canonical(model_visible(branch_record(current, 0, 'visual')))
    assert 'DRAFT_ACT_MARKER' not in branch and 'DRAFT_TARGET_MARKER' not in branch and 'ACTUAL_OBSERVATION_MARKER' in branch and 'hate_video_fixture' not in branch
    count += 1
    overview = [dict(id='overview0', time=0., path='internal/HateMM/hate_video_overview.png')]
    content, paths = original_content(overview, [], w, 'verification', plan['questions'][0])
    visible = canonical(content)
    assert all(x not in visible for x in ['DRAFT_ACT_MARKER', 'DRAFT_TARGET_MARKER', 'ACTUAL_OBSERVATION_MARKER', 'hate_video_fixture', 'hate_video_overview'])
    assert paths == [overview[0]['path'], w['frames'][0]['path']]
    assert 'DRAFT_ACT_MARKER' in canonical(planner_content(w, draft))
    count += 1
    return count


def main():
    out = ROOT / 'runs/20261005_m1_verification/cpu_checks'
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler(out / 'run.log'), logging.StreamHandler(sys.stdout)])
    logging.info('host %s', socket.gethostname())
    (out / 'run.pid').write_text(str(__import__('os').getpid()))
    cfg = dict(GT_read=False, host=socket.gethostname(), code='experiments/20261005_m1_verification/selfcheck.py; sources2026-10-05', command='python -u ' + ' '.join(sys.argv))
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    j = cpu_renderer()
    checks = logical_checks(j)
    asr = {ds: load_asr(ds) for ds in ('HateMM', 'HateClipSeg')}
    windows_count = total_spans = 0
    largest = None
    for row in selected_rows(False):
        segments = asr[row['dataset']].get(row['video_id'], [])
        for i, (a, b) in enumerate(fixed_windows(float(row['duration']), 8)):
            w = dict(i=i, start=a, end=b, body=window_text(segments, a, b), frames=[])
            spans = len(catalog(w))
            windows_count += 1
            total_spans += spans
            if largest is None or spans > largest['spans']:
                largest = dict(dataset=row['dataset'], video_id=row['video_id'], window=i, spans=spans, body_chars=len(w['body']))
    sizes = []
    for row in selected_rows(True):
        # Only actual pixels/rawPTS reused for CPU templates, no prior model observation.
        folder = ROOT / 'data/temporal_entity_discourse_graph' / row['dataset'] / row['video_id']
        previous = json.loads((folder / 'metadata.json').read_text())
        validate_frames(previous['source'], row, folder / 'frames')
        segments = asr[row['dataset']].get(row['video_id'], [])
        windows = windows_for(row, segments, previous['source'], folder)
        overview = overview_for(row)
        q = dict(field='act', question='What actual local act is observable?')
        for stage in ('draft', 'verification'):
            content, paths = original_content(overview, [list(s) for s in segments], windows[0], stage, q if stage == 'verification' else None)
            text = j.render([j.turn('system', SPEC[stage + '_system']), dict(role='user', content=content)], True)
            assert row['video_id'] not in text and '.png' not in text and '/HateMM/' not in text and '/HateClipSeg/' not in text
            images = [Image.open(ROOT / p).convert('RGB') for p in paths]
            try:
                enc = j.encode(text, images)
            finally:
                for im in images:
                    im.close()
            sizes.append(dict(dataset=row['dataset'], video_id=row['video_id'], stage=stage, images=len(paths), input_tokens=len(enc['input_ids'][0]), rope_delta=image_rope_delta(j, enc), source_checked=str(folder.relative_to(ROOT))))
            checks += 1
            logging.info('actual %s/%s %s images=%d tokens=%d', row['dataset'], row['video_id'], stage, len(paths), len(enc['input_ids'][0]))
    result = dict(**cfg, checks=checks, all_source_windows=windows_count, total_legal_text_spans=total_spans, largest=largest, actual_fixed5_source_inputs=sizes,
        GPU_run=False, semantic_claim=False, fixture_note='Author grammar choices are synthetic; actual5 source pixels/rawPTS/current prompts checked. No model semantics, correctness or performance measured.')
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    logging.info('CPU_CHECKS_DONE checks=%d windows=%d', checks, windows_count)


if __name__ == '__main__':
    main()
