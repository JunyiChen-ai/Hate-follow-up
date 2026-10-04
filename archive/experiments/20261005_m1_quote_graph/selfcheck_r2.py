#!/usr/bin/env python3
"""Actual fixed-five source/token binding checks; no labels or alternative scores."""
import json
from measure_r2 import bind_context
from extract import ROOT, CACHE, selected_rows, validate
from reader import compile_branch
from src.mllm_renderer import cpu_renderer
from src.mllm_judge import yesno_question
from src.video_inputs import load_asr


def main():
    renderer = cpu_renderer()
    asr = {ds: load_asr(ds) for ds in ('HateMM', 'HateClipSeg')}
    out = ROOT / 'runs/20261005_m1_quote_graph/r2_cpu_checks'
    out.mkdir(parents=True, exist_ok=True)
    result = dict(GT_read=False, alternative_scores_computed=False, videos=[])
    for row in selected_rows(True):
        ds, video = row['dataset'], row['video_id']
        metadata = json.loads((CACHE / ds / (video + '.json')).read_text())
        original = json.loads((ROOT / 'runs/20261005_m1_quote_graph/r1_full_smoke/records' / ds / (video + '.json')).read_text())
        segments = asr[ds].get(video, [])
        validate(metadata, row, segments, renderer)
        ctx, native = bind_context(renderer, row, segments, original['base']['extra']['stance'])
        assert native == original['native_input']
        counts = dict(native_fallback=0, graph=0, absent=0)
        for window, packet, trace in zip(metadata['windows'], metadata['packets'], original['traces']):
            if not window['body'].strip():
                counts['absent'] += 1
            elif not packet['selected']:
                question = yesno_question(window['i'], len(metadata['windows']), window['start'], window['end'], window['body'], 'speech')
                ids, suffix = renderer.branch_ids(ctx['msgs'], question, ctx['history'], head_text=ctx['head'])
                assert renderer.tok.encode(suffix, add_special_tokens=False) == ids
                assert window['body'] in question
                counts['native_fallback'] += 1
            else:
                ids, expected = compile_branch(renderer, ctx, metadata['windows'], packet)
                assert trace['ids'] == ids
                assert all(trace[k] == v for k, v in expected.items())
                counts['graph'] += 1
        result['videos'].append(dict(dataset=ds, video_id=video, native_tokens=len(native['ids']),
                                     overview_images=len(native['files']), source_packets=len(metadata['packets']), **counts))
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    print('CPU_CHECKS_DONE', flush=True)


if __name__ == '__main__':
    main()
