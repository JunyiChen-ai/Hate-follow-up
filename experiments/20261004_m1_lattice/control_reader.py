"""Declared, independent speech-lattice controls; no annotations or scores as inputs."""
import copy
import torch

from lattice import graph_bias, graph_positions, TAIL
from reader import compile_branch, structural

ARMS = ('full', 'onebest', 'flat', 'binary', 'wrong_mass', 'wrong_audio_window')
CONTROL_VERSION = 'R1 lattice matched controls, sources2026-10-05'


def donor_map(windows):
    available = [w['i'] for w in windows if w['available']]
    if not available:
        return {}
    return {i: available[(k + len(available) // 2) % len(available)]
            for k, i in enumerate(available)}


def compile_control(j, ctx, window, nwindows, arm, donor=None):
    assert arm in ARMS
    source = donor if arm == 'wrong_audio_window' else window
    assert source is not None and source['available'] and window['available']
    ids, graph, h, t, trace = compile_branch(j, ctx, source, nwindows)
    trace = copy.deepcopy(trace)
    graph = copy.deepcopy(graph)
    head_ids, tail_ids = trace['head_tokens'], trace['tail_tokens']
    coverage = dict(source_binding_changed=False, mass_slots_changed=0,
                    mass_slots_singleton=0, mass_slots_zero=0)
    if arm == 'onebest':
        text = window['confusion']['onebest']
        words = j.tok.encode(' ' + text, add_special_tokens=False) if text else []
        graph = dict(ids=words, path_length=len(words),
            slots=[dict(alternatives=[dict(text=text, mass=1.)])] if words else [],
            nodes=[dict(slot=0, alternative=0, subword=k, logical=k, mass=1., physical=k)
                   for k in range(len(words))])
    elif arm == 'wrong_mass':
        for slot in graph['slots']:
            options = slot['alternatives']
            if len(options) <= 1:
                coverage['mass_slots_singleton'] += 1
                continue
            masses = [a['mass'] for a in options]
            if any(m == 0 for m in masses):
                coverage['mass_slots_zero'] += 1
                continue
            shift = len(options) // 2
            rotated = masses[shift:] + masses[:shift]
            coverage['mass_slots_changed'] += masses != rotated
            for alternative, mass in zip(options, rotated):
                alternative['mass'] = mass
        for node in graph['nodes']:
            node['mass'] = graph['slots'][node['slot']]['alternatives'][node['alternative']]['mass']
    elif arm == 'wrong_audio_window' and donor['i'] != window['i']:
        coverage['source_binding_changed'] = True
        instruction = (
            f"Judge destination window {window['i'] + 1} of {nwindows}, from {window['start']:.1f}s "
            f"to {window['end']:.1f}s of this video. The transcription above comes from source "
            f"window {donor['i'] + 1} of {nwindows}, from {donor['start']:.1f}s to {donor['end']:.1f}s.\n")
        assert trace['tail_text'].count(TAIL) == 1
        trace['tail_text'] = trace['tail_text'].replace(TAIL, '\n\n' + instruction + TAIL)
        tail_ids = j.tok.encode(trace['tail_text'], add_special_tokens=False)
        t = len(tail_ids)
    ids = head_ids + graph['ids'] + tail_ids
    trace.update(arm=arm, graph=graph, head_tokens=head_ids, tail_tokens=tail_ids,
                 control_version=CONTROL_VERSION, destination_window=window['i'],
                 source_window=source['i'], source_nominal_interval=[source['start'], source['end']],
                 source_actual_interval=source['crop']['actual_interval'], coverage=coverage)
    return ids, graph, h, t, trace


@torch.no_grad()
def control_margin(j, cache, ctx, window, nwindows, arm, donor=None):
    ids, graph, h, t, trace = compile_control(j, ctx, window, nwindows, arm, donor)
    n = cache.get_seq_length()
    start = n + int(ctx['rope'][0, 0])
    assert start == int(ctx['positions'].max()) + 1
    position_arm = 'flat' if arm in ('flat', 'onebest') else 'full'
    bias_arm = arm if arm in ('flat', 'binary') else 'full'
    logical = graph_positions(graph, h, t, start, position_arm)
    if arm == 'full':
        z, original = structural(j, cache, ctx, window, nwindows)
        assert all(original[k] == trace[k] for k in ('graph', 'head_tokens', 'tail_tokens'))
    else:
        j.model.model.rope_deltas = ctx['rope'].clone()
        try:
            if arm == 'onebest':
                z = j.cached_margin(cache, ids, in_place=True)
            else:
                bias = graph_bias(graph, h, t, n, bias_arm)
                j.model.model.rope_deltas = torch.tensor(
                    [[logical[-1] + 1 - (n + len(ids))]], device=j.device)
                out = j.model.model(input_ids=torch.tensor([ids], device=j.device),
                    past_key_values=cache, use_cache=True,
                    position_ids=torch.tensor(logical, device=j.device)[None, None, :].expand(3, 1, -1),
                    attention_mask=torch.as_tensor(bias, device=j.device, dtype=j.model.dtype)[None, None, :, :])
                hidden = out.last_hidden_state[0, -1]
                del out
                z = j.margins_fp32(hidden[None])[0]
        finally:
            cache.crop(n)
            j.model.model.rope_deltas = ctx['rope'].clone()
    assert cache.get_seq_length() == n and torch.equal(j.model.model.rope_deltas, ctx['rope'])
    trace.update(margin=z, physical_tokens=len(ids), logical_positions=logical,
                 prefix_tokens=n, prefix_logical_start=start, graph_tokens=len(graph['ids']),
                 slot_count=len(graph['slots']), position_arm=position_arm, bias_arm=bias_arm,
                 ordinary_serial=arm == 'onebest', attention_dtype=str(j.model.dtype))
    return z, trace
