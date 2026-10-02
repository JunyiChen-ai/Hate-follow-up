"""Exact frame/transcript token spans for frozen-MLLM window attention interventions.

Promoted unchanged from the 2026-10-02 Grounder prototype; no labels.
"""
import re
import numpy as np


def span_mask(offsets, start, end):
    return (offsets[:,1] > start) & (offsets[:,0] < end)


def token_regions(judge, prefix_text, enc, frames, segments, windows):
    """Reproduce processor expansion, then align character spans to exact tokens."""
    image_token = judge.processor.image_token
    pieces = prefix_text.split(image_token)
    assert len(pieces) == len(frames)+1 == len(judge.img_tokens)+1
    expanded = pieces[0]
    for piece, count in zip(pieces[1:], judge.img_tokens):
        expanded += image_token*count + piece
    aligned = judge.tok(expanded, add_special_tokens=False, return_offsets_mapping=True)
    assert aligned["input_ids"] == enc["input_ids"][0].tolist(), "expanded-token alignment failed"
    offsets = np.asarray(aligned["offset_mapping"])
    P = len(offsets)
    visual = np.zeros(P,bool); speech = np.zeros(P,bool)
    local = [np.zeros(P,bool) for _ in windows]
    counts = []
    cursor = 0
    for frame_index, (t, _) in enumerate(frames):
        a = expanded.index(f"[t={t:.1f}s]\n", cursor)
        b = expanded.index("<|vision_end|>", a)+len("<|vision_end|>")
        mask = span_mask(offsets,a,b)
        visual |= mask
        nimage = int(np.sum(mask & (np.asarray(aligned["input_ids"]) == judge.image_token_id)))
        assert nimage == judge.img_tokens[frame_index], (nimage,judge.img_tokens[frame_index])
        for i,(s,e) in enumerate(windows):
            if s <= t < e or (i == len(windows)-1 and abs(t-e)<1e-6): local[i] |= mask
        counts.append(nimage); cursor=b
    a = expanded.index("\nTranscript:\n",cursor)+len("\nTranscript:\n")
    b = expanded.index("\nBased on this platform",a)
    speech |= span_mask(offsets,a,b)
    cursor = a
    for s,e,text in segments:
        if not text.strip():continue
        line = f"[{s:.1f}s-{e:.1f}s] {text.strip()}"
        start = expanded.index(line,cursor); end=start+len(line);cursor=end
        text_start = start+len(f"[{s:.1f}s-{e:.1f}s] ")
        words = list(re.finditer(r"\S+",text.strip()))
        for i,(w0,w1) in enumerate(windows):
            lo,hi=max(s,w0),min(e,w1)
            if hi<=lo or not words:continue
            j0=int(round((lo-s)/(e-s)*len(words)));j1=int(round((hi-s)/(e-s)*len(words)))
            if j1<=j0:continue
            local[i] |= span_mask(offsets,start,text_start)
            local[i] |= span_mask(offsets,text_start+words[j0].start(),text_start+words[j1-1].end())
    assert not np.any(visual & speech)
    scaffolding = ~(visual|speech)
    # Every selected local token must belong to one of the media regions.
    assert all(not np.any(m & scaffolding) for m in local)
    shifted=[]
    for keep in local:
        wrong=np.zeros(P,bool)
        for region in (visual,speech):
            ids=np.flatnonzero(region)
            wrong[ids]=np.roll(keep[ids],len(ids)//2)
        assert wrong.sum()==keep.sum()
        shifted.append(wrong)
    return {"local":local,"shifted":shifted,"scaffolding":scaffolding,
            "visual":visual,"speech":speech,"prefix_len":P,
            "image_counts":counts}

