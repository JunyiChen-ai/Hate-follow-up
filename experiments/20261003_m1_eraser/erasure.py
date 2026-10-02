"""Deterministic raw-media removal; no model, scores or labels."""

def erase_frames(frames,start,end,last=False):
    indices=[i for i,(t,_) in enumerate(frames) if start<=t and (t<end or (last and t<=end))]
    removed=set(indices)
    return [x for i,x in enumerate(frames) if i not in removed],indices


def erase_speech(segments,start,end):
    kept=[];deleted=[];count=0
    for k,(s,e,text) in enumerate(segments):
        lo,hi=max(s,start),min(e,end)
        words=text.split()
        if hi<=lo or not words:
            kept.append((s,e,text));continue
        a=max(0,min(len(words),int(round((lo-s)/(e-s)*len(words)))))
        b=max(0,min(len(words),int(round((hi-s)/(e-s)*len(words)))))
        if a==b:
            kept.append((s,e,text));continue
        deleted.append({'segment':k,'start':s,'end':e,'word_indices':list(range(a,b)),
                        'words':words[a:b]})
        count+=b-a
        rest=words[:a]+words[b:]
        if rest:kept.append((s,e,' '.join(rest)))
    return kept,deleted,count
