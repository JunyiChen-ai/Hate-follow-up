"""Lexical block-cosine discourse partition; no labels or prediction scores."""
from collections import Counter
import math
from pathlib import Path
import json
import re
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
SPEC=json.loads((Path(__file__).parent/'spec.json').read_text())
LEXICAL=re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def cosine(left,right):
    a=Counter(left);b=Counter(right)
    denominator=math.sqrt(sum(x*x for x in a.values())*sum(x*x for x in b.values()))
    return sum(x*b.get(key,0) for key,x in a.items())/denominator if denominator else 0.


def partition(words):
    assert all(w['id']==i for i,w in enumerate(words))
    lexical=[dict(term=m.group().casefold(),word=w['id']) for w in words for m in LEXICAL.finditer(w['text'])]
    size=SPEC['pseudo_sentence_words'];k=SPEC['lexical_block_sentences']
    sentences=[lexical[i:i+size] for i in range(0,len(lexical),size)]
    gaps=list(range(k,len(sentences)-k+1))
    raw=[cosine([x['term'] for sentence in sentences[g-k:g] for x in sentence],
                [x['term'] for sentence in sentences[g:g+k] for x in sentence]) for g in gaps]
    radius=SPEC['smooth_width']//2
    smoothed=[sum(raw[max(0,i-radius):min(len(raw),i+radius+1)])/len(raw[max(0,i-radius):min(len(raw),i+radius+1)]) for i in range(len(raw))]
    depths=[];valleys=[]
    for i,value in enumerate(smoothed):
        left=right=i
        while left>0 and smoothed[left-1]>=smoothed[left]:left-=1
        while right+1<len(smoothed) and smoothed[right+1]>=smoothed[right]:right+=1
        depths.append((smoothed[left]-value)+(smoothed[right]-value))
        plateau=i
        while plateau>0 and smoothed[plateau-1]==value:plateau-=1
        # Only earliest member of a local-minimum plateau; endpoints permitted.
        end=i
        while end+1<len(smoothed) and smoothed[end+1]==value:end+=1
        valleys.append(plateau==i and (i==0 or smoothed[i-1]>value) and (end==len(smoothed)-1 or smoothed[end+1]>value))
    mean=sum(depths)/len(depths) if depths else 0.
    accepted=[]
    for index in sorted(range(len(gaps)),key=lambda i:(-depths[i],gaps[i])):
        gap=gaps[index]
        if valleys[index] and depths[index]>mean and all(abs(gap-old)>=SPEC['boundary_spacing_sentences'] for old in accepted):accepted.append(gap)
    boundaries=[]
    for gap in sorted(accepted):
        word=sentences[gap][0]['word']
        if sentences[gap-1][-1]['word']==word:word+=1
        if 0<word<len(words) and word not in boundaries:boundaries.append(word)
    bounds=[0]+sorted(boundaries)+[len(words)]
    segments=[dict(id=i,start_word=a,end_word=b) for i,(a,b) in enumerate(zip(bounds,bounds[1:])) if b>a]
    return dict(lexical=lexical,pseudo_sentence_word_ids=[[x['word'] for x in sentence] for sentence in sentences],
                gaps=gaps,cosine=raw,smoothed=smoothed,depths=depths,mean_depth=mean,
                accepted_pseudo_gaps=sorted(accepted),boundaries=sorted(boundaries),segments=segments)


def scope(words,segmentation,start,end):
    assert math.isfinite(start) and math.isfinite(end) and 0<=start<end
    local=[w['id'] for w in words if start<=(w['start']+w['end'])/2<end]
    local_set=set(local)
    touched=[s for s in segmentation['segments'] if any(s['start_word']<=i<s['end_word'] for i in local)]
    context=[i for s in touched for i in range(s['start_word'],s['end_word']) if i not in local_set]
    return dict(bounds=[start,end],local_ids=local,context_ids=context,segment_ids=[s['id'] for s in touched])
