"""Independent official DTW path, probability and tokenizer coverage checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from alignment import dtw,align_weights,word_groups,timed_words


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    from transformers import AutoProcessor
    from transformers.models.whisper.generation_whisper import _dynamic_time_warping,_median_filter
    generator=np.random.default_rng(0);checks=0
    for shape in [(1,1),(1,20),(20,1),(7,19),(30,100)]:
        for kind in ('zero','integer','normal'):
            matrix=np.zeros(shape) if kind=='zero' else (generator.integers(-2,3,shape) if kind=='integer' else generator.normal(size=shape))
            actual=dtw(matrix);expected=_dynamic_time_warping(matrix)
            assert all(np.array_equal(a,b) for a,b in zip(actual,expected)),(shape,kind)
            checks+=1
    tokens=12;frames=17;prefix=4;heads=[(0,1),(1,0)]
    torch.manual_seed(0);weights=[torch.randn(1,2,prefix+tokens+1,frames).softmax(-1) for _ in range(2)]
    actual,jumps=align_weights(weights,heads,prefix,tokens,frames)
    reference=torch.stack([weights[l][0,h].float() for l,h in heads])
    std,mean=torch.std_mean(reference,dim=-2,keepdim=True,unbiased=False)
    reference=_median_filter(((reference-mean)/std).unsqueeze(0),7)[0].mean(0)[prefix-1:-1].numpy()
    assert np.array_equal(actual,reference)
    rows,times=_dynamic_time_warping(-reference);expected=times[np.r_[True,np.diff(rows)!=0]]*.02
    assert np.array_equal(jumps,expected)
    zeros=[torch.full((1,1,7,2),.5)]
    matrix,jumps_zero=align_weights(zeros,[(0,0)],4,2,2);assert not matrix.any() and np.isfinite(jumps_zero).all()
    capped=[torch.randn(1,1,448,10).softmax(-1)]
    matrix,jumps_cap=align_weights(capped,[(0,0)],4,444,10,teacher_eos=False)
    assert matrix.shape==(445,10) and len(jumps_cap)==445,'capped token prefix was dropped or expanded past448'
    processor=AutoProcessor.from_pretrained('openai/whisper-large-v3',local_files_only=True)
    examples=[]
    for language,text in [('en','Hello, café. Another speaker says no.'),('zh','你好 世界'),('en',' repeated repeated')]:
        ids=processor.tokenizer.encode(text,add_special_tokens=False)
        words,groups,indices=word_groups(processor.tokenizer,ids,language)
        raw=np.arange(len(ids)+1)*.02
        aligned=timed_words(processor.tokenizer,ids,language,raw,30,30.01,1)
        assert [x for w in aligned for x in w['tokens']]==ids and ''.join(w['text'] for w in aligned)==text
        assert all(30<=w['start']<=w['end']<=30.01 for w in aligned)
        assert any(w['raw_dtw'][1]>30.01 for w in aligned)
        examples.append(dict(language=language,text=text,words=words,token_coverage=len(ids)))
    summary=dict(PASS=True,official_path_exact_examples=checks,official_probability_matrix_exact=True,
        zero_std_short_audio_finite=True,real_tokenizer_literal_coverage=examples,
        raw_and_clipped_times_preserved=True,scope='CPU primitives/real tokenizer only; no actual Whisper/Qwen forwards or performance')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print('PASS official DTW/attention/token/source time checks')


if __name__=='__main__':main()
