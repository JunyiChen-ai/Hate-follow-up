"""Independent numeric/time/coverage examples for the declared lexical operator."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from partition import SPEC,cosine,partition,scope


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    examples=[]
    # Dense dot product is independent of Counter implementation.
    for a,b in [([],[]),(['a'],[]),(['a']*4+['b'],['b']*3+['c']),(['X'],['X'])]:
        terms=sorted(set(a+b));left=np.array([a.count(t) for t in terms],dtype=float);right=np.array([b.count(t) for t in terms],dtype=float)
        d=np.linalg.norm(left)*np.linalg.norm(right);expected=float(left@right/d) if d else 0.
        assert math.isclose(cosine(a,b),expected,rel_tol=1e-14,abs_tol=1e-14)
        examples.append(dict(a=a,b=b,cosine=cosine(a,b)))
    for count in [0,1,19,20,21,239,240,260,1000]:
        words=[dict(id=i,text='same',start=float(i),end=float(i+1)) for i in range(count)]
        p=partition(words);assert p['boundaries']==[]
        assert [i for s in p['segments'] for i in range(s['start_word'],s['end_word'])]==list(range(count))
        assert all(len(x)==20 for x in p['pseudo_sentence_word_ids'][:-1])
    words=[dict(id=i,text='astronomy' if i<400 else 'cooking',start=i/20,end=(i+1)/20) for i in range(800)]
    p=partition(words);assert p['boundaries']==[400],p['boundaries']
    expected=[i for i in p['gaps'] if i==20];assert p['accepted_pseudo_gaps']==expected
    ownership=scope(words,p,0,8);assert ownership['local_ids']==list(range(160))
    assert ownership['context_ids']==list(range(160,400)) and ownership['segment_ids']==[0]
    exact=[dict(id=0,text='at',start=7.5,end=8.5),dict(id=1,text='end',start=15.5,end=16.5)]
    pe=partition(exact);assert scope(exact,pe,0,8)['local_ids']==[]
    assert scope(exact,pe,8,16)['local_ids']==[0]
    assert scope(exact,pe,16,24)['local_ids']==[1]
    unicode=[dict(id=0,text="DON’T naïve 你好_42!",start=0.,end=1.)]
    pu=partition(unicode);assert [x['term'] for x in pu['lexical']]==['don’t','naïve','你好','42']
    # A source word crosses multiple pseudo sentences: never split the original word.
    multi=[dict(id=0,text=' '.join(['one']*420),start=0.,end=1.)]+[dict(id=i+1,text='two',start=i+1.,end=i+2.) for i in range(400)]
    pm=partition(multi)
    assert all(0<b<len(multi) for b in pm['boundaries'])
    assert [i for s in pm['segments'] for i in range(s['start_word'],s['end_word'])]==list(range(len(multi)))
    summary=dict(PASS=True,scope='lexical operator CPU only; no real ASR/Qwen/GT/performance',spec=SPEC,
        dense_reference=examples,topic_boundary=p['boundaries'],exact8s_midpoint=True,
        short_flat_empty_unicode_original_word_coverage=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print('PASS lexical numeric/coverage/source time checks')


if __name__=='__main__':main()
