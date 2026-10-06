"""Real tokenizer compound syntax, physical-token replay and unchanged caps."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from timeline import ROOT,SPEC,INTERFACE
from compound_closure import CompleteQuotedStream,prefix_consumed,stream_scope,OriginalStream
import src.structured_source_generation as base
from src.mllm_renderer import cpu_renderer


def main():
    assert INTERFACE=='C';ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    j=cpu_renderer();following=',"action":"';inventory=CompleteQuotedStream(j,128,tokens=[]);inventory.following_literal=following
    choices=inventory.closing_tokens();compound=next(i for i,s in choices.items() if s==',')
    prefix=j.tok.encode('{"reason":"',add_special_tokens=False);body=j.tok.encode('Actual source observed',add_special_tokens=False)
    remaining=j.tok.encode('"action":"',add_special_tokens=False);tail=j.tok.encode('PROGRESS_BAR"}',add_special_tokens=False)
    all_tokens=prefix+body+[compound]+remaining+tail
    stream=CompleteQuotedStream(j,128,tokens=all_tokens,description_words=16,field_tokens=32)
    stream.force('{"reason":"');stream.following_literal=following;stream.description();event=stream.events[-1]
    assert event['text']=='Actual source observed' and event['reason']=='model_quote' and stream.pending==','
    stream.force(following);assert not stream.pending
    stream.force('PROGRESS_BAR"}');assert stream.tokens==all_tokens
    parsed=json.loads(j.tok.decode(stream.tokens));assert parsed==dict(reason='Actual source observed',action='PROGRESS_BAR')
    assert len(stream.tokens)==len(all_tokens) and stream.events[-2]['compound_literal_chars']==1
    # JSON whitespace outside a key is legal; whitespace inside the key is not.
    assert prefix_consumed(',\n',following)==1 and prefix_consumed(',"',following)==2
    assert prefix_consumed('," ',following) is None and prefix_consumed('}',following) is None
    for literal in (']','}',',"'):
        candidate=CompleteQuotedStream(j,128,tokens=[]);candidate.following_literal=literal
        closures=candidate.closing_tokens()
        assert j.tok.encode('"',add_special_tokens=False)[0] in closures
        assert all(prefix_consumed(s,literal) is not None for s in closures.values())
    quote=j.tok.encode('"',add_special_tokens=False)[0];bounded=CompleteQuotedStream(j,128,tokens=body+[quote],description_words=3,field_tokens=32)
    bounded.following_literal=following;bounded.description()
    assert bounded.events[-1]['reason']=='word_cap' and bounded.events[-1]['text']=='Actual source observed'
    saved=base.Stream
    try:
        with stream_scope():assert base.Stream is CompleteQuotedStream;raise RuntimeError('fixture')
    except RuntimeError:pass
    assert base.Stream is saved is OriginalStream
    summary=dict(PASS=True,scope='actual Qwen tokenizer/scoped physical-token replay only; no model/GPU/GT/performance',compound_token=compound,
        logical_json_exact=True,actual_token_not_retokenized=True,following_literal_consumed_once=True,whitespace_role_exact=True,
        caps_unchanged_UNKNOWN_still_required=True,scope_restored_after_exception=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPOUND_CPU_PASS')


if __name__=='__main__':main()
