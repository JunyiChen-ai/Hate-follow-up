"""Bind raw tokenizer spans to actual processor-expanded image token rows."""


def expand(raw_ids,raw_offsets,image_token_id,image_counts,actual_ids):
    assert len(raw_ids)==len(raw_offsets);ids=[];offsets=[];cursor=0;image=0
    while cursor<len(raw_ids):
        token=raw_ids[cursor]
        if token!=image_token_id:
            ids.append(token);offsets.append(raw_offsets[cursor]);cursor+=1;continue
        end=cursor+1
        while end<len(raw_ids) and raw_ids[end]==image_token_id:end+=1
        assert image<len(image_counts);count=image_counts[image];run=end-cursor
        assert run in (1,count),'unexpected raw image-placeholder multiplicity'
        ids.extend([token]*count)
        offsets.extend(raw_offsets[cursor:end] if run==count else [raw_offsets[cursor]]*count)
        cursor=end;image+=1
    assert image==len(image_counts) and ids==actual_ids,'processor expanded IDs differ from bound raw/token/image sequence'
    return offsets
