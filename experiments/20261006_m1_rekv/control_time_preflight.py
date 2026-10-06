"""Real-tokenizer full source suffix exposure before timestamp intervention."""
import argparse
import json
from pathlib import Path
from control_time import permutation
from inputs import ROOT,selected_rows,SPEC
from src.mllm_renderer import cpu_renderer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    j=cpu_renderer();rows=[];maps={};root=ROOT/'runs/20261006_m1_rekv/r1_full_main'
    for number,row in enumerate(selected_rows(),1):
        bundle=json.loads((root/'records'/row['dataset']/(row['video_id']+'.json')).read_text());blocks=bundle['source_blocks']
        actual=[b['actual_time'] for b in blocks];indices=[b['source_index'] for b in blocks]
        # Full unexpanded suffix tokenization includes real chat syntax around
        # the timestamp. Pixels/processor expansion are fixed, never invented.
        lengths=[len(j.tok(b['input_evidence']['suffix_text'],add_special_tokens=False)['input_ids']) for b in blocks]
        donor,stats=permutation(actual,lengths,indices)
        changed=0
        for index,target in enumerate(donor):
            evidence=blocks[index]['input_evidence'];text=evidence['suffix_text'];old=SPEC['source_text'].format(time=actual[index]);new=SPEC['source_text'].format(time=actual[target])
            assert text.count(old)==1
            modified=text.replace(old,new,1)
            before=j.tok(text,add_special_tokens=False)['input_ids'];after=j.tok(modified,add_special_tokens=False)['input_ids']
            assert len(before)==len(after)==lengths[index]
            assert [i for i,t in enumerate(before) if t==j.image_token_id]==[i for i,t in enumerate(after) if t==j.image_token_id]
            changed+=int(actual[index]//8!=actual[target]//8)
        exposures=0;total=0
        for trace in bundle['traces']:
            if not trace['branch']:continue
            start,end=trace['bounds']
            for layer in trace['branch']['layers']:
                for index in layer['selected_ids']:
                    total+=1;exposures+=int((start<=actual[index]<end)!=(start<=actual[donor[index]]<end))
        key=row['dataset']+'/'+row['video_id'];maps[key]=dict(timestamp_donors=donor,actual_times=actual,presented_times=[actual[d] for d in donor],
            suffix_token_lengths=lengths,groups=stats)
        rows.append(dict(dataset=row['dataset'],video_id=row['video_id'],source_frames=len(blocks),cross_window_frames=changed,
            selected_block_occurrence_role_changes=exposures,total_selected_block_reads=total))
        if number%25==0:print('TIME_PREFLIGHT',number,flush=True)
    result=dict(PASS=True,GT_read=False,scope='actual complete source suffix/real-tokenizer timestamp exposure only; pixels/grid kept bysamepaths, actual expanded processor/GPU test stillrequired',
        coverage=len(rows),per_video=rows,maps=maps)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('TIME_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':main()
