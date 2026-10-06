"""Actual333 raw/native/tokenizer inputs; explicitly no new ASR or model calls."""
import argparse
import json
from pathlib import Path
import av
from inputs import ROOT,SPEC,selected_rows
from src.audio_inputs import resolve_video
from src.video_inputs import frame_paths,load_asr
from src.mllm_renderer import cpu_renderer
from transformers import AutoProcessor,GenerationConfig


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    j=cpu_renderer();whisper=AutoProcessor.from_pretrained(SPEC['asr_model'],local_files_only=True)
    config=GenerationConfig.from_pretrained(SPEC['asr_model'],local_files_only=True);assert len(config.alignment_heads)==10
    for field in ('UNKNOWN;','NONE;','W00000000;','W00002047;'):assert j.tok.encode(field,add_special_tokens=False)
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};records=[]
    for row in selected_rows():
        path=resolve_video(row)
        with av.open(str(path)) as container:assert container.streams.video;audio=bool(container.streams.audio)
        frames=frame_paths(row['dataset'],row['video_id'],20);assert 18<=len(frames)<=20
        segments=asr[row['dataset']].get(row['video_id'],[]);msgs,files=j.prefix_messages(frames,segments);text,enc=j.encode_prefix(msgs,files)
        records.append(dict(dataset=row['dataset'],video_id=row['video_id'],native_frames=len(frames),native_prefix_tokens=enc['input_ids'].shape[1],actual_audio_stream=audio,
            source_blocks=len(__import__('src.video_inputs',fromlist=['fixed_windows']).fixed_windows(float(row['duration']),30))))
    result=dict(PASS=True,GT_read=False,coverage=len(records),scope='actual333 raw/native/real processors/grammar only; no new ASR/audio decode/8B/capacity/performance claim',
        official_alignment_heads=config.alignment_heads,records=records)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS actual333 input/real processors; not new source validity')


if __name__=='__main__':main()
