"""Actual32-layer random-weight Whisper teacher-forced interface, CPU only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from alignment import align_weights,timed_words
from transformers import AutoProcessor,WhisperConfig,WhisperModel,GenerationConfig


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0);torch.set_num_threads(4)
    p=AutoProcessor.from_pretrained('openai/whisper-large-v3',local_files_only=True)
    g=GenerationConfig.from_pretrained('openai/whisper-large-v3',local_files_only=True)
    config=WhisperConfig(vocab_size=51866,num_mel_bins=128,d_model=80,encoder_layers=32,decoder_layers=32,
        encoder_attention_heads=20,decoder_attention_heads=20,encoder_ffn_dim=160,decoder_ffn_dim=160,
        max_source_positions=50,max_target_positions=448,pad_token_id=p.tokenizer.eos_token_id)
    config._attn_implementation='eager';model=WhisperModel(config).eval()
    text='Hello, this is a literal source.';content=p.tokenizer.encode(text,add_special_tokens=False)
    prefix=[g.decoder_start_token_id,g.lang_to_id['<|en|>'],g.task_to_id['transcribe'],g.no_timestamps_token_id]
    with torch.no_grad():output=model(input_features=torch.randn(1,128,100),decoder_input_ids=torch.tensor([prefix+content+[p.tokenizer.eos_token_id]]),use_cache=False,output_attentions=True,return_dict=True)
    assert len(output.cross_attentions)==32 and all(a.shape[1]==20 for a in output.cross_attentions)
    matrix,jumps=align_weights(output.cross_attentions,g.alignment_heads,len(prefix),len(content),50)
    words=timed_words(p.tokenizer,content,'en',jumps,0,1,0)
    assert ''.join(w['text'] for w in words)==text and all(0<=w['start']<=w['end']<=1 for w in words)
    np.save(out/'matrix.npy',matrix,allow_pickle=False)
    summary=dict(PASS=True,actual_encoder_layers=32,actual_decoder_layers=32,actual_heads=20,official_alignment_heads=g.alignment_heads,
        teacher_rows=matrix.shape[0],content_tokens=len(content),literal_words=words,
        scope='actual HF architecture with random weights/narrow width and1s features, CPU; not pretrained Whisper generation/GPU/ASR truth or performance')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('PASS actual32layer Whisper teacher-forced cross attention')


if __name__=='__main__':main()
