#!/usr/bin/env python3
"""Same-Whisper real-window beams and independent-slot DAGs, without labels."""
import argparse
import copy
import json
import logging
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.video_inputs import load_manifest,load_asr,fixed_windows,window_text
from lattice import ASR_MODEL,CACHE_VERSION,CONSTANTS,confusion
from audio import resolve_video,decode_audio,crop_audio,RATE
DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/temporal_speech_lattice'


def selected_rows(smoke):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:
        rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def tick():torch.cuda.synchronize();return time.perf_counter()


class Recognizer:
    def __init__(self):
        from transformers import AutoProcessor,WhisperForConditionalGeneration
        self.processor=AutoProcessor.from_pretrained(ASR_MODEL)
        self.model=WhisperForConditionalGeneration.from_pretrained(
            ASR_MODEL,dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.counts=dict(encoder=0,decoder=0)
        self.hooks=[]
        for name in self.counts:
            self.hooks.append(getattr(self.model.model,name).register_forward_pre_hook(
                lambda *_,name=name:self.counts.__setitem__(name,self.counts[name]+1)))

    def features(self,audio):
        assert 0<len(audio)<=30*RATE and np.isfinite(audio).all()
        result=self.processor.feature_extractor(audio,sampling_rate=RATE,
            return_tensors='pt',padding='max_length',truncation=True)
        assert result['input_features'].shape[-1]==3000
        return result['input_features'].to('cuda',torch.float16)

    @torch.no_grad()
    def language(self,audio):
        language=int(self.model.detect_language(input_features=self.features(audio))[0])
        assert language in self.model.generation_config.lang_to_id.values()
        return language

    @torch.no_grad()
    def beams(self,audio,language):
        from transformers.generation.utils import GenerationMixin
        from transformers.generation.logits_process import LogitsProcessorList,SuppressTokensLogitsProcessor,SuppressTokensAtBeginLogitsProcessor
        g=copy.deepcopy(self.model.generation_config);g._from_model_config=False
        prefix=[g.decoder_start_token_id,language,g.task_to_id['transcribe'],g.no_timestamps_token_id]
        suppress=sorted(set(g.suppress_tokens or [])|set(range(g.no_timestamps_token_id+1,self.model.config.vocab_size)))
        begin=list(g.begin_suppress_tokens or [])
        processors=LogitsProcessorList([SuppressTokensLogitsProcessor(suppress,device='cuda')])
        if begin:processors.append(SuppressTokensAtBeginLogitsProcessor(begin,begin_index=len(prefix),device='cuda'))
        g.suppress_tokens=None;g.begin_suppress_tokens=None;g.forced_decoder_ids=None
        g.num_beams=5;g.num_return_sequences=5;g.do_sample=False;g.length_penalty=1.
        g.early_stopping=False;g.max_length=448;g.max_new_tokens=None
        g.return_timestamps=False;g.return_dict_in_generate=True;g.output_scores=True
        features=self.features(audio)
        # The Whisper long-form wrapper repeats deterministic input/onebest for nreturn>1.
        # Generic short-form encoder/decoder generation returns the actual top-five beams.
        out=GenerationMixin.generate(self.model,input_features=features,generation_config=g,
            decoder_input_ids=torch.tensor([prefix],device='cuda'),logits_processor=processors)
        assert out.sequences.shape[0]==5 and out.sequences.shape[1]<=448
        scores=out.sequences_scores.float().cpu().tolist();records=[]
        eos=set(g.eos_token_id if isinstance(g.eos_token_id,list) else [g.eos_token_id])
        for i,(ids,score) in enumerate(zip(out.sequences.cpu().tolist(),scores)):
            assert ids[:len(prefix)]==prefix and np.isfinite(score)
            generated=ids[len(prefix):];stop=next((j for j,v in enumerate(generated) if v in eos),len(generated))
            content=generated[:stop]
            assert all(v<=g.no_timestamps_token_id for v in content),'timestamp token in no-timestamps beam'
            records.append(dict(beam=i,prefix_tokens=prefix,tokens=ids[:len(prefix)+stop+(stop<len(generated))],
                generated_tokens=content,text=self.processor.tokenizer.decode(content,skip_special_tokens=True).strip(),
                score=score,truncated=stop==len(generated)))
        trace=dict(api='GenerationMixin.generate on same Whisper short-form model',prefix_tokens=prefix,
            num_beams=5,num_return_sequences=5,max_length=448,length_penalty=1.,do_sample=False,
            return_timestamps=False,suppressed_tokens=suppress,begin_suppressed_tokens=begin,
            decoder_steps=len(out.scores),physical_sequences=5,
            score_semantics='HF length-normalized beam sequence score; FP32 softmax weights, not calibrated posterior')
        del out,features
        return records,trace


def validate(r,row,segments):
    assert r['cache_version']==CACHE_VERSION and r['constants']==CONSTANTS and r['model']==ASR_MODEL
    assert r['GT_read'] is False and r['dataset']==row['dataset'] and r['video_id']==row['video_id']
    assert r['duration']==float(row['duration']) and r['manifest_video_path']==row['video_path']
    assert Path(r['input_video']).stem==row['video_id'] and r['segments']==[list(s) for s in segments]
    windows=fixed_windows(r['duration'],8);assert len(r['windows'])==len(windows)
    for i,(w,(a,b)) in enumerate(zip(r['windows'],windows)):
        body=window_text(segments,a,b)
        assert w['i']==i and w['start']==a and w['end']==b and w['native_body']==body
        if w['available']:
            assert body.strip() and len(w['beams'])==5 and w['crop']['samples']
            assert w['confusion']==confusion(w['beams'])
            assert w['generation']['num_beams']==5 and w['generation']['physical_sequences']==5
            lo,hi=w['crop']['samples'];assert int(np.floor(a*RATE))<=lo<hi<=int(np.ceil(b*RATE))
            assert w['crop']['observed_intervals'] and w['crop']['actual_interval']==[lo/RATE,hi/RATE]
            assert all(lo<=x<y<=hi for x,y in w['crop']['observed_intervals']+w['crop']['gaps'])
        else:assert w['reason'] in ('native_no_speech','missing_actual_audio')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261004_m1_lattice'/('r1_extract_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    config=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=ASR_MODEL,cache_version=CACHE_VERSION,
        constants=CONSTANTS,GT_read=False,smoke=a.smoke,torch=torch.__version__,transformers=transformers.__version__,
        command='python -u '+' '.join(sys.argv),code='experiments/20261004_m1_lattice/{audio,lattice,extract}.py; sources2026-10-05')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');torch.manual_seed(0)
    rows=selected_rows(a.smoke);asr={ds:load_asr(ds) for ds in DATASETS};model=Recognizer()
    for i,row in enumerate(rows,1):
        ds,vid=row['dataset'],row['video_id'];segments=asr[ds].get(vid,[])
        path=CACHE/ds/(vid+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():validate(json.loads(path.read_text()),row,segments);logging.info('%d/%d reuse %s/%s',i,len(rows),ds,vid);continue
        source=resolve_video(row);first=dict(model.counts);torch.cuda.reset_peak_memory_stats();t0=tick()
        samples,observed,timeline=decode_audio(source,float(row['duration']));audio_seconds=tick()-t0
        language=None;language_crop=None;records=[]
        if any(window_text(segments,x,y).strip() for x,y in fixed_windows(float(row['duration']),8)) and observed.any():
            first_sample=int(np.flatnonzero(observed)[0]);start=first_sample/RATE
            actual,language_crop=crop_audio(samples,observed,start,min(float(row['duration']),start+30))
            language=model.language(actual)
        for number,(start,end) in enumerate(fixed_windows(float(row['duration']),8)):
            body=window_text(segments,start,end);record=dict(i=number,start=start,end=end,native_body=body,available=False)
            if not body.strip():record['reason']='native_no_speech'
            else:
                actual,crop=crop_audio(samples,observed,start,end);record['crop']=crop
                if not len(actual):record['reason']='missing_actual_audio'
                else:
                    assert language is not None
                    beams,generation=model.beams(actual,language)
                    record.update(available=True,beams=beams,generation=generation,confusion=confusion(beams))
            records.append(record)
        r=dict(config,dataset=ds,video_id=vid,duration=float(row['duration']),input_video=str(source),
            manifest_video_path=row['video_path'],input_asr=f'data/asr_whisper_large_v3/{ds}/timestamped_chunks.jsonl',
            segments=[list(s) for s in segments],timeline=timeline,language_token=language,language_crop=language_crop,
            windows=records,audio_seconds=audio_seconds,standalone_seconds=tick()-t0,
            actual_forwards={k:model.counts[k]-first[k] for k in first},peak_GiB=torch.cuda.max_memory_allocated()/2**30)
        validate(r,row,segments);temp=path.with_suffix('.partial');temp.write_text(json.dumps(r)+'\n');temp.replace(path)
        logging.info('%d/%d %s/%s recognized=%d %.2fs',i,len(rows),ds,vid,sum(w['available'] for w in records),r['standalone_seconds'])
    metas=[json.loads((CACHE/r['dataset']/(r['video_id']+'.json')).read_text()) for r in rows]
    (out/'summary.json').write_text(json.dumps(dict(coverage=len(metas),GT_read=False,
        standalone_seconds=sum(r['standalone_seconds'] for r in metas),
        actual_forwards={k:sum(r['actual_forwards'][k] for r in metas) for k in model.counts}),indent=2)+'\n')
    (CACHE/'PROVENANCE.md').write_text('# Temporal speech lattice provenance\n\n'
        'Code: experiments/20261004_m1_lattice/{audio,lattice,extract}.py, sources2026-10-05.\n'
        'Same frozen openai/whisper-large-v3; actual media origins/crops/beams/weights and generation settings in each JSON.\n'
        'No GT; identity/duration from data/omsl_v6_inputs/manifests/all_test.jsonl; native availability from shared ASR loader.\n'
        'Real mono16k sample timeline, no silent time compression; per-video detection, fixed beam5, independent-slot approximation.\n'
        'Command: '+config['command']+'; allocated lab Slurm launch.\n'
        'Hosts: '+', '.join(sorted({r['host'] for r in metas}))+'; generated dates: '+', '.join(sorted({r['date'] for r in metas}))+'.\n')
    logging.info('EXTRACTION_DONE coverage=%d',len(metas))


if __name__=='__main__':main()
