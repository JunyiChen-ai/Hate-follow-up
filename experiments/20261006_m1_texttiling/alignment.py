"""Whisper official-head DTW and literal original-word timing, without heuristics."""
import numpy as np
import torch
from partition import SPEC


def dtw(matrix):
    """Vectorized antidiagonals; official CPU strict diagonal/up/else-left ties."""
    x=np.asarray(matrix,dtype=np.float64);assert x.ndim==2 and min(x.shape)>0 and np.isfinite(x).all()
    n,m=x.shape;cost=np.full((n+1,m+1),np.inf,np.float32);cost[0,0]=0.
    trace=np.full((n+1,m+1),-1,np.int8)
    for diagonal in range(n+m-1):
        row=np.arange(max(0,diagonal-m+1),min(n-1,diagonal)+1)+1;col=diagonal-(row-1)+1
        a,b,c=cost[row-1,col-1],cost[row-1,col],cost[row,col-1]
        direction=np.where((a<b)&(a<c),0,np.where((b<a)&(b<c),1,2))
        value=np.where(direction==0,a,np.where(direction==1,b,c))
        cost[row,col]=x[row-1,col-1]+value;trace[row,col]=direction
    trace[0,:]=2;trace[:,0]=1;row,col=n,m;points=[]
    while row>0 or col>0:
        points.append((row-1,col-1));direction=trace[row,col]
        if direction==0:row-=1;col-=1
        elif direction==1:row-=1
        elif direction==2:col-=1
        else:raise AssertionError('invalid DTW path')
    text,time=np.array(points[::-1],dtype=int).T
    assert min(text)>=0 and min(time)>=0 and max(text)==n-1 and max(time)==m-1
    return text,time


def word_groups(tokenizer,tokens,language):
    from transformers.models.whisper.tokenization_whisper import _split_tokens_on_spaces,_split_tokens_on_unicode
    function=_split_tokens_on_unicode if language in ('zh','ja','th','lo','my','yue') else _split_tokens_on_spaces
    # Keep punctuation groups as actual token groups; do not shift their times.
    words,groups,indices=function(tokenizer,tokens)
    assert [x for group in indices for x in group]==list(range(len(tokens)))
    assert [x for group in groups for x in group]==tokens
    assert ''.join(words)==tokenizer.decode(tokens,decode_with_timestamps=True)
    return words,groups,indices


def align_weights(attentions,heads,prefix_count,content_count,nframes,median_width=None,teacher_eos=True):
    from transformers.models.whisper.generation_whisper import _median_filter
    # HF eager returns softmax attention weights: do not softmax them again.
    weights=torch.stack([attentions[layer][0,head,:,:nframes].float() for layer,head in heads])
    assert weights.shape[1]==prefix_count+content_count+int(teacher_eos) and weights.shape[-1]>0
    std,mean=torch.std_mean(weights,dim=-2,keepdim=True,unbiased=False)
    normalized=torch.where(std>0,(weights-mean)/std.clamp_min(torch.finfo(weights.dtype).tiny),torch.zeros_like(weights))
    filtered=_median_filter(normalized.unsqueeze(0),SPEC['alignment_median_width'] if median_width is None else median_width)[0]
    matrix=filtered.mean(0)[prefix_count-1:(-1 if teacher_eos else None)].cpu().numpy()
    assert matrix.shape==(content_count+1,nframes) and np.isfinite(matrix).all()
    text,time=dtw(-matrix)
    jumps=np.r_[True,np.diff(text)!=0];jump_seconds=time[jumps]*.02
    assert len(jump_seconds)==content_count+1 and np.all(np.diff(jump_seconds)>=0)
    return matrix,jump_seconds


def timed_words(tokenizer,content,language,jumps,start,end,block):
    words,groups,indices=word_groups(tokenizer,content,language)
    result=[]
    for text,group,index in zip(words,groups,indices):
        raw=[start+float(jumps[index[0]]),start+float(jumps[index[-1]+1])]
        used=[max(start,min(end,x)) for x in raw]
        assert used[0]<=used[1]
        result.append(dict(text=text,tokens=group,token_indices=index,raw_dtw=raw,start=used[0],end=used[1],block=block))
    return result


class Recognizer:
    def __init__(self):
        from transformers import AutoProcessor,WhisperForConditionalGeneration
        self.processor=AutoProcessor.from_pretrained(SPEC['asr_model'],local_files_only=True)
        self.model=WhisperForConditionalGeneration.from_pretrained(SPEC['asr_model'],dtype=torch.float16,
            attn_implementation='eager',local_files_only=True).to('cuda').eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.counts=dict(encoder=0,decoder=0)
        self.hooks=[getattr(self.model.model,name).register_forward_pre_hook(
            lambda *_,name=name:self.counts.__setitem__(name,self.counts[name]+1)) for name in self.counts]
        self.heads=self.model.generation_config.alignment_heads
        assert self.heads and self.model.config.max_target_positions>=SPEC['asr_max_length']

    def features(self,audio):
        assert 0<len(audio)<=SPEC['source_audio_seconds']*SPEC['sample_rate'] and np.isfinite(audio).all()
        return self.processor.feature_extractor(audio,sampling_rate=SPEC['sample_rate'],return_tensors='pt',
            padding='max_length',truncation=True).input_features.to('cuda',torch.float16)

    @torch.no_grad()
    def language(self,audio):
        identifier=int(self.model.detect_language(input_features=self.features(audio))[0])
        key=next(k for k,v in self.model.generation_config.lang_to_id.items() if v==identifier)
        return identifier,key.removeprefix('<|').removesuffix('|>')

    @torch.no_grad()
    def block(self,audio,language_id,language,start,end,number):
        import copy
        from transformers.generation.utils import GenerationMixin
        from transformers.generation.logits_process import LogitsProcessorList,SuppressTokensLogitsProcessor,SuppressTokensAtBeginLogitsProcessor
        generation=copy.deepcopy(self.model.generation_config);generation._from_model_config=False
        prefix=[generation.decoder_start_token_id,language_id,generation.task_to_id['transcribe'],generation.no_timestamps_token_id]
        suppress=sorted(set(generation.suppress_tokens or [])|set(range(generation.no_timestamps_token_id+1,self.model.config.vocab_size)))
        processors=LogitsProcessorList([SuppressTokensLogitsProcessor(suppress,device='cuda')])
        if generation.begin_suppress_tokens:processors.append(SuppressTokensAtBeginLogitsProcessor(generation.begin_suppress_tokens,begin_index=len(prefix),device='cuda'))
        generation.suppress_tokens=None;generation.begin_suppress_tokens=None;generation.forced_decoder_ids=None
        generation.num_beams=1;generation.num_return_sequences=1;generation.do_sample=False
        generation.max_length=SPEC['asr_max_length'];generation.max_new_tokens=None
        generation.return_timestamps=False;generation.return_dict_in_generate=True;generation.output_scores=False
        features=self.features(audio)
        output=GenerationMixin.generate(self.model,input_features=features,generation_config=generation,
            decoder_input_ids=torch.tensor([prefix],device='cuda'),logits_processor=processors)
        ids=output.sequences[0].tolist();assert ids[:len(prefix)]==prefix
        eos=generation.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        generated=ids[len(prefix):];stop=next((i for i,t in enumerate(generated) if t in eos),len(generated));content=generated[:stop]
        assert all(t<self.processor.tokenizer.eos_token_id for t in content)
        matrix=np.empty((0,0),np.float32);jumps=np.empty(0,np.float64);words=[]
        if content:
            teacher_eos=len(prefix)+len(content)<self.model.config.max_target_positions
            teacher=prefix+content+([self.processor.tokenizer.eos_token_id] if teacher_eos else [])
            aligned=self.model.model(input_features=features,decoder_input_ids=torch.tensor([teacher],device='cuda'),
                output_attentions=True,use_cache=False,return_dict=True)
            nframes=min(aligned.cross_attentions[0].shape[-1],max(1,int(np.ceil(len(audio)/320))))
            matrix,jumps=align_weights(aligned.cross_attentions,self.heads,len(prefix),len(content),nframes,teacher_eos=teacher_eos)
            words=timed_words(self.processor.tokenizer,content,language,jumps,start,end,number)
            del aligned
        result=dict(block=number,prefix_tokens=prefix,tokens=ids,content_tokens=content,words=words,
            truncated=stop==len(generated),text=self.processor.tokenizer.decode(content,skip_special_tokens=True),
            heads=self.heads,raw_jump_seconds=jumps.tolist(),alignment_matrix_shape=list(matrix.shape),
            teacher_eos=bool(content and len(prefix)+len(content)<self.model.config.max_target_positions),
            generation_api='generic HF GenerationMixin greedy, no longform ASR heuristics')
        del output,features
        return result,matrix
