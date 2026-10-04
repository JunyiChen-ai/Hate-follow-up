"""Independent path enumeration, word identity, real tiny-model attention checks."""
import itertools
import math
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from alignment import path_distribution,words_from_segments,build_blocks,CrossAttentionCapture,WhisperAligner
from reader import token_support,arm_support,PriorReader


def enumerate_paths(c):
    paths=[]
    def walk(cells):
        i,j=cells[-1]
        if (i,j)==(c.shape[0]-1,c.shape[1]-1):paths.append(cells);return
        for di,dj in ((1,0),(0,1),(1,1)):
            if i+di<c.shape[0] and j+dj<c.shape[1]:walk(cells+[(i+di,j+dj)])
    walk([(0,0)])
    logs=np.array([sum(c[i,j] for i,j in p)-(len(p)-1)*math.log(3) for p in paths])
    weights=np.exp(logs-logs.max());weights/=weights.sum()
    visits=np.zeros_like(c,dtype=float)
    for w,p in zip(weights,paths):
        for i,j in p:visits[i,j]+=w
    return visits/visits.sum(1,keepdims=True),logs.max()+math.log(np.exp(logs-logs.max()).sum()),paths,logs


def test_paths():
    for K,T in itertools.product(range(1,4),repeat=2):
        c=np.random.default_rng(K*10+T).random((K,T))
        p,h,z=path_distribution(c);expected,ez,paths,logs=enumerate_paths(c)
        np.testing.assert_allclose(p,expected,atol=1e-12);assert abs(z-ez)<1e-12
        winner=paths[int(logs.argmax())];hit=np.zeros_like(c)
        for i,j in winner:hit[i,j]=1
        hit/=hit.sum(1,keepdims=True);np.testing.assert_allclose(h,hit)
    p,h,z=path_distribution(np.zeros((40,300)))
    assert np.isfinite(z) and np.isfinite(p).all() and np.allclose(p.sum(1),1)


def test_word_identity():
    class Tokenizer:
        def encode(self,char,**kwargs):return [ord(char)%90]* (4 if char!=' ' else 1)
        def convert_tokens_to_ids(self,x):return 100
    segs=[(0,80,'a'*430),(81,122,'go go no!')];words=words_from_segments(segs)
    blocks=build_blocks(segs,words,Tokenizer())
    actual=[(x['word'],x['char']) for b in blocks for x in b['items'] if x['word']>=0]
    expected=[(0,'a')]*430+[(1,'g'),(1,'o'),(2,'g'),(2,'o'),(3,'n'),(3,'o')]
    assert actual==expected
    assert all(b['end']-b['start']<=22 and sum(len(x['tokens']) for x in b['items'])<=400 for b in blocks)
    lone=[(0,70,'a')]; lw=words_from_segments(lone); repeated=build_blocks(lone,lw,Tokenizer())
    assert len(repeated)==4 and abs(sum(x.get('weight',1.) for b in repeated for x in b['items'])-1)<1e-12
    empty=[(0,100,'!!!')]; assert build_blocks(empty,words_from_segments(empty),Tokenizer())==[]
    offsets=np.array([[0,2],[2,5],[5,8],[8,10]])
    support,bound=token_support(offsets,[(0,2,4),(1,4,8)],np.array([.2,.8]))
    np.testing.assert_allclose(support,[1.,.4,.8,1.]);assert bound.tolist()==[False,True,True,False]
    a=dict(words=[1,2],windows=[0,1,2],soft=[[.2,.7,.1],[.6,.3,.1]],viterbi=[[.5,.3,.2],[.1,.1,.8]],proportional=[[1,0,0],[0,1,0]])
    np.testing.assert_array_equal(arm_support(a,'hard'),[[1,0,0],[0,0,1]])
    np.testing.assert_allclose(arm_support(a,'shifted'),np.roll(a['soft'],1,axis=1))


def test_real_whisper_capture():
    from transformers import WhisperConfig,WhisperModel
    torch.manual_seed(0)
    c=WhisperConfig(vocab_size=70,d_model=24,encoder_layers=1,decoder_layers=2,
        encoder_attention_heads=6,decoder_attention_heads=6,encoder_ffn_dim=32,decoder_ffn_dim=32,
        max_source_positions=5,max_target_positions=12,pad_token_id=0,bos_token_id=1,eos_token_id=2)
    c._attn_implementation='sdpa';m=WhisperModel(c).eval();x=torch.randn(1,5,24);ids=torch.tensor([[1,3,4,5,2]])
    reference=m.decoder(input_ids=ids,encoder_hidden_states=x,use_cache=False).last_hidden_state
    cap=CrossAttentionCapture(m.decoder,1,3,5)
    with cap.capture():got=m.decoder(input_ids=ids,encoder_hidden_states=x,use_cache=False).last_hidden_state
    assert torch.equal(reference,got);matrix,heads=cap.matrix()
    assert matrix.shape==(3,5) and len(heads)==10
    np.testing.assert_allclose(np.linalg.norm(matrix,axis=0),1,atol=1e-7)


def test_empty_audio():
    class Tokenizer:
        def encode(self,char,**kwargs):return [ord(char)%90]
        def convert_tokens_to_ids(self,x):return 100
    aligner=WhisperAligner.__new__(WhisperAligner);aligner.tok=Tokenizer()
    # No decoder/encoder exist: falling through to any model call fails immediately.
    segments=[(24.,26.,'Torsdagsforskning !!!')]
    result=aligner.align(np.zeros(int(19.27*16000),dtype=np.float32),segments,[(0,8),(8,16),(16,19.272562)])
    assert len(result['empty_audio_blocks'])==1 and result['encoder_calls']==result['decoder_calls']==0
    assert result['language'] is None
    np.testing.assert_array_equal(result['soft'],result['proportional'])
    np.testing.assert_array_equal(result['viterbi'],result['proportional'])
    assert all(x['donor'] is None for x in result['inheritance'])


def test_real_qwen_prior():
    from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
    from src.mllm_judge import Judge
    import copy
    torch.manual_seed(0)
    c=Qwen3VLConfig(text_config=dict(vocab_size=60,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=4,
        rope_scaling=dict(rope_type='default',mrope_section=[1,1,0]),max_position_embeddings=64),
        vision_config=dict(depth=1,hidden_size=16,intermediate_size=32,num_heads=4,
            patch_size=2,temporal_patch_size=1,spatial_merge_size=1,out_hidden_size=16,
            deepstack_visual_indexes=[]),image_token_id=55,video_token_id=56,vision_start_token_id=57)
    c._attn_implementation='sdpa';model=Qwen3VLForConditionalGeneration(c).eval()
    for param in model.parameters():param.requires_grad_(False)
    j=Judge.__new__(Judge);j.model=model;j.device=torch.device('cpu');j.dtype=torch.float32;j.yes_ids=[3];j.no_ids=[4];j.softcap=None
    cache=model.model(input_ids=torch.tensor([[6,7,8,9]]),use_cache=True).past_key_values
    delta=model.model.rope_deltas.clone();reader=PriorReader(j);ids=[10,11,12];n=cache.get_seq_length()
    ordinary=j.cached_margin(cache,ids,in_place=True);cache.crop(n);model.model.rope_deltas=delta.clone()
    ones=np.ones(n+len(ids));neutral=reader.margin(cache,ids,ones,delta);assert neutral==ordinary
    changed=ones.copy();changed[1:3]=1e-6
    value=reader.margin(cache,ids,changed,delta);assert value!=neutral
    clone=copy.deepcopy(cache);assert reader.margin(clone,ids,changed,delta)==value
    model.model.rope_deltas=delta.clone()
    complete=copy.deepcopy(cache)
    mask=torch.tensor(np.log(np.maximum(changed,1e-6)),dtype=torch.float32)[None,None,None,:].expand(1,1,len(ids),-1).clone()
    for k in range(len(ids)):mask[:,:,k,n+k+1:]=float('-inf')
    out=model.model(input_ids=torch.tensor([ids]),past_key_values=complete,use_cache=True,attention_mask=mask)
    expected=j.margins_fp32(out.last_hidden_state[:,-1])[0]
    assert abs(expected-value)<1e-6 and cache.get_seq_length()==n
    assert all(p.grad is None for p in model.parameters());reader.close()


if __name__=='__main__':
    for check in (test_paths,test_word_identity,test_empty_audio,test_real_whisper_capture,test_real_qwen_prior):
        check();print(check.__name__,'PASS',flush=True)
