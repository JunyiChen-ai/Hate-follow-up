"""Character alignment and explicit Gibbs DTW path distribution. No labels."""
from contextlib import contextmanager
import math
import re
import string
import numpy as np
import torch
import torch.nn.functional as F

ALIGN_MODEL = 'openai/whisper-large-v3'
CACHE_VERSION = 'R1 character-only Gibbs alignment, sources 2026-10-04'
ALGORITHM = dict(block_seconds=22., context_seconds=4., char_tokens=400, top_heads=10,
                 median_width=3, temperature=1., transition_probability=1/3,
                 sample_rate=16000, frame_seconds=.02, support_floor=1e-6,
                 row_definition='input_character_only', head_norm='average_then_column', seed=0)


def path_distribution(matrix):
    """Visits all paths with diag/up/left; beta excludes the current cell energy."""
    c = np.asarray(matrix, dtype=np.float64)
    assert c.ndim == 2 and min(c.shape) > 0 and np.isfinite(c).all()
    K, T = c.shape
    a = np.full((K, T), -np.inf); b = a.copy(); v = a.copy()
    previous = np.full((K, T), -1, dtype=np.int8)
    penalty = math.log(3.)
    a[0, 0] = v[0, 0] = c[0, 0]
    for d in range(1, K+T-1):
        i = np.arange(max(0, d-T+1), min(K-1, d)+1); j = d-i
        up = np.where(i > 0, a[np.maximum(i-1, 0), j], -np.inf)
        left = np.where(j > 0, a[i, np.maximum(j-1, 0)], -np.inf)
        diag = np.where((i > 0)&(j > 0), a[np.maximum(i-1, 0), np.maximum(j-1, 0)], -np.inf)
        a[i, j] = c[i, j]+np.logaddexp(np.logaddexp(up, left), diag)-penalty
        options = np.stack((np.where(i > 0, v[np.maximum(i-1, 0), j], -np.inf),
            np.where(j > 0, v[i, np.maximum(j-1, 0)], -np.inf),
            np.where((i > 0)&(j > 0), v[np.maximum(i-1, 0), np.maximum(j-1, 0)], -np.inf)))
        previous[i, j] = options.argmax(0)
        v[i, j] = c[i, j]+options.max(0)-penalty
    b[-1, -1] = 0.
    for d in range(K+T-3, -1, -1):
        i = np.arange(max(0, d-T+1), min(K-1, d)+1); j = d-i
        ni = np.minimum(i+1, K-1); nj = np.minimum(j+1, T-1)
        up = np.where(i+1 < K, c[ni, j]+b[ni, j], -np.inf)
        left = np.where(j+1 < T, c[i, nj]+b[i, nj], -np.inf)
        diag = np.where((i+1 < K)&(j+1 < T), c[ni, nj]+b[ni, nj], -np.inf)
        b[i, j] = np.logaddexp(np.logaddexp(up, left), diag)-penalty
    visits = np.exp(a+b-a[-1, -1])
    assert np.isfinite(visits).all() and np.all(visits.sum(1) > 0)
    posterior = visits/visits.sum(1, keepdims=True)
    hard = np.zeros_like(posterior); i, j = K-1, T-1
    while True:
        hard[i, j] = 1.
        if i == j == 0: break
        direction = previous[i, j]
        if direction == 0: i -= 1
        elif direction == 1: j -= 1
        elif direction == 2: i -= 1; j -= 1
        else: raise AssertionError('invalid Viterbi path')
    hard /= hard.sum(1, keepdims=True)
    return posterior, hard, float(a[-1, -1])


def words_from_segments(segments):
    words = []
    for segment, (s, e, text) in enumerate(segments):
        matches = list(re.finditer(r'\S+', text.strip()))
        for k, m in enumerate(matches):
            words.append(dict(id=len(words), segment=segment, index=k, text=m.group(),
                start=s+(e-s)*k/len(matches), end=s+(e-s)*(k+1)/len(matches),
                char_start=m.start(), char_end=m.end()))
    return words


def build_blocks(segments, words, tokenizer):
    """Contiguous whole-word bisection, with explicit character splitting of oversized words."""
    remove = str.maketrans('', '', string.punctuation.replace("'", ''))
    def items_for(ws):
        items = []
        for k, w in enumerate(ws):
            if k: items.append(dict(word=-1, char=' ', tokens=tokenizer.encode(' ', add_special_tokens=False)))
            for char_index,char in enumerate(w['text'].translate(remove)):
                ids = tokenizer.encode(char, add_special_tokens=False)
                assert ids and max(ids) < tokenizer.convert_tokens_to_ids('<|endoftext|>')
                items.append(dict(word=w['id'], char=char, char_index=char_index, tokens=ids))
        return items
    blocks = []
    def add(ws):
        items = items_for(ws)
        count = sum(len(x['tokens']) for x in items)
        span = ws[-1]['end']-ws[0]['start']
        if not any(x['word'] >= 0 for x in items): return
        if span <= 22 and count <= 400:
            if any(x['word'] >= 0 for x in items):
                blocks.append(dict(start=ws[0]['start'], end=ws[-1]['end'], items=items))
            return
        if len(ws) > 1:
            middle = len(ws)//2; add(ws[:middle]); add(ws[middle:]); return
        # Recursive character bisection preserves identity and both resource limits.
        def chars(group,s,e):
            if e-s <= 22 and sum(len(x['tokens']) for x in group) <= 400:
                if any(x['word'] >= 0 for x in group): blocks.append(dict(start=s,end=e,items=group))
                return
            if len(group)==1:
                assert len(group[0]['tokens'])<=400
                parts=int(math.ceil((e-s)/22))
                for k in range(parts):
                    item={**group[0],'weight':1/parts,'repeated_single_character':True}
                    blocks.append(dict(start=s+(e-s)*k/parts,end=s+(e-s)*(k+1)/parts,items=[item]))
                return
            middle=len(group)//2; t=s+(e-s)*middle/len(group)
            chars(group[:middle],s,t);chars(group[middle:],t,e)
        chars(items,ws[0]['start'],ws[-1]['end'])
    for segment in range(len(segments)):
        ws = [w for w in words if w['segment'] == segment]
        if ws: add(ws)
    return blocks


def frame_windows(nframes, start, windows):
    times = start+(np.arange(nframes)+.5)*.02
    ends = np.asarray([b for a, b in windows])
    return np.minimum(np.searchsorted(ends, times, side='right'), len(windows)-1)


def proportional_support(words, windows):
    out = np.zeros((len(words), len(windows)))
    for w in words:
        for i, (a, b) in enumerate(windows):
            out[w['id'], i] = max(0., min(b, w['end'])-max(a, w['start']))
        if out[w['id']].sum() <= 0:
            center = (w['start']+w['end'])/2
            out[w['id'], min(np.searchsorted([b for a,b in windows], center), len(windows)-1)] = 1.
    if len(words): out /= out.sum(1, keepdims=True)
    return out


class CrossAttentionCapture:
    """Read actual already-scaled Q/K, preserving the original SDPA computation."""
    def __init__(self, decoder, char_start, char_count, nframes):
        self.decoder = decoder; self.start = char_start; self.count = char_count; self.frames = nframes
        self.active = None; self.top = []; self.visited = []; self.hooks = []

    @contextmanager
    def capture(self):
        original = F.scaled_dot_product_attention
        def pre(i):
            def f(*args): self.active = i
            return f
        def post(*args): self.active = None
        for i, layer in enumerate(self.decoder.layers):
            self.hooks.append(layer.encoder_attn.register_forward_pre_hook(pre(i)))
            self.hooks.append(layer.encoder_attn.register_forward_hook(post))
        def wrapped(q, k, v, attn_mask=None, dropout_p=0., is_causal=False, *, scale=None, enable_gqa=False):
            if self.active is not None:
                assert q.shape[0] == 1 and not is_causal and not enable_gqa
                rows = q[0, :, self.start:self.start+self.count].float()
                keys = k[0, :, :self.frames].float()
                raw = rows@keys.transpose(-1,-2)*(scale if scale is not None else q.shape[-1]**-.5)
                if self.frames > 1:
                    raw = F.pad(raw, (1,1), mode='reflect').unfold(-1,3,1).median(-1).values
                weights = raw.softmax(-1)
                scores = weights.norm(dim=-1).sum(-1)+weights.norm(dim=-2).sum(-1)
                for h in range(weights.shape[0]):
                    self.top.append((float(scores[h]), self.active, h, weights[h].detach().cpu().numpy()))
                self.top.sort(key=lambda x: (-x[0],x[1],x[2])); self.top = self.top[:10]
                self.visited.append(self.active)
            kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            if scale is not None: kwargs['scale'] = scale
            if enable_gqa: kwargs['enable_gqa'] = True
            return original(q,k,v,**kwargs)
        F.scaled_dot_product_attention = wrapped
        try: yield self
        finally:
            F.scaled_dot_product_attention = original
            for h in self.hooks: h.remove()
            self.active = None
        assert self.visited == list(range(len(self.decoder.layers))) and len(self.top) == 10

    def matrix(self):
        mean = np.mean([a for s,l,h,a in self.top], axis=0).astype(np.float64)
        mean /= np.maximum(np.linalg.norm(mean, axis=0, keepdims=True),1e-12)
        return mean, [dict(layer=l, head=h, score=s) for s,l,h,a in self.top]


class WhisperAligner:
    def __init__(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.processor = WhisperProcessor.from_pretrained(ALIGN_MODEL,local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(ALIGN_MODEL,dtype=torch.float16,
            attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.tok = self.processor.tokenizer
        assert self.model.config.decoder_layers == 32 and self.model.config.decoder_attention_heads == 20
        self.lang_ids = sorted(self.model.generation_config.lang_to_id.values())

    @torch.no_grad()
    def align(self, audio, segments, windows):
        words = words_from_segments(segments); blocks = build_blocks(segments,words,self.tok)
        soft = np.zeros((len(words),len(windows))); hard = soft.copy(); counts = np.zeros(len(words))
        traces = []; language = None; decoders = 0
        for bi, block in enumerate(blocks):
            sample_start = max(0,int(round((block['start']-4)*16000)))
            sample_end = min(len(audio),int(round((block['end']+4)*16000)))
            assert 0 < sample_end-sample_start <= 480000
            wave = audio[sample_start:sample_end]
            features = self.processor.feature_extractor(wave,sampling_rate=16000,return_tensors='pt').input_features
            encoded = self.model.model.encoder(features.to('cuda',torch.float16),return_dict=True).last_hidden_state
            sot = self.tok.convert_tokens_to_ids('<|startoftranscript|>')
            if language is None:
                h = self.model.model.decoder(input_ids=torch.tensor([[sot]],device='cuda'),
                    encoder_hidden_states=encoded,use_cache=False,return_dict=True).last_hidden_state[0,-1]
                language = self.lang_ids[int((h.float()@self.model.proj_out.weight[self.lang_ids].float().T).argmax())]
                decoders += 1
            prefix = [sot,language,self.tok.convert_tokens_to_ids('<|transcribe|>'),self.tok.convert_tokens_to_ids('<|notimestamps|>')]
            chars = [token for item in block['items'] for token in item['tokens']]
            ids = prefix+chars+[self.tok.eos_token_id]
            assert len(chars) <= 400 and len(ids) <= self.model.config.max_target_positions
            nframes = min(encoded.shape[1], int(math.ceil(len(wave)/320)))
            capture = CrossAttentionCapture(self.model.model.decoder,len(prefix),len(chars),nframes)
            with capture.capture():
                self.model.model.decoder(input_ids=torch.tensor([ids],device='cuda'),
                    encoder_hidden_states=encoded,use_cache=False,return_dict=True)
            decoders += 1
            matrix, heads = capture.matrix(); posterior, path, logZ = path_distribution(matrix)
            which = frame_windows(nframes,sample_start/16000,windows)
            cursor = 0
            for item in block['items']:
                n = len(item['tokens'])
                if item['word'] >= 0:
                    pp = posterior[cursor:cursor+n].mean(0); hp = path[cursor:cursor+n].mean(0)
                    weight=item.get('weight',1.)
                    soft[item['word']] += weight*np.bincount(which,weights=pp,minlength=len(windows))
                    hard[item['word']] += weight*np.bincount(which,weights=hp,minlength=len(windows))
                    counts[item['word']] += weight
                cursor += n
            assert cursor == len(chars)
            traces.append(dict(block=bi,nominal_start=block['start'],nominal_end=block['end'],
                audio_start=sample_start/16000,audio_end=sample_end/16000,char_tokens=len(chars),
                items=block['items'],heads=heads,frames=nframes,logZ=logZ,language=language,
                posterior_row_sum_min=float(posterior.sum(1).min()),posterior_row_sum_max=float(posterior.sum(1).max())))
            del encoded,capture
        proportional = proportional_support(words,windows); inherited = []; originally_aligned=counts>0
        for w in words:
            wid = w['id']
            if counts[wid] == 0:
                donors = [x['id'] for x in words if x['segment']==w['segment'] and originally_aligned[x['id']]]
                if donors:
                    donor = min(donors,key=lambda x:(abs(x-wid),x))
                    soft[wid] = soft[donor]/counts[donor]; hard[wid] = hard[donor]/counts[donor]
                else: donor = None; soft[wid] = hard[wid] = proportional[wid]
                counts[wid] = 1; inherited.append(dict(word=wid,donor=donor))
        if len(words):
            soft /= soft.sum(1,keepdims=True); hard /= hard.sum(1,keepdims=True)
        assert np.isfinite(soft).all() and (not len(words) or np.allclose(soft.sum(1),1))
        return dict(words=words,soft=soft.tolist(),viterbi=hard.tolist(),proportional=proportional.tolist(),
                    blocks=traces,inheritance=inherited,encoder_calls=len(blocks),decoder_calls=decoders,
                    character_tokens=sum(t['char_tokens'] for t in traces),language=language)
