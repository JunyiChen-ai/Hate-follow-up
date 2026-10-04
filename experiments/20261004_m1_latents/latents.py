"""Paper-equation latent-input optimization; frozen backbone, no label input."""
from contextlib import contextmanager
import math
import time
import torch
import torch.nn.functional as F

K, POS, NEG = 4, 2, 4
WARMUP, SEARCH, TOP = 5, 15, 20
TAU, LR, SIGMA, DECAY, ALPHA = .1, .02, .01, .95, .001


def question_rows(judge, question, suffix, ids):
    """Offsets select original question content and exclude chat-template tokens."""
    assert suffix.count(question) == 1
    start = suffix.index(question)
    stop = start + len(question)
    enc = judge.tok(suffix, add_special_tokens=False, return_offsets_mapping=True)
    assert enc['input_ids'] == ids
    selected = [i for i, (a, b) in enumerate(enc['offset_mapping'])
                if b > a and a >= start and b <= stop]
    assert selected and all(ids[i] not in judge.tok.all_special_ids for i in selected)
    return selected


def attention_probabilities(q, k, mask=None, causal=False, scale=None, enable_gqa=False):
    """Exact SDPA probability semantics for supplied post-RoPE tensors."""
    assert q.ndim == k.ndim == 4 and q.shape[0] == k.shape[0] == 1
    if q.shape[1] != k.shape[1]:
        assert enable_gqa and q.shape[1] % k.shape[1] == 0
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    logits = q.float() @ k.float().transpose(-2, -1)
    logits *= scale if scale is not None else 1 / math.sqrt(q.shape[-1])
    if causal:
        assert mask is None
        visible = torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool).tril()
        logits.masked_fill_(~visible, float('-inf'))
    if mask is not None:
        if mask.dtype == torch.bool:
            logits.masked_fill_(~mask, float('-inf'))
        else:
            logits += mask.float()
    return torch.softmax(logits, dim=-1)


class QueryAttention:
    """Read-only SDPA interception. The original kernel supplies every model output."""
    def __init__(self, rows, image_positions, suffix_length, expected_layers):
        self.rows = rows
        self.image_positions = image_positions
        self.suffix_length = suffix_length
        self.expected_layers = expected_layers
        self.layers = 0
        self.total = None

    @contextmanager
    def capture(self):
        original = F.scaled_dot_product_attention

        def wrapped(q, k, v, attn_mask=None, dropout_p=0., is_causal=False,
                    scale=None, enable_gqa=False):
            assert q.shape[-2] == self.suffix_length and dropout_p == 0.
            # Select query rows before the large product; preserve the actual mask rows.
            selected = q[:, :, self.rows, :]
            mask = attn_mask
            if mask is not None and mask.shape[-2] != 1:
                mask = mask[..., self.rows, :]
            if is_causal:
                assert mask is None
                full = torch.ones(q.shape[-2], k.shape[-2], device=q.device,
                                  dtype=torch.bool).tril()
                mask = full[self.rows, :]
            probabilities = attention_probabilities(selected, k, mask, False, scale, enable_gqa)
            relevance = probabilities[..., self.image_positions].mean(dim=(0, 1, 2))
            self.total = relevance if self.total is None else self.total + relevance
            self.layers += 1
            del probabilities
            kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            if scale is not None:
                kwargs['scale'] = scale
            if enable_gqa:
                kwargs['enable_gqa'] = True
            return original(q, k, v, **kwargs)

        F.scaled_dot_product_attention = wrapped
        try:
            yield self
        finally:
            F.scaled_dot_product_attention = original
        assert self.layers == self.expected_layers, (self.layers, self.expected_layers)

    def relevance(self):
        assert self.total is not None and self.layers == self.expected_layers
        out = self.total / self.layers
        assert torch.isfinite(out).all() and (out >= 0).all()
        return out.detach()


def support_indices(relevance):
    assert relevance.ndim == 1 and relevance.numel() >= K * (POS + NEG)
    positives = torch.argsort(relevance, descending=True, stable=True)[:K * POS].reshape(K, POS)
    ascending = torch.argsort(relevance, descending=False, stable=True)
    # Even tied relevance must produce disjoint sets; stable ordering alone does not guarantee it.
    ascending = ascending[~torch.isin(ascending, positives.reshape(-1))]
    negatives = ascending[:K * NEG].reshape(K, NEG)
    assert set(positives.flatten().tolist()).isdisjoint(negatives.flatten().tolist())
    return positives, negatives


def wrong_support_indices(indices, counts):
    """Swap actual support content to a fixed half-video frame offset, never use labels."""
    assert sum(counts) > 0 and all(n > 0 for n in counts)
    cumulative = [0]
    for n in counts:
        cumulative.append(cumulative[-1] + n)
    mapped, bindings = [], []
    for index in indices.flatten().tolist():
        frame = next(i for i in range(len(counts)) if cumulative[i] <= index < cumulative[i + 1])
        local = index - cumulative[frame]
        target = (frame + len(counts) // 2) % len(counts)
        patch = local * counts[target] // counts[frame]
        mapped.append(cumulative[target] + patch)
        bindings.append(dict(frame=frame, patch=local, target_frame=target, target_patch=patch))
    return torch.tensor(mapped, device=indices.device).reshape(indices.shape), bindings


def contrastive_loss(h, positive, negative):
    p = F.cosine_similarity(h[:, None, :], positive, dim=-1) / TAU
    n = F.cosine_similarity(h[:, None, :], negative, dim=-1) / TAU
    return -(math.log((POS + NEG) / POS) + torch.logsumexp(p, -1)
             - torch.logsumexp(torch.cat((p, n), -1), -1)).mean()


def warmup(initial, positive, negative):
    with torch.enable_grad():
        h = initial.detach().float().clone().requires_grad_(True)
        positive, negative = positive.detach().float(), negative.detach().float()
        opt = torch.optim.Adam([h], lr=LR, betas=(.9, .999), eps=1e-8, weight_decay=0.)
        losses = []
        for _ in range(WARMUP):
            opt.zero_grad(set_to_none=True)
            loss = contrastive_loss(h, positive, negative)
            assert torch.isfinite(loss)
            losses.append(float(loss.detach()))
            loss.backward()
            opt.step()
        losses.append(float(contrastive_loss(h, positive, negative).detach()))
    return h.detach(), losses


def progression_entropy(logits):
    top = logits.float().topk(TOP, dim=-1).values
    lp = torch.log_softmax(top, dim=-1)
    entropy = -(lp.exp() * lp).sum(-1)
    reward = (entropy[:-1] - entropy[1:]).clamp_min(0).mean()
    assert torch.isfinite(entropy).all() and torch.isfinite(reward)
    return float(reward), entropy.tolist()


@torch.no_grad()
def search(initial, evaluate, generator):
    """One selected evaluated state; no prediction ensemble or unscored-center selection."""
    center = initial.detach().clone()
    reward, entropy, _ = evaluate(center)
    best, best_reward, best_index = center.clone(), reward, -1
    trace = [dict(index=-1, reward=reward, entropy=entropy)]
    for i in range(SEARCH):
        sigma = SIGMA * DECAY ** i
        eps = torch.randn(center.shape, device=center.device, dtype=center.dtype,
                          generator=generator) * sigma
        candidate = center + eps
        reward, entropy, _ = evaluate(candidate)
        trace.append(dict(index=i, sigma=sigma, reward=reward, entropy=entropy))
        if reward > best_reward:
            best, best_reward, best_index = candidate.clone(), reward, i
        center = center + ALPHA / sigma ** 2 * reward * eps
        assert torch.isfinite(center).all()
    return best, dict(best_index=best_index, best_reward=best_reward, trajectory=trace)


class LatentReader:
    def __init__(self, judge):
        self.j = judge
        self.weight = judge.model.get_output_embeddings().weight.detach().float()

    @torch.no_grad()
    def evaluate(self, cache, h, delta, fresh=False):
        import copy
        j, n = self.j, cache.get_seq_length()
        c = copy.deepcopy(cache) if fresh else cache
        j.model.model.rope_deltas = delta.clone()
        positions = torch.arange(n, n + h.shape[0], device=j.device)[None, :] + delta
        kwargs = dict(inputs_embeds=h[None].to(j.dtype), position_ids=positions[None].expand(3, -1, -1),
                      attention_mask=torch.ones(1, n + h.shape[0], device=j.device, dtype=torch.long),
                      past_key_values=c, use_cache=True)
        if 'cache_position' in j.forward_params:
            kwargs['cache_position'] = torch.arange(n, n + h.shape[0], device=j.device)
        try:
            out = j.model.model(**kwargs)
            hidden = out.last_hidden_state[0]
            logits = hidden.float() @ self.weight.T
            if j.softcap:
                logits = torch.tanh(logits / j.softcap) * j.softcap
            reward, entropy = progression_entropy(logits)
            margin = j.margins_fp32(hidden[-1:])[0]
        finally:
            c.crop(n)
            j.model.model.rope_deltas = delta.clone()
        assert cache.get_seq_length() == n
        return reward, entropy, margin

    def read(self, cache, embeddings, relevance, counts, delta, generator, arm='full', smoke=False):
        p, n = support_indices(relevance)
        detail = dict(positive=p.tolist(), negative=n.tolist(), relevance_max=float(relevance.max()),
                      relevance_sum=float(relevance.sum()), actual_content_changed=None)
        if arm == 'wrong_support':
            p, bp = wrong_support_indices(p, counts)
            n, bn = wrong_support_indices(n, counts)
            detail.update(wrong_positive=bp, wrong_negative=bn,
                          actual_content_changed=bool(not torch.equal(embeddings[p], embeddings[
                              torch.tensor(detail['positive'], device=p.device)])))
        positive, negative = embeddings[p].float().detach(), embeddings[n].float().detach()
        initial = positive.mean(1)
        h = initial.clone()
        detail['initial_norm'] = initial.norm(dim=-1).tolist()
        if arm not in ('initial', 'search_only'):
            h, detail['losses'] = warmup(h, positive, negative)
        if arm not in ('initial', 'warmup'):
            h, trace = search(h, lambda state: self.evaluate(cache, state, delta), generator)
            detail.update(trace)
        reward, entropy, margin = self.evaluate(cache, h, delta)
        detail.update(final_reward=reward, final_entropy=entropy, final_norm=h.norm(dim=-1).tolist(),
                      input_change_norm=float((h - initial).norm()), margin=margin)
        if arm not in ('initial', 'warmup'):
            assert abs(reward - detail['best_reward']) < 1e-6
        if smoke:
            if h.is_cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            fresh = self.evaluate(cache, h, delta, fresh=True)
            assert fresh == (reward, entropy, margin), ('fresh-cache discrepancy', fresh, (reward, entropy, margin))
            alternate = self.evaluate(cache, torch.roll(h, 1, dims=0), delta)
            detail.update(cloned_cache_exact=True, alternate_slot_margin=alternate[2],
                          changed_slots_change_margin=alternate[2] != margin)
            if h.is_cuda:
                torch.cuda.synchronize()
            detail['diagnostic_seconds'] = time.perf_counter() - started
        return margin, detail
