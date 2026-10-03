"""Same-model reference attention-output preservation; no labels or score fusion."""
import torch


def blend(output, reference, alpha):
    assert output.shape == reference.shape and 0 <= alpha <= 1
    if alpha == 0:
        return output
    return ((1-alpha)*output.float()+alpha*reference.float()).to(output.dtype)


class AttentionPreserver:
    def __init__(self, judge):
        self.judge = judge
        self.layers = judge.model.model.language_model.layers
        assert judge.family == 'qwen3_vl' and len(self.layers) == 36
        self.mode = None
        self.reference = []
        self.visited = []
        self.geometry = []
        self.handles = [layer.self_attn.o_proj.register_forward_hook(self.hook(i))
                        for i, layer in enumerate(self.layers)]

    def hook(self, i):
        def observe(module, inputs, output):
            if self.mode is None:
                return None
            assert output.ndim == 3 and output.shape[:2] == (1, self.query_length)
            assert i == len(self.visited)
            self.visited.append(i)
            if self.mode == 'capture':
                self.reference.append(output.detach().clone())
                return None
            assert self.mode == 'mix' and len(self.reference) == 36
            ref = self.reference[i]
            new = blend(output, ref, self.alpha)
            self.geometry.append(torch.stack((output.float().norm(), ref.float().norm(),
                                              (new.float()-output.float()).norm())))
            return new
        return observe

    @torch.no_grad()
    def run(self, cache, ids, rope, mode, alpha=.5):
        assert self.mode is None and mode in ('capture', 'mix') and len(ids)
        if mode == 'capture':
            self.reference = []
            self.query_ids = list(ids)
        else:
            assert list(ids) == self.query_ids and len(self.reference) == 36
        self.judge.model.model.rope_deltas = rope.clone()
        n = cache.get_seq_length()
        self.query_length = len(ids)
        self.visited = []
        self.geometry = []
        self.alpha = alpha
        self.mode = mode
        try:
            h = self.judge._step(cache, ids)
        finally:
            self.mode = None
            cache.crop(n)
        assert self.visited == list(range(36))
        assert torch.equal(self.judge.model.model.rope_deltas, rope)
        # Match the shared native 1 x D FP32 projection and reduction path.
        logits = self.judge._logits_fp32(h[None], self.judge.yes_ids+self.judge.no_ids)[0]
        ny = len(self.judge.yes_ids)
        margin = float(torch.logsumexp(logits[:ny], 0)-torch.logsumexp(logits[ny:], 0))
        assert torch.isfinite(logits).all()
        geometry = torch.stack(self.geometry).cpu().tolist() if self.geometry else []
        return margin, logits.cpu().numpy(), geometry

    def clear(self):
        assert self.mode is None
        self.reference = []
        self.geometry = []

    def close(self):
        self.clear()
        for h in self.handles:
            h.remove()
