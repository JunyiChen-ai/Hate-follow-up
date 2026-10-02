# M1 Highlighter: local visual-semantic value guidance

Candidate15, declared2026-10-03 before Reinforcer or Projector full performance
is read. Independent candidate, not combined with either. Frozen Qwen3-VL-8B,
native20frames/fullASR/policy/prompts/global answer/native speech, unchanged r6
and canonical4fps evaluator. Development-selected; no implementation before
independent rule4 proposal review.

## Source and hypothesis

Source: Zhao et al., *Tell Model Where to Look: Mitigating Hallucinations in
MLLMs by Vision-Guided Attention*, arXiv2511.20032v1:
https://arxiv.org/html/2511.20032v1 ; https://github.com/beta-nlp/VGA .
Read paper sections3–4/A.3/E and official Qwen2.5 evaluation/attention code.
Source snapshots (read-only, no content hashes): `third_party/vga_source_read/`.

VGA obtains a spatial guidance distribution from the vocabulary projection of
visual-token hidden states, then adds a guidance-weighted visual value vector
to native attention output, with a head-dependent balancing coefficient. This
differs from imposing a mask (Grounder), amplifying existing attention logits
(Amplifier), reallocating sink attention (Recycler), or contrasting two output
vectors (Projector). Native access to the full context is preserved.

Our hypothesis: explicit access to semantically informative patches inside the
queried window can help local visual discrimination without deleting context.
This is not established by the source's single-image object-hallucination result.
The transfer replaces object-name grounding by object-agnostic visual semantic
salience restricted to timestamped frames in the queried window. It is an
adaptation, not an exact source reproduction or a new semantic certainty measure.

## Exact R1, fixed before scoring

1. Encode the native full prefix once, unchanged. Capture the actual final-normalized
   hidden states at image-token positions (all noncontiguous frame blocks). Compute
   their full-vocabulary logits with the original output head in FP32. For each
   visual token, take its10 largest probabilities under the FULL-vocabulary softmax
   and sum `-p*log(p)/log(10)`. Do NOT renormalize the top10. This follows official
   object-agnostic code; paper equation4 omits the probability factor. This is a
   truncated entropy statistic, not probability of hate or validated certainty.
   Compute from finite top10 log-probabilities; p=0 contributes0, never0*log(0).
   Process128visual rows at a time to bound vocabulary memory; no labels or objects
   extracted by an extra model. No token IDs selected from observed test outcomes.
2. For each visual-window question, restrict those nonnegative values to image
   tokens whose frame timestamps satisfy `start<=t<end` (last window includes
   `t==duration`), then sum-normalize to G. If no frame is inside the window, use
   native computation exactly. If all retained weights are zero, use uniform G
   on those local image tokens. No nearest-frame substitution or hidden filling.
3. Only the LAST row of the window-query suffix changes, in language layers4..17
   inclusive (zero-indexed: borrow source Qwen start4, stop before Qwen3 midpoint18).
   All32heads; native SDPA computes O. Repeat actual cached V for GQA and form
   D=sum(G_j*V_j), using only local image keys. In FP32, compute
   `s_h=(1+cos(O_h,D_h))/2`, with each cosine norm denominator clamped at1e-8
   and the numerical cosine clipped to[-1,1].
   `gamma_h=relu(2-H*s_h/sum(s))`; if sum(s)==0 set gamma_h=1.
   Add `0.2*f*gamma_h*D_h` to O_h, where f is the fraction of LOCAL image weights
   G_j>1e-8. Cast back to model dtype before native o_proj. This reproduces the
   author's active-support fraction on the selected image region, not on all
   out-of-window zero padding. No renormalization of the resulting attention
   output, no explicit QK reconstruction, no new positional encoding.
4. Original earlier query rows, global score/forced answer, speech questions and
   prefix KV remain native. Crop branch suffix KV and restore wrappers afterwards.
   Read the original finalNorm/head12-token Yes/No margin. No decoding momentum,
   generated-caption feedback, new encoder, extra context or output-score ensemble.

Source beta0.2/topK10/start4 borrowed without a local scan; end17/window restriction,
last-row-only intervention, FP32 vocabulary and guidance arithmetic and Qwen3
DeepStack/multi-image handling are explicit adaptations. Source parameters were
tuned on hallucination benchmarks, not parameter-free. Source paper has conflicting
beta/gamma naming and start/end descriptions; official all.sh Qwen uses4..15 and.2.
Object-agnostic code uses entropy, unlike paper equation4; implementation follows
the code formula stated above. No caption generation so its later-token suppression
does not apply. Salience can favor irrelevant objects; test results decide utility.

## Cost and falsification

Let V be visual windows and B=V+available speech. Deployed3+B outer forwards,
native prefix reused; paired native+guided collection3+B+V. Capturing prefix image
states and projecting all of them to the full vocabulary adds real compute and
temporary memory even though outer-call count is unchanged. Store only salience,
frame/token/window mapping and layer update diagnostics; discard full logits.
Initial paired full333 estimate15–25GPUmin on5090, deployment10–15min; replace
with actual5video smoke timings/peak memory before full launch. No new media caches.

Main arms native/highlight; native must exactly reproduce base_gridA/global/speech.
Independent code review before GPU, actual36layer small multimodal model FP32/BF16,
noncontiguous20frame mapping, GQA, original SDPA/causal/earlier-row invariance,
zero intervention, full-vocabulary salience, branch restoration and full333
canonical analysis plumbing. Smoke first2per corpus plus long-prefixHMM114, no GT.

Acceptance: final within+.01 both vs current/native, pooled losses<=.005; same
pipeline/constants both corpora. No final main gain>=.01 =>archive. Qualifying
main result triggers uniform-local-G ablation and wrong-window-G/value support
control (fixedseed0 derangement mapping declared before run; no valid mismatch
for single-window/single-support cases, report them separately). Removal of a
claimed component must cost>=.01 on a common main metric both. Compare visual raw
ordering and max/speech/final ordering, unchanged-order and single-window gains,
paired-video uncertainty, content cases and cost. A salience map or nonzero
intervention alone does not support a localization mechanism.

Proposal-review clarifications: cached local image values are already contextualized
by causally preceding input, including earlier frames; this selects a local support,
not a semantically pure local representation. Uniform-G changes D and consequently
head balancing gamma and possibly f. Save G/f/gamma/update norm diagnostics; that
control can test the whole weighting rule, not attribute its effect solely to patch
semantic ordering. A stronger component-specific claim needs additional controls.

Inputs already inspected: STATUS and previous candidate reports, original
yesno_question/prefix construction, VGA primary sources; no new GT or performance
screening for Highlighter. FV-Action/arXiv2608.08315 was read as nearby binary
scanning, not selected: our existing reader already uses binary window judgments.
VideoTree/VAP were source-discovered, not implemented; no external dataset added.


Independent proposal PASS: `docs/reviews/20261003_m1_highlighter_proposal.md`.
All narrow clarifications incorporated before implementation. Own numerical and
actual36layer multimodal FP32/BF16 checks PASS:
`runs/20261003_m1_highlighter/selfcheck/numerics.json`. Includes full-vocabulary
salience oracle, underflow-safe entropy, local/end-boundary/empty-frame mapping,
zero/empty intervention exact native, all prefix KV/mRoPE restoration and actual
layers4..17. Independent code review underway; no GPU or performance read yet.
Native timing includes a read-only visual hidden capture but excludes vocabulary
projection; report this measurement limitation. Deployment retains an FP32 LM
head copy (approximately2.3GiB on Qwen3-8B) to avoid copying it for every chunk;
peak memory and projection time must be measured in real smoke.
