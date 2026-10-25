# Rule 4 proposal review: candidate 50 (act-state reading of the window)

Date 2026-10-25. **VERDICT: PASS** (A1 the mask form, and A2 the transfer form if A1 passes)

One independent agent did this review, once, before any GPU run.

Files read:
- `RESEARCH_ITERATION_RULES.md` rule 4;
- `CLAUDE.md`;
- the two earlier M3 review records of today;
- the 2026-10-02 Grounder proposal review (`docs/reviews/20261002_m1_grounder_proposal.md`);
- `archive/experiments/20261002_m1_grounder/README.md`;
- the team scratchpad: `review_m1.md` §3, §4 and §5.3, `design_m1.md` proposal 3, `design_story.md` proposal 2 and §2.2, `lit_m1.md` §C4, §C7 and §F, and `history.md` M1-f.

The engineer README `experiments/20261025_m1_act_state/README.md` did not exist when the search ended, so this review covers the design as written in the scratchpad.

No repo file was changed apart from this one. No GPU was used. No hash or checksum was computed.

## Candidate as reviewed

Today, r6_bma reads each 8 s window with an isolated Yes/No branch. The branch forks from a cached prefix that holds:
- the policy rules;
- 20 frames;
- the full timestamped transcript;
- the model's video-level verdict;
- a one-word stance turn.

The branch attends to the whole prefix.

**A1, the mask form.** At every layer, the window branch may attend only to:
- the scaffold tokens;
- its own window's frame tokens and transcript tokens in the prefix;
- its own question tokens;
- one global span: the verdict question, the model's verdict answer and the stance turn.

Everything else in the video is masked.

**A2, the transfer form.** It runs only if A1 passes. The hidden state at the verdict and stance positions is moved into a short local read of the window.

**Controls:**
- `all_local`: the same mask without the verdict span. Grounder ran this arm on 2026-10-02.
- A1-other: the same mask, with the verdict-turn states of a donor video that has the same verdict sign.

**Thesis.** The video-level act reaches each moment through the model's own decision state, not through the content of the rest of the video.

## Search

32 web searches (31 standard, 1 extended). I also searched the five baseline-paper HTML files in the repo root.

**Pages opened beyond the search summary:**
- FBHM, arXiv 2605.31349 (Hugging Face page, method details);
- MultiHateLoc, arXiv 2512.10408v2 (HTML, architecture);
- RAMF, arXiv 2512.02743 (abstract page);
- "Language Models Can Control Their Own Attention", arXiv 2609.02737 (HTML, first 100k characters);
- SteerVAD, arXiv 2602.24021 (abstract page and HTML, first 100k characters).

The SteerVAD PDF could not be fetched because it was too large.

**Only the search summary was read:** everything else.

### A. Hate work (hateful video, hateful memes, multimodal hate)

1. Hateful video localization with an attention mask or hidden state on an MLLM → LELA, MARS, DecAF. DecAF uses attention rollout for video segmentation, not for hate. LELA and MARS use only prompting.
2. Hateful video detection with a task vector or activation patching → HVGuard, LELA, MARS. None uses internal states.
3. Hateful memes with in-context vectors or hidden-state steering → **FBHM (arXiv 2605.31349, 2026)**. This is the only hate work found that edits hidden states.
   - It learns one steering vector and one scale per layer, and adds them to the hidden state.
   - It trains them on 500 labelled samples. The loss is KL to the 32-shot in-context output, plus cross-entropy on the label.
   - It classifies whole memes. It has no attention mask, it copies no state between passes, and it does no localization.
   - Others in this search: Evolver, RA-HMD and 2504.09914 use prompting, retrieval or fine-tuning.
4. Activation steering for hate speech → ActAdd, SafeSteer, CAST. These steer generation or refusal in text. None detects or localizes hate in content.
5. Harmful video segments with local attention plus a global token → 2508.10974, a gaze-attention paper with learned context tokens, TG-Heads, GLUS. None is hate localization with a mask.
6. Hateful video localization on HateMM / HateClipSeg in 2026 → LELA, the HateClipSeg dataset paper. Neither has a global-plus-local attention design.
7. Multimodal hate with local/global attention → MM-HSD, MLCA, RAMF, MultiHateLoc.
   - These are trained fusion networks. The MultiHateLoc HTML confirms full-sequence self-attention, no window-plus-summary mask, no MLLM, and training with video labels.
   - RAMF's "local-global context fusion" is a detection fusion module. Its abstract does not mention any frozen-MLLM mask.
8. Mechanistic interpretability of hateful memes → Miyanishi et al. 2023 (gradient attention attribution), Hee and Lee (ICWSM 2025), RA-HMD, AdamMeme. None restricts attention or transfers states.
9. Content moderation with attention masking or context isolation, span-level toxicity → nothing relevant.

The earlier records add to this list:
- the 2026-10-02 Grounder review: MultiHateLoc, HVGuard, RAMF, CLARA, LEAF and SAGE, from their primary sections;
- `lit_m1.md` §F: Pro-Cap, ExplainHM, MARS, RAMF, HVGuard, LEAF, SafeLens, LELA, MultiHateLoc, IARE and CLARA.

None restricts a reader's attention to its local span plus a decision state, and none transfers such a state.

### B. General precedents and how candidate 50 differs

**Local span plus global tokens in the mask.**

- **Longformer and BigBird.** Sliding-window attention plus a few global tokens such as CLS or the question.
  - These are trained architectures.
  - Their global tokens are fixed positions, attend in both directions, and exist for efficiency.
  - In A1 the model is frozen and the mask applies only to the branch queries. The single global span is the model's own generated decision about this input.
- **StreamingLLM and LM-Infinite.** These are training-free. Their mask keeps a few starting tokens plus a recent window, so the global part is an attention sink that keeps attention numerically stable.
- **Star Attention (ICML 2025).** Each local block also attends to a shared first "anchor" block, which absorbs attention sinks. Then the query attends to everything.
  - In all three of these papers, the global part is a position, not content.
  - None of them is used to score parts of the input.
- **Gist tokens (NeurIPS 2023), LightThinker (EMNLP 2025), Video-XL's visual summarization tokens (CVPR 2025).** Later tokens see only learned summary tokens.
  - All three are trained, and their purpose is compression.
  - A1 trains nothing. Its "summary" is the model's own verdict turn, and the window's own evidence stays visible.
  - The scratchpad review already named Gist as the mask idea behind A1.
- **"Language Models Can Control Their Own Attention" (arXiv 2609.02737, 2026).** This is the closest topology to A1.
  - It is training-free. In "focus" mode the model attends only to a chunk it names, the scaffold (system instruction and question) and its own response so far. Other chunks are masked.
  - It is used for long-context text QA.
  - Differences from A1:
    - in that paper the model chooses the chunk; in A1 the chunk is fixed by the window's time;
    - its visible own text is running reasoning, not a decision about the whole input;
    - it does not score parts of the input;
    - it has no video and no hate.
- **Parallel Context Windows, Superposition Prompting, Block-Attention (ICLR 2025).** Units are encoded without seeing each other, and the final query sees all of them. This is the reverse of A1. It is the topology r6 itself already borrows.

**Masks that keep tokens by attention.**

- **SnapKV (NeurIPS 2024), H2O, FastV, SparseVLM, TopV.**
  - These keep the tokens that the question or the end of the prompt attends to, in order to compress the cache.
  - A1's kept set is fixed by meaning and time: the decision span plus the window's own tokens. It is not chosen by attention, and the purpose is not compression.
- **TG-Heads, "MLLMs Know When Before Speaking" (arXiv 2605.21954).**
  - It finds the interval that the prefill attention concentrates on.
  - It then re-runs Qwen3-VL-8B with the visual context masked to that interval, for temporal grounding.
  - It has no global decision span and no hate.

**Blocking attention edges as a tool.**

- **Attention knockout.** Geva et al. 2023; Zhang et al., "Cross-modal Information Flow in MLLMs", CVPR 2025; arXiv 2411.17491.
  - These papers block attention between token groups at chosen layers to trace where information flows.
  - They use blocking for analysis. A1 uses blocking as part of the scoring path.
  - A1-other is an interchange intervention with a donor input, a standard form in causal mediation work.

**Moving a state into another pass (A2).**

- **Task vectors (Findings EMNLP 2023), function vectors (ICLR 2024), ICV (ICML 2024), I2CL (ICLR 2025), multimodal task vectors, cross-modal task representations (ICML 2025), Patchscopes (ICML 2024).**
  - These build a state from demonstrations or from a source prompt.
  - They add or patch that state into a query pass, to transfer a task or to inspect the state.
  - ICV's tasks include safety (detoxifying generated text). That is generation, not detection.
  - In A2, the state is this same video's own verdict state, and it conditions reads of parts of the same input for localization.
- **SteerVAD (ICLR 2026, arXiv 2602.24021).** This is the closest work in video anomaly detection. HTML method read.
  - It uses a frozen MLLM.
  - Its "global context" is the hidden state of the first generated token.
  - A meta-controller turns that context into scaling of 4 selected attention-head outputs.
  - The meta-controller is trained with per-segment anomaly labels on 1% of the training set. The scaled features go to a trained logistic scorer.
  - It does no attention routing, it copies no state between passes, and it has no hate.
- **HiProbe-VAD and HeadHunt-VAD.** Probes on hidden states or heads, fitted with labels. Not hate.
- **arXiv 2607.12304.** It uses residual-stream activation patching on a video VLM, but only to analyze temporal order in benchmarks.

**Repo baseline HTML files** (`ruler.html`, `t3al.html`, `tfvtg.html`, `vadtree.html`, `vera.html`). Searched for "attention mask", "global token", "hidden state", "task vector", "steering", "patch", "summary token" and "global context". The only hit is AnomalyRuler using "steering" in the sense of prompting.

### Conclusion of the search

- No hateful video detection or localization work restricts a reader's attention to its own span plus the model's decision state.
- No hateful video work moves a decision state into local reads.
- Among hateful memes, FBHM uses learned steering vectors for detection. That is a different source: the vectors are learned from labels, it is not transfer from the same instance, it has no mask, and it does no localization.
- The general precedents above are trained, aimed at efficiency or analysis, or keep sinks or attention-selected tokens. None scores the parts of an input under the model's own decision about that input.

## The four STOP conditions

1. **Source method already used for hateful video detection or localization: no.** See the search.
   - The mask topology has precedents in general work: Gist, Dynamic Attention, Star Attention, LM-Infinite.
   - Moving states between passes has precedents in general work: ICV, I2CL, function vectors, Patchscopes.
   - None of them is in hate. FBHM is for memes, uses a different mechanism, and does detection only.
2. **Pure ensemble: no.**
   - There is one frozen MLLM, with one read per window and modality.
   - `all_local` and A1-other are controls. They are not averaged into the score.
3. **Pure calibration, post-processing or smoothing: no.**
   - A1 changes which tokens the branch can attend to, while the window logit is being computed inside the model.
   - A2 changes the hidden states of that computation.
   - The r6 time model is unchanged, and the candidate makes no claim about it.
4. **Pure engineering trick: no.**
   - Prompt wording, the encoder, the frames, the resolution and the constants are all unchanged.
   - The access set is not a tuned constant. It states a testable route for information: the act reaches the moment only through the decision span. The donor control tests that route directly.

Two notes, neither of which is a STOP reason:
- A1 is one more cell of Grounder's mask design (M1-f in `history.md`). Rule 4 does not list "same family as an earlier repo run" as a STOP reason.
- The low prior estimated before the run (about 5%, from the `all_local` per-video rescaling bound) is reasoning before implementation. Rule 4 explicitly excludes that as a reason to STOP. The test results will decide.

## Advisories before the run (not blocking)

1. **Write the claim as a route, not as isolation.**
   - In the r6 prefix, the window's own frame and transcript states were computed with causal attention to everything before them.
   - The verdict-turn states also attend to the whole video.
   - So A1 blocks *direct* access. The rest of the video still reaches the branch through the own-window states and the decision states.
   - Use Grounder's wording ("restricts direct attention access, not all information flow").
   - The difference between A1 and `all_local` isolates the decision span's contribution. It does not show that the moment is read "from its own evidence alone".
2. **The donor control must change only the states.**
   - A1-other is meant to differ from A1 only in the states of the verdict span.
   - So the donor's verdict tokens *and* its one-word stance token should match the target's.
   - Log how often a matching donor exists, and the fallback rule if none exists.
   - Write down the RoPE re-rotation of the donor keys (`src/pre_rotary_memory.py`) and the position offsets before the run.
3. **Fix A2's form before running it.** The two design notes describe two different A2 forms:
   - `design_m1.md` #3 adds the act state minus a content-free state, at the question's last token, at one layer (18 of 36, scale 1). This is the ICV / function-vector form.
   - `design_story.md` #2 patches the verdict-position states into a placeholder at every layer. This is the Patchscopes form.
   - Choose one, and declare the layer, the scale, the positions and the content-free baseline in the README before any A2 run (rule 7).
4. **Cite the closest work and claim only the difference.**
   - Mask topology: Gist, Dynamic Attention (2609.02737), Star Attention, StreamingLLM / LM-Infinite, Longformer / BigBird.
   - Blocking edges: attention knockout (Geva et al.; Zhang et al., CVPR 2025).
   - A2: ICV, I2CL, task vectors, function vectors, MTV, Patchscopes.
   - Hate-domain steering: FBHM.
   - VAD precedent of a global state steering local features: SteerVAD.
   - What can be claimed: the reader's own decision about the whole video is the one channel through which the rest of the video reaches each moment's read, for localization, with no training. The mask or the vector by itself cannot be claimed.
5. **Read the result against the existing cells.**
   - Grounder `verdict_only` (verdict blocked in the late layers, media visible) equalled base.
   - `all_local` lost .046 / .049 within.
   - If A1 recovers a gain over `all_local` but does not beat A1-other, the decision span acts as a per-video level for the reads, which is what P10 and the HVL permutation result found. Archive under that heading.
   - The stage-0 GT-free stop rules and the stage-1 pass conditions in `review_m1.md` §5.3 are the right form for this.
6. **Cost (rule 5, for the README).** A1 adds no model calls: Grounder measured the branch time at .991× base. A2 adds one short local pass per window and modality, about 1.5× the r6 reading cost according to the design notes. State both in the README, with the GPU minutes per corpus.
