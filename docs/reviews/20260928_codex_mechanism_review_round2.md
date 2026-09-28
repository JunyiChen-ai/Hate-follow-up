# Codex consultation on the mechanism, round 2, 2026-09-28

Round 1: `docs/reviews/20260928_codex_mechanism_review.md`. This round ran in the same Codex thread (gpt-6-astra). The
sandbox still could not run shell commands, so everything below came from pasted text.

**What Codex received:**
- the continuity check: `runs/20260928_headroom/continuity/table.txt`;
- the resolution check: within .954 / .986 if every 8 s window carried its true hate share;
- our doubts about the dense student;
- facts about the read:
  - a single Yes/No log-odds per branch;
  - the prefix content;
  - reading windows alone costs .021 / .071 within;
  - the window-level plateau is .76 / .62;
  - the topic confound;
  - 8-reader average against the 32B alone;
  - rule 14(e);
- the available caches and hardware.

**Question:** the bottleneck is now the accuracy of each window's single read, and no ensemble is allowed. Propose at
most 3 redesigns that make each read more accurate, or that take more information from the same model and forward
pass. Otherwise say plainly that nothing is likely to work, and name the one constraint change that would open the
most promising path.

A check of its one proposal on cached reads is in `experiments/20260928_headroom/README.md`, section "Codex
consultation, round 2".

## Reply (verbatim)

1. **I accept the rejection of continuity-based coupling and withdraw the dense-student recommendation.**

   The continuity check is adequate to reject the proposed signals at the current resolution. It does not exclude every possible discourse representation, but pursuing richer representations now would change the proposal rather than rescue it. [Source: `runs/20260928_headroom/continuity/table.txt`.]

   The oracle-window results establish that the 8-second grid preserves enough information for much better localisation. They do not prove that M3 is optimal given noisy observations, but they strongly support prioritising measurement accuracy over finer sampling. [Source: `experiments/20260928_headroom/continuity_check.py` and its reported outputs.]

   The dense student lacked a credible source of better supervision. Distillation might reproduce r6 efficiently, but neither distillation nor temporal consistency explains how to correct the teacher's topic/attack confusion. Existing self-training failures reinforce that objection. [Source: `research-wiki/DIRECTIONS.md`, SDL/NGA.] I would not spend GPU time on it.

2. **The only redesign I would currently fund is a small pilot of layer-dependent access to context. Its probability of clearing both corpora is low.**

   **Mechanism — replace part of M1.** Keep the shared prefix, isolated branches and existing questions. In early transformer layers, each window branch attends to the complete prefix as now. In the final layers, restrict its direct attention to the policy, its own question and window transcript, and any visual tokens belonging to that window. Block direct access to the global verdict and out-of-window evidence in those final layers.

   Early layers can resolve references and interpret the window using global context. Later layers must make the decision with direct access concentrated on local evidence. This does **not** make the representation context-free: global information remains in contextualised representations and residual connections. The intervention changes repeated access to context, not the availability of context altogether.

   **Why this could escape the plateau.** It changes the computation that produces the answer, rather than changing the wording or applying another function to the same final logits. The hypothesis is specific: repeated late-layer access to global topic and verdict information overwhelms weaker local evidence about the speech act. Existing results do not establish that hypothesis, but they also do not directly test it.

   Removing context entirely hurts, while the novelty review describes the stance turn mainly as a broad score shift. Together these motivate separating contextual interpretation from final evidence access; they do not establish that the separation will help. [Source: `docs/reviews/20260927_final_novelty_review.md` §§6.1, 6.3.]

   **Closest previous attempts.** Window-only reading removes context throughout the computation. Context subtraction subtracts completed scores. Branch isolation prevents windows from seeing each other's answers. This proposal retains early contextual processing but changes attention access by layer within each branch. Those are concrete computational differences. [Sources: `research-wiki/DIRECTIONS.md`, A1 "共享前缀 + 每窗隔离分支"; A2 "语境对比"; novelty review §6.1.]

   Attention masking itself is established machinery. A defensible contribution would require evidence that **when the model accesses context matters for local discrimination**, not merely a favourable mask configuration.

   **Cost.** Zero additional MLLM calls; retain the two forwards per video and reuse the prefix cache across branches. Existing reads provide the reference, but cannot produce the intervention outputs. Frames and transcripts are reusable. Target approximately the existing **1.5–2 seconds/video**, allowing at most 25% measured overhead. Layer-specific masking could disrupt efficient attention execution; reject the implementation if it materially exceeds that budget.

   **Cheapest falsification test.** Existing scalar caches cannot test this mechanism. Use at most 50 videos per corpus, selected before inspecting outcomes. Fix one depth split in advance—for example, restrict attention in the final quarter of layers. Do not search many depths.

   Compare:

   - Original full-context branches.
   - The proposed layer-dependent restriction.
   - An equally sized restriction retaining a different window's evidence.
   - Global-verdict removal alone, to test whether that explains the effect.

   Measure official frame metrics and window-level within AUC. Inspect whether attack-versus-mention discrimination improves, rather than only video offsets.

   **Kill criteria:** stop if the correct-window restriction does not outperform the wrong-window control, if verdict removal explains the entire gain, or if window discrimination improves on only one corpus. For advancing beyond the pilot, require ≥.01 frame-level within improvement on both and no pooled regression beyond the permitted floor. With only 50 videos, these are screening decisions, not conclusive statistical exclusions. Full-corpus confirmation and the same controls remain necessary before a novelty claim.

   I would not propose a second or third redesign merely to fill the list. Intermediate-layer decoding, hidden-state directions, extra answer tokens and further semantic decomposition presently lack enough evidence to distinguish them from the failed readout variations. The earlier probe limitations justify uncertainty, not another confident recommendation. [Source: `research-wiki/DIRECTIONS.md`, A2 "真标签上限", B "读出瓶颈"/"答案对比", and D.1.]

3. **Nothing currently has a high probability of satisfying all the constraints. The single change I would choose is allowing a small amount of segment-level supervision to adapt the reader.**

   Specifically, supervise the distinction between a target-group mention and an endorsed attack in raw window-plus-context inputs. Adapt the single backbone or a small adapter, keeping the same deployment calls and temporal protocol. Use separate training annotations; never use gate-test labels in fitting.

   This directly supplies the distinction that deterministic prompting repeatedly fails to extract. The repository's supervised representation probe shows that at least some relevant distinctions can be present but inaccessible to label-free readouts; it does **not** establish successful supervised temporal localisation. [Source: `research-wiki/DIRECTIONS.md`, B "读出瓶颈".] The failed supervised combinations of cached scores further suggest that supervision would need to reach the reader, rather than another downstream combiner. [Source: `experiments/20260928_headroom/README.md`.]

   This would abandon the label-free—and training-free—claim. I recommend it only as the explicit constraint change, not as a compliant redesign.

   Reader diversity has stronger **direct measured** evidence of additional information, but the demonstrated eight-reader solution conflicts with both the ensemble prohibition and the inference-cost constraint. Relaxing only the ensemble rule does not make that demonstrated solution affordable; the tested cheap variants have not resolved this. [Source: `experiments/20260928_headroom/README.md`.]
