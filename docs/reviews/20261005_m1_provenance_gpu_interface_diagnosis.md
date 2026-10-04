# Candidate 25: frozen GPU source-interface diagnosis

Date: 2026-10-05. Independent same-model reviewer; **same-family provisional**. This is the requested narrow implementation/source-interface diagnosis of the five-video Slurm 138 input artifacts, not another proposal review or an effectiveness evaluation.

**Conclusion: the recorded interface A did not satisfy its declared structural guard. No observation-changing implementation defect was established in the inspected actual GPU generation, input construction or parsing paths. Do not relax the guard or interpret these observations as evidence against the scientific idea.** A local Transformers 4.57.6 decoding-position defect was reproduced, then explicitly excluded as an explanation for this run after inspecting both GPU machines' actual Transformers 5.15.1 implementation.

## Actual observations

All 158 ledger calls are invalid under the frozen exact schema: 156 omit `utterance.speaker` (usually emitting `utterance: {}`), and two entities omit the required `quote` field. Empty entity/action/quotation arrays are themselves legal; the missing required fields are not. None of these ledger calls exhausted the 512-token limit. For example, the 24-token empty ledger containing `utterance: {}` is correctly rejected rather than silently converted to `{"speaker":"UNKNOWN"}`.

Of 22 link calls, nine return a root array rather than the required object; thirteen return the legal empty `{"edges":[]}`. Root-array samples additionally contain invalid entity relations or unavailable endpoints, but root schema failure already rejects the whole call. There is no valid subset salvage. All five final graphs have zero edges and all five have zero windows retrieving a remote window. Deterministic nonempty utterance nodes survive invalid ledgers, as the proposal explicitly requires.

| Corpus/video | Ledgers invalid/total | Link root arrays | Valid empty link objects | Graph nodes | Edges / remote windows |
|---|---:|---:|---:|---:|---:|
| HateClipSeg bit_0EHvMSiEHVoc | 29/29 | 4 | 0 | 17 | 0 / 0 |
| HateClipSeg bit_0nXuyV2rypaf | 33/33 | 0 | 5 | 33 | 0 / 0 |
| HateMM hate_video_1 | 12/12 | 0 | 2 | 6 | 0 / 0 |
| HateMM hate_video_114 | 79/79 | 4 | 6 | 77 | 0 / 0 |
| HateMM non_hate_video_4 | 5/5 | 1 | 0 | 1 | 0 / 0 |

The declared per-corpus ledger-plus-remote guard therefore fails. This is a source-interface observation, independent of any metric or paired prediction.

## Input, generation and parser checks

Read the full frozen experiment README, `graph.py`, `inputs.py`, `extract.py`, `src/source_generation.py`, relevant native Judge rendering/encoding/cache methods, CPU renderer, the source-interface summary, and all five `data/temporal_entity_discourse_graph/*/*/metadata.json` source artifacts. The audit uses source metadata only; no GT, metrics or score arrays were opened.

For every ledger, independently re-rendered its system/user input; checked actual image paths and grids, expanded image placeholder token counts and complete input IDs, output-token decoding, token limits, actual forward count and visual-forward count. Reprocessed actual images for the first ledger in each video, five calls total, using the CPU processor. Reprocessed all 22 text-only link inputs. All checks passed. These processor replays use the local CPU environment; they establish consistency with saved IDs/grids, not numerical equivalence of GPU pixel embeddings or logits. The saved source prompt includes the required `utterance.speaker`, entity `quote`, local source IDs and literal window body: those fields were not lost during input serialization.

Re-executed all strict parsers, graph compilation and packet retrieval from saved observations and reproduced the stored parsed objects, graphs and packets exactly. This confirms missing fields were genuinely absent from decoded outputs, rather than dropped by parsing. Saved generation accounting has one multimodal prefill per ledger and one subsequent decoder forward per retained output token; link calls have no visual forward. The manual greedy head uses the declared FP32 projection and EOS termination. No schema-constrained decoding was declared or implemented in interface A.

## Decoding-position issue: actual environment is decisive

`source_generation.generate` calls native `Judge._step` without an explicit `cache_position`. The local **Transformers 4.57.6** Qwen3-VL implementation consequently assigns position zero to a single cached token when its multimodal `rope_deltas` are already set. A CPU probe executing the installed forward-position branch with captured arguments reproduces this on representative saved inputs. This is a real local-version hazard, **not a finding that Slurm 138 used incorrect positions**.

Read and retained the actual `modeling_qwen3_vl.py` files from **uoa-lab2 and uoa-lab3 HateVLM, Python 3.12, Transformers 5.15.1**, plus their package version metadata. In these implementations `compute_3d_position_ids` derives position directly from `past_key_values.get_seq_length()`, then adds the saved multimodal delta. Text-only prefill leaves no multimodal delta; the text decoder derives positions from cache length itself. The omission of `cache_position` therefore does not reset positions on this GPU path.

Executed the actual remote-source position function, extracted by AST, on CPU for both hosts, representative ledger and link prefixes from all five videos, and three consecutive cached steps (60 cases). Multimodal cases advance as cache length plus delta; text cases return the intended text-decoder fallback, whose source uses cache length. This probe loads no model weights and does not claim logits equivalence. Accordingly, retract the tentative local-version suspicion as an explanation of this GPU run. Do not modify frozen native Judge or baseline behavior on its basis.

## Consequence and limits

The evidence supports **interface A structurally unmet on the fixed five**, with no identified correction that would recover its intended interface without a design change. It does not establish why the model omitted fields, that the visual observations were factually empty, or that a persistent multimodal provenance graph cannot work. GPU logits and pixel embedding values were not replayed; absence of an identified bug is limited to the concrete inspected paths and checks above.

A future constrained decoder, grammar, revised schema/prompt, required-field completion, retry or model-output coercion would change interface A. If pursued, label it a newly declared interface B and preserve the original failure record; do not present it as repairing a demonstrated A implementation bug, silently accept invalid calls, or reduce the existing structural guard. This diagnosis does not authorize or implement B, start another run, change the revision budget, or judge subsequent paired/full results.

Evidence: `runs/20261005_m1_provenance/interface_review/audit.py` and `audit.json`; `position_probe.py` and `position_probe.json` (explicitly local 4.57.6 only); `lab2_qwen3_vl.py`, `lab3_qwen3_vl.py`, both package-version text files; `gpu_version_position_probe.py` and its JSON output. All writes were restricted to this report and that evidence directory. No GPU job, GT read, score read, content hash, production edit or external notification was performed.
