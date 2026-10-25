# Rule-4 proposal review: candidate 51 (agent-retrieved knowledge for the verdict and window reads)

2026-10-26. Independent reviewer, one pass, no GPU. Read: rule 4, the candidate README, the verdict-oracle README,
`research-wiki/DIRECTIONS.md`, the closed-family rows of `docs/reviews/20261025_m1m3_team/history.md`, the CHER full
text, and the released code of the WWW Companion '26 hate-video agent (Zenodo 18208988; read, not run).

## Verdict: PASS

- **Source method already used for hateful video detection / localization?** No. CHER (Ailneni and Harabagiu, EMNLP
  2026) was checked in full text: a ReAct agent infers entities, groups, symbols and social domain; it queries the
  Wikipedia API (5 summaries per query, first five lines kept) plus WVS, Cultural Atlas and minority-group profiles;
  it filters by two self-reflection criteria; then HDE identification and counterfactual contrast. Memes only; its
  limitations name video as future work. No hateful-video paper found does LLM-driven clue extraction, Wikipedia
  retrieval and a reflection filter. Generic external-context retrieval exists for hateful video (Yadav and Singh
  below); it narrows the claim but is not the CHER pipeline, so no STOP.
- **Pure ensemble?** No for arms K / Kp: the API model supplies facts, and the 8B reader decides. Arms J / KJ put a
  second model's decision and confidence into the reader. That is close to the rule-3 ban on combining two models'
  decisions, but it is not a pure ensemble. The user allowed it on 2026-10-26. See advisory 1.
- **Pure calibration / post-processing / smoothing?** No.
- **Pure engineering trick?** No. It adds a new information source through a declared pipeline (clues, retrieval,
  filter, time-anchored insertion). Under rule 5, the new input is not novelty by itself.

## Closest prior work

1. **CHER** (EMNLP 2026, https://aclanthology.org/2026.emnlp-main.2538/): the source. One GPT-5 model retrieves and
   decides, one decision per meme, no time dimension.
2. **Yadav and Singh, "An Interpretable Agentic Framework for Multimodal Hate Video Analysis with Explicit Evidence
   Attribution"** (WWW Companion 2026, doi 10.1145/3774905.3796488). Its released code has a `RetrievalAgent`
   labelled "external context fetching (OSINT)". The agent builds a query from OCR, ASR and named entities. It runs
   BM25 recall plus FAISS reranking over an offline corpus of policy documents, hate-symbol references and
   hate-word lexicons. The top 5 untimed snippets go to a rule-weighted ranker or an LLM orchestrator for a video
   label. The query is not written by an LLM, there is no reflection filter, no Wikipedia and no binding to time.
3. **IARE** (arXiv 2606.11953; ACM doi 10.1145/3805712.3809637). At inference, GPT-4o-mini writes captions and
   harmful elements, and these are inserted as context for a LoRA-trained Qwen2.5-VL-7B / GLM-4.1V. So "a stronger
   model writes context for a smaller hateful-video reader" already exists, though with a trained reader, no
   retrieval and video level only.
4. Other hateful-video retrieval fetches examples, not knowledge: MoRE (WWW 2025, doi 10.1145/3696410.3714560,
   similar videos, trained); HATELENS (Findings EMNLP 2026, 2026.findings-emnlp.1336, annotated precedents).

## What the candidate must claim

Allowed claims, in narrowing order:
- (a) Label-free, CHER-style agentic knowledge acquisition (LLM-written queries, Wikipedia, reflection filter)
  carried over to video.
- (b) Retrieved knowledge **anchored in time and bound to the window questions** for frame-level localization. No
  prior work found does this.
- (c) A frozen small reader consumes a stronger agent's knowledge without training.

Not allowed:
- "First external knowledge for hateful video" (ruled out by Yadav and Singh).
- "First stronger-model context for a small hate-video reader" (ruled out by IARE; KARD and GKP in general NLP).
- The judgment arms as a knowledge result.

If only Kp passes (the verdict prefix), the claim narrows to (a) at video level. Then the distance from Yadav and
Singh is small: LLM-written queries plus reflection versus BM25 over a curated corpus.

## Advisories (each changes what the result can show)

1. **Missing control for model strength (J / KJ).**
   - Add a CPU-only arm that uses qwen3-vl-plus's own judgment (`violates` × `confidence`) as z_video, decoded with
     r6.
   - Add a J0 arm: a judgment written without the notes (one extra call per video).
   - Without these, a J / KJ gain cannot be split between knowledge and a better judge. If plus-alone ≥ KJ, the 8B
     adds nothing. Rule 14(f) needs this number too.
2. **Window-insert controls for the distinguishing claim (b).**
   - Kother changes both the content and the times, so it does not test whether the time binding matters.
   - Add Kshift: the video's own notes at wrong windows (fixed circular shift by half the video).
   - Add Kmark: an insert of equal length with no content at the noted windows.
   - Reasons: (i) the StreamingTOM placement controls (history.md C39) showed that re-showing material next to the
     question alone raises within; (ii) note times are a localization signal from the stronger model, not knowledge.
3. **The gate is too weak on HateClipSeg.**
   - The verdict oracle gives 10 false positives in 118 HCS test videos. That is about 3 in a 40-video subset, so
     "≥ 40 % flipped" comes down to 2 videos. Pooled ROC / PR on 40 videos has noise well above .005.
   - Running all 118 HCS test videos costs about ¥8 by the README's own price.
   - Count new false positives as well as new false negatives. Knowledge about slurs and events can push topic-only
     videos to Yes (DIRECTIONS D5).
4. **Within is ungated.**
   - G1 uses only pooled metrics and flip counts. The verdict cannot move within (oracle), so claim (b) rests on K
     versus Kp within, which is not gated.
   - Report subset within with video counts: about 16 HateMM and 34 HCS videos with both classes.
5. Record the API model version string and the Wikipedia retrieval date. Keep the notes as text in `runs/`, since the
   knowledge pack is not reproducible otherwise.

## Searches run (31; plus fetches of the CHER PDF, HATELENS, IARE, LELA, SafeLens, Zenodo)

CHER memes; hateful video + Wikipedia / knowledge / web search / RAG agent (6 variants, 3 extended); MIND; Pro-Cap;
MoRE; ProKDA; LELA; meme knowledge graphs; MAESTRO; LoReHM; Mr.Harm; ReAct meme cultural knowledge; implicit-hate
world knowledge; VAD knowledge / RAG (2); KARD; large-model hints; fake-news video evidence; MATCH; WWW'26 agent (2);
RA-HMD; GKP; RGCL; Video-RAG; time-anchored knowledge localization.

## Papers checked

- Hateful video: Yadav and Singh; IARE; MoRE; HATELENS; LELA 2602.09637 (no retrieval); SafeLens AAAI '26 demo
  (segment-level, tuned policy LLM, no retrieval stated); MARS 2601.15115; RAMF 2512.02743; HVGuard EMNLP '25; LEAF
  Findings ACL '26; CLARA 2608.15905; MultiHateLoc 2512.10408; MAESTRO (no retrieval found); MATCH (in-video
  evidence).
- Memes / text hate: ProKDA 2609.19778 (agentic Wikipedia queries built using the label, trained);
  MemeGraphs 2305.18391 (Wikidata); MIND ACL '25 2507.06908; LoReHM EMNLP '24 2411.05383; Pro-Cap 2308.08088; RGCL
  ACL '24 2311.08110; RA-HMD EMNLP '25 2502.13061; Mr.Harm Findings EMNLP '23 2312.05434; Lin NLP4PI '22 2212.14100
  (Wikipedia entity linking); LLM-generated context 2510.15685.
- Strong-to-small context: GKP ACL '22 2110.08387; KARD NeurIPS '23 2305.18395; Li et al. 2210.06726; Speculative
  Thinking 2504.12329.
- Video anomaly / misinformation: LAVAD CVPR '24 2404.01014; AnomalyRuler 2407.10299; SlowFastVAD 2025
  (retrieval-augmented VLM, arXiv id not verified); CSI Findings ACL '26 (external knowledge, fake-news video);
  FactGuard 2602.22963; Video-RAG NeurIPS '25 2411.13093 (in-video text only).

Not finding a paper does not prove none exists. The WWW Companion '26 full text was not accessible; only its code
was read.
