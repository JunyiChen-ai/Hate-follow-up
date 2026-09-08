# Sealed Protocol Deviation — 2026-08-28

During implementation of the post-inference GT builder, the first 1000 bytes of
`/home/jehc223/Retrieval-hate/data/gt/HateClipSeg/gold_segments.json` were
printed with `head -c 1000` before A12 inference completed.  This violated the
literal preregistered instruction not to open the gold source until all frozen
predictions were complete.

The printed prefix contained only the beginning of `bit_0EHvMSiEHVoc`, an ID in
the already-developed p11 test cohort, not either of the first displayed sealed
manifest IDs (`bit_0SYLs1h6WtM2`, `bit_0ZrexCZJ860o`).  No sealed-cohort ID was
looked up, joined, scored, summarized, or used to change VASTA predictions or
parameters.  A12 continued under the preregistered frozen configuration.

Consequences:

- The numerical evaluation can still be reported as prediction-frozen and with
  no observed sealed label used for model selection.
- The run must not be described as perfectly process-sealed without disclosing
  this deviation.
- GT construction and all metric computation remain prohibited until A12 and
  the prediction-only factorial file are complete and hashed.
