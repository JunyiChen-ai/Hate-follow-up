# Draft — code/protocol request to MultiHateLoc authors (owner sends)

To: Zeyu Fu (University of Exeter; corresponding author per the paper)
Cc: co-authors as listed on arXiv 2512.10408
Subject: MultiHateLoc (WWW '26) — code and evaluation protocol request

Dear Dr. Fu,

I am a researcher working on hateful-video temporal localization, and
we are benchmarking against MultiHateLoc ("Towards Temporal
Localisation of Multimodal Hate Content in Online Videos", WWW 2026)
as the closest published system to our work.

The paper states "Code is available at
https://github.com/mmilabuk/multihateloc", but the repository
currently contains only a LICENSE file. Could you share the
implementation, or failing that, answer three protocol questions that
the paper does not specify?

1. The frame grid: how many frames per video (or what fps) define T,
   both for the model input and for the frame-level mAP/AUC
   evaluation?
2. The train/test split used for the HateMM and MultiHateClip results
   (Table 1), and whether non-hateful videos' frames enter the
   evaluation as negatives.
3. The rule converting the datasets' second-resolution span
   annotations into per-frame ground-truth labels (boundary handling,
   and the treatment of hateful videos that carry no span annotation).

We have reimplemented the method from the paper under a fully
specified protocol (1 fps grid, documented span-to-frame rule,
published ground-truth arrays) and will clearly label our numbers as
a reimplementation under stated assumptions; access to your code
would let us report your method at its best instead.

Thank you for your time.

Best regards,
Junyi Chen
University of Auckland
jehc223@aucklanduni.ac.nz
