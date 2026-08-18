"""Class-name text prompts for the binary hateful-video collapse.

Vendored from https://github.com/lessiYin/DSANet @ eb335b2
(src/utils/descriptions.py). Upstream carries two tables, DESCRIPTIONS_ORI
(14 UCF-Crime classes) and DESCRIPTIONS_ORI_XD (7 XD-Violence classes); both
are kept verbatim below for reference and are unused by this port.

PORT PATCH (patch D3) -- DESCRIPTIONS_HATE is new.

Design decision, the binary collapse
------------------------------------
DSANet's fine-grained branch aligns visual features against one text embedding
per anomaly *category*. The hateful-video benchmarks this port targets are
binary at the video level: HateMM labels a video Hate or Non Hate, and
MultiHateClip's three-way Majority_Voting collapses to Hateful+Offensive versus
Normal (the mapping fixed in CLAUDE.md). There is therefore exactly one
"anomalous" category, and the class-name branch reduces to a two-way contrast.

The two prompts are "normal content" and "hateful content". Three constraints
pin the choice down:

1. Slot 0 must be the normal class. CLAS2 reads labels[:, 0] as the normal
   column, CLASM_BKG pins its target at column 0, the orthogonality term
   treats text_features[0] as the normal embedding, and inference reports
   1 - softmax(...)[:, 0]. Reordering the dict silently inverts every one of
   those. Python dicts preserve insertion order, so the order below is load
   bearing.

2. One string per class, no ensembling. Upstream's tables are already one
   description per class (the mean over `descriptions` is a mean over a
   single element), so a single prompt keeps the port faithful and avoids
   turning the text branch into a prompt ensemble.

3. Category words, not instructions. Upstream prompts are bare category names
   ("fighting", "explosion", "normal"), fed through CLIP's text encoder as
   nouns. "hateful content" / "normal content" keeps that register and keeps
   the two strings minimally different, so the contrast the alignment loss
   sees is the hateful/normal axis rather than a difference in phrasing.

Consequence worth stating up front: with two classes DSANet's test-time
`refine_scores_hierarchical` degenerates. It renormalises the non-normal
columns of the alignment softmax, and with a single non-normal column that
renormalisation returns 1, so its output equals sigmoid(logits1). See
scripts/reproduction_baselines/dsanet/infer.py, which therefore emits the
alignment score by VadCLIP's own formula as well.
"""

DESCRIPTIONS_HATE = {
    "normal": [
        "normal content"
    ],
    "hateful": [
        "hateful content"
    ],
}

# ---------------------------------------------------------------- upstream
# Kept verbatim from DSANet @ eb335b2 for provenance. Unused by this port.

DESCRIPTIONS_ORI = {
    "normal": ["normal"],
    "abuse": ["abuse"],
    "arrest": ["arrest"],
    "arson": ["arson"],
    "assault": ["assault"],
    "burglary": ["burglary"],
    "explosion": ["explosion"],
    "fighting": ["fighting"],
    "roadaccidents": ["roadaccidents"],
    "robbery": ["robbery"],
    "shooting": ["shooting"],
    "shoplifting": ["shoplifting"],
    "stealing": ["stealing"],
    "vandalism": ["vandalism"],
}

DESCRIPTIONS_ORI_XD = {
    "A": ["normal"],
    "B1": ["fighting"],
    "B2": ["shooting"],
    "B4": ["riot"],
    "B5": ["abuse"],
    "B6": ["car accident"],
    "G": ["explosion"],
}
