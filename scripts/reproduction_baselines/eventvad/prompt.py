#!/usr/bin/env python
"""The event-scoring prompt, reconstructed, and the score parser.

Upstream `src/score/event_score.py:23` reads

    abnormal_prompt = "prompt"

a literal placeholder. Nothing in the release records the prompt the paper's
numbers were produced with, so it is reconstructed from the paper. Everything
quoted below is transcribed from arXiv:2504.13092; the reconstruction and the
inferences it rests on are argued in DESIGN_EVENTVAD.md, gap G2.

Figure 2, "Event-Centric Anomaly Scoring", is the only place the prompt is
shown. It gives the input in two tagged lines and the output in a third:

    #Question: Are there any obvious or potential anomalies in the video?

    #Instruction: Let's think step by step to judge the anomaly of the video.
    Finally, output the anomaly score based on the thinking process.

    #Answer: A man is seen with a gun, pointing it at the employees. He takes
    money from the register and ... Therefore, the final score is 0.8.

Section 3.4 says what the two input lines are for:

    "we propose a semantic-driven hierarchical prompting framework. This
    framework directs multimodal large language models to produce structured
    outputs: first generating video content descriptions and then deriving
    anomaly scores. When processing a video, the multimodal large language
    models initially generate descriptive text by identifying surface and
    latent semantic features. It subsequently outputs an anomaly score based
    on this description, enabling cross-modal evaluation against predefined
    criteria."

and section 4.2 insists the whole of it is those two lines:

    "compared to LAVAD, EventVAD's prompt setup is very straightforward. We
    achieve multi-stage reasoning in MLLM through hierarchical prompting to
    enhance its scene-understanding capability."

Table 5's third ablation column is the `#Instruction` line. Section 4.3 names
it "the implementation of deliberative reasoning in MLLM outputs before
anomaly scoring" and explains:

    "the quantitative experiments show the structured MLLM output can be
    specified, so that it can summarize the video content before outputting
    the abnormal score, and this process of letting the model think can better
    help him understand these small fragments and give a more reasonable
    score."

so `#Question` alone is the "Thinking = no" row and `#Question` +
`#Instruction` is the "Thinking = yes" row that produced the headline 82.03.

One call per event, single pass, greedy -- upstream passes `do_sample=False`.
"Hierarchical" names the two roles inside the one answer, describe then score;
it is not two queries.
"""

from __future__ import annotations

import re

# ------------------------------------------------------- verbatim from Fig 2
PAPER_QUESTION = ("Are there any obvious or potential anomalies "
                  "in the video?")
PAPER_INSTRUCTION = ("Let's think step by step to judge the anomaly of the "
                     "video. Finally, output the anomaly score based on the "
                     "thinking process.")

# The tag layout is Figure 2's. `#Answer:` is the model's turn, so it is not
# part of the input; `mm_infer` appends the chat template's own generation
# prompt after this text.
_PAPER = "#Question: %s\n#Instruction: %s" % (PAPER_QUESTION, PAPER_INSTRUCTION)

# Table 5, Thinking = no. The `#Instruction` line is the ablated component, so
# it is removed and nothing is put in its place beyond the demand for a score
# that the removed line was also carrying. Without those five words the model
# is not asked for a number at all and the row would ablate two things.
_NO_THINKING = ("#Question: %s\n#Instruction: Output the anomaly score."
                % PAPER_QUESTION)

# The paper's "predefined criteria" are never stated and Figure 2's only
# example score is 0.8, so the range is an inference (DESIGN G2-c). This arm
# makes it explicit; it is not the default, because the sentence is not in the
# paper.
_BOUNDED = _PAPER + ("\nThe anomaly score is a single real number between "
                     "0 and 1, where 0 is completely normal and 1 is "
                     "certainly anomalous.")

ARMS = {
    "paper": _PAPER,
    "no_thinking": _NO_THINKING,
    "bounded": _BOUNDED,
}

DEFAULT_ARM = "paper"


def build_prompt(arm=DEFAULT_ARM):
    if arm not in ARMS:
        raise ValueError("unknown arm %r (expected one of %s)"
                         % (arm, ", ".join(sorted(ARMS))))
    return ARMS[arm]


# ------------------------------------------------------------------ parsing
_NUM = r"[-+]?\d*\.?\d+"
_PATTERNS = (
    # Figure 2's own closing sentence, and the phrasings around it.
    re.compile(r"final\s+(?:anomaly\s+)?score\s+is\s*[:=]?\s*(%s)" % _NUM,
               re.I),
    re.compile(r"(?:anomaly\s+)?score\s+is\s*[:=]?\s*(%s)" % _NUM, re.I),
    re.compile(r"(?:final\s+)?(?:anomaly\s+)?score\s*[:=]\s*(%s)" % _NUM,
               re.I),
)
_TRAILING = re.compile(r"(%s)\s*[.\s]*$" % _NUM)


def parse_score(text):
    """Return (score_or_None, status, raw_number_or_None).

    Upstream's parser is `float(output.strip())`, which cannot read the answer
    Figure 2 shows -- prose ending "Therefore, the final score is 0.8." That
    fast path is tried first anyway, so an output upstream could have read is
    read identically here, and the sentence patterns follow.

    The number is returned as written. Range handling is the caller's, so that
    a run can report how often it fired.
    """
    if text is None:
        return None, "empty", None
    text = text.strip()
    if not text:
        return None, "empty", None

    try:                                    # upstream's fast path
        return float(text), "bare_float", float(text)
    except ValueError:
        pass

    for pat in _PATTERNS:
        hits = pat.findall(text)
        if hits:
            try:
                value = float(hits[-1])
            except ValueError:
                continue
            return value, "sentence", value

    hit = _TRAILING.search(text)
    if hit:
        try:
            value = float(hit.group(1))
        except ValueError:
            return None, "unparsed", None
        return value, "trailing_number", value

    return None, "unparsed", None


def normalise_score(value):
    """Map a parsed number onto [0, 1] and say which rule was used.

    Figure 2's example is 0.8 and the paper never states a range, so a model
    that answers on a 0-10 or 0-100 scale is answering a question the paper
    did not pin down. Rescaling by the smallest containing decade keeps such
    an answer comparable with the in-range ones; pooled ROC-AUC is computed
    across videos, so leaving an 8 beside a 0.8 would corrupt the ranking
    rather than preserve it. Every rescale is counted in the run report.
    """
    if value is None:
        return None, "none"
    if value != value or value in (float("inf"), float("-inf")):
        return None, "nonfinite"
    if 0.0 <= value <= 1.0:
        return value, "in_range"
    if 0.0 <= value <= 10.0:
        return value / 10.0, "div10"
    if 0.0 <= value <= 100.0:
        return value / 100.0, "div100"
    if value < 0.0:
        return 0.0, "clamped_low"
    return 1.0, "clamped_high"
