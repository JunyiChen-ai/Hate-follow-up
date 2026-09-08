#!/usr/bin/env python3
"""Frozen registry for the twelve paper-mechanism adaptations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class AdapterSpec:
    adapter_id: str
    name: str
    venue: str
    mechanism: str
    implementation: str
    supervision: str
    asset_policy: str


SPECS = (
    AdapterSpec("A01", "NumPro", "CVPR 2025", "numbered-frame time interface",
                "training_free", "external_pretraining_allowed", "prompt transplant"),
    AdapterSpec("A02", "VTimeCoT", "ICCV 2025", "visual timeline tool reasoning",
                "training_free", "external_pretraining_allowed", "prompt/tool transplant"),
    AdapterSpec("A03", "TemporalConsistency", "CVPR 2025", "perturbation verification",
                "training_free", "external_pretraining_allowed", "test-time objective transplant"),
    AdapterSpec("A04", "OmniVTG", "CVPR 2026", "predict-reflect-revise grounding",
                "training_free", "external_pretraining_allowed", "inference transplant"),
    AdapterSpec("A05", "MUSEG", "ACL 2026", "timestamp-aware multi-segment grounding",
                "checkpoint_or_training_free", "external_temporal_pretraining_allowed", "checkpoint/mechanism"),
    AdapterSpec("A06", "DisTime", "ICCV 2025", "distributional boundary representation",
                "checkpoint_or_sampling", "external_temporal_pretraining_allowed", "checkpoint/mechanism"),
    AdapterSpec("A07", "ED-VTG", "ICCV 2025", "query enrichment and detection",
                "checkpoint_or_training_free", "external_temporal_pretraining_allowed", "checkpoint/mechanism"),
    AdapterSpec("A08", "TGB", "EMNLP 2024", "low-dimensional motion bridge",
                "checkpoint", "external_temporal_pretraining_allowed", "official adapter preferred"),
    AdapterSpec("A09", "GroundingGPT", "ACL 2024", "image/video/audio grounding",
                "checkpoint", "external_temporal_pretraining_allowed", "official checkpoint required"),
    AdapterSpec("A10", "Vid-Group", "ICCV 2025", "unlabelled proposal pretraining",
                "checkpoint", "unlabelled_external_pretraining", "official checkpoint required"),
    AdapterSpec("A11", "Seq2Time", "CVPR 2025", "relative sequence-time tokens",
                "checkpoint", "self_supervised_external_pretraining", "official checkpoint preferred"),
    AdapterSpec("A12", "TimeLens", "CVPR 2026", "interleaved time encoding and RLVR",
                "checkpoint", "external_temporal_pretraining_allowed", "official checkpoint required"),
)


def main() -> int:
    print(json.dumps([asdict(x) for x in SPECS], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

