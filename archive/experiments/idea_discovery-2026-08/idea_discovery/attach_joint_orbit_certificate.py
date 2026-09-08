#!/usr/bin/env python3
"""Attach an exact joint-orbit certificate as a separate prediction channel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.idea_discovery.project_role_orthogonal_transport import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--prediction-method", required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--certificate-method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    predictions = load(args.prediction, args.prediction_method)
    certificates = load(args.certificate, args.certificate_method)
    with args.out.open("w") as handle:
        for key in sorted(set(predictions) & set(certificates)):
            output = dict(predictions[key])
            certificate_evidence = certificates[key].get("modality_evidence", {})
            output["method"] = "dual_channel_joint_orbit_transport_v1"
            output["modality_evidence"] = {
                **output.get("modality_evidence", {}),
                "joint_adjusted_p": certificate_evidence["joint_adjusted_p"],
                "joint_evalue": certificate_evidence["joint_evalue"],
            }
            output["raw"] = {
                **output.get("raw", {}),
                "gt_access": False,
                "module": "dual_channel_joint_orbit_transport",
                "prediction_channel": args.prediction_method,
                "certificate_channel": args.certificate_method,
                "separation_principle": "certificate_is_not_reinterpreted_as_posterior",
                "certificate_forwarded_unchanged": True,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps({"n": len(set(predictions) & set(certificates))}, indent=2))


if __name__ == "__main__":
    main()
