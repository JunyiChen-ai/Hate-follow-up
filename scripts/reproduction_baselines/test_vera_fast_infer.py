#!/usr/bin/env python3
"""CPU/static checks for the independent accelerated VERA runner."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import vera_adapter as legacy
import vera_fast_infer as fast


class FastInferTests(unittest.TestCase):
    def test_indices_equal_frozen_formula(self):
        for fps in (23.976, 25.0, 29.97):
            for duration in (1.2, 17.3, 283.64):
                frame_count = max(1, round(duration * fps))
                duration = frame_count / fps
                for start in (0.0, 1.0, max(0.0, duration - 0.3)):
                    end = min(duration, start + 10.0)
                    times = np.linspace(start,
                                        max(start, end - 1 / max(fps, 1)), 8)
                    expected = np.clip(np.rint(times * fps).astype(int),
                                       0, frame_count - 1)
                    np.testing.assert_array_equal(
                        fast.frame_indices(start, 10.0, 8, fps, duration,
                                           frame_count), expected)

    def test_batch_one_delegates_to_frozen_predict_in_order(self):
        calls = []
        def fake(_model, _tok, images, _prompts):
            calls.append(images)
            return len(images), str(images)
        batches = [[1], [2, 3], [4]]
        with mock.patch.object(legacy, "predict", side_effect=fake):
            got = fast.predict_batch(None, None, batches, [], 1)
        self.assertEqual(calls, batches)
        self.assertEqual(got, [(1, "[1]"), (2, "[2, 3]"), (1, "[4]")])

    def test_frozen_completion_validator_accepts_atomic_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v.json"
            payload = {"video_id": "v", "duration": 2.0, "segments": [
                {"start": 0.0, "end": 2.0, "score": 1,
                 "response": "Output: 1"}]}
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload))
            temporary.replace(path)
            self.assertTrue(legacy.valid_raw_result(path, "v", 1))

    def test_batch_one_uses_frozen_question_and_parser(self):
        self.assertIn("Frame8: <image>", legacy.question(["guide"], 8))
        with mock.patch.object(legacy, "predict", return_value=(0, "Output: 0")):
            self.assertEqual(
                fast.predict_batch(object(), object(), [[object()] * 8],
                                   ["guide"], 1),
                [(0, "Output: 0")])


if __name__ == "__main__":
    unittest.main()
