#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import vera_sparse_k16 as producer


class SparseProducerTests(unittest.TestCase):
    def test_index_rule(self):
        np.testing.assert_array_equal(producer.sparse_indices(1), [0])
        np.testing.assert_array_equal(producer.sparse_indices(4), [0, 1, 2, 3])
        expected = np.unique(np.rint(np.linspace(0, 226, 16)).astype(int))
        np.testing.assert_array_equal(producer.sparse_indices(227), expected)

    def test_synthetic_manifest_passes_v6_preflight(self):
        from relation_v6.audited_data import verify_sparse_manifest, VGG_INDEX
        from relation_v2.protocol import frozen_splits
        timeline = json.loads(Path(VGG_INDEX).read_text())
        video_id = frozen_splits("hateclipseg")["train"][0]
        n_frames = int(timeline[video_id]["n_frames"])
        starts = producer.expected_starts(video_id, timeline)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = directory / "raw"; raw.mkdir()
            duration = n_frames - 0.2
            segments = [{"start": float(start),
                         "end": min(duration, float(start) + 10.0),
                         "score": int(i % 2), "response": f"Output: {i % 2}"}
                        for i, start in enumerate(starts)]
            (raw / f"{video_id}.json").write_text(json.dumps({
                "video_id": video_id, "duration": duration,
                "segments": segments}))
            selection = {"prompts": ["fixed"], "attention_backend": "test",
                         "backbone": "test"}
            manifest = directory / "manifest.json"
            producer.build_manifest(raw, manifest, [video_id], selection, timeline)
            info = verify_sparse_manifest(
                str(manifest), frozen_splits("hateclipseg")["train"])
            self.assertEqual(info["cohort"], [video_id])
            self.assertTrue(Path(info["root"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
