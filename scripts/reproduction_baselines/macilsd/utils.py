# -*- coding: utf-8 -*-
"""Sequence bookkeeping, from MACIL_SD @ c20943f utils.py.

`random_extract`, `uniform_extract`, `pad`, `process_feat`,
`process_test_feat` and `cosine_scheduler` are byte-identical to upstream.

PORT PATCH (patch M2): upstream's `Prepare_logger` is dropped. It opens a file
handler under a relative `log/` directory at import-adjacent call time, which
fails unless the process happens to be cwd'd into the clone, and this port logs
one summary line per epoch to stdout instead (patch M8). Nothing else in the
module changes.

A note on the temporal unit, because it is the whole reason MACIL-SD ports
cleanly here. Upstream `process_feat(..., length=200)` counts *I3D snippets*:
XD-Violence video is 24 fps and a snippet is 16 frames, so 200 rows is 133.3 s.
This study's I3D features were extracted at the same 24 fps with the same
16-frame snippet (`.times.json` records `decode_fps: 24, snippet_frames: 16`),
so one row here is 0.666667 s exactly as upstream. `--max-seqlen 200` therefore
means the same physical window it meant on XD-Violence and is kept verbatim.
"""

import math

import numpy as np


def random_extract(feat, t_max):
    r = np.random.randint(len(feat) - t_max)
    return feat[r:r + t_max]


def uniform_extract(feat, t_max):
    r = np.linspace(0, len(feat) - 1, t_max, dtype=np.uint16)
    return feat[r, :]


def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
        return np.pad(feat, ((0, min_len - np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat


def process_feat(feat, length, is_random=True):
    if len(feat) > length:
        if is_random:
            return random_extract(feat, length)
        else:
            return uniform_extract(feat, length)
    else:
        return pad(feat, length)


def process_test_feat(feat, length):
    tem_len = len(feat)
    num = math.ceil(tem_len / length)
    if len(feat) < length:
        return pad(feat, length)
    else:
        return pad(feat, num * length)


def cosine_scheduler(base_value, final_value, curr_epoch, epochs):
    value = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * curr_epoch / epochs))
    return value
