"""Actual PTS sampling; ownership follows observations instead of targets."""
import math
import time
from pathlib import Path
import numpy as np
from PIL import Image
from retrieval import ROOT, SPEC
from src.actual_video_frames import resolve_video


def targets_for(duration):
    assert duration > 0
    return [i / SPEC['source_fps'] for i in range(math.ceil(duration * SPEC['source_fps']))]


def select(entries, duration):
    """Return the first actual frame at/after each target, or missing."""
    targets = targets_for(duration)
    cursor = 0
    result = []
    for target in targets:
        while cursor < len(entries) and entries[cursor]['time'] < target:
            cursor += 1
        result.append(entries[cursor]['index'] if cursor < len(entries) else None)
    return result


def decode_entries(row):
    import av
    source = resolve_video(row)
    duration = float(row['duration'])
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        origin = float(container.start_time / av.time_base) if container.start_time is not None else None
        previous = -math.inf
        for index, frame in enumerate(container.decode(stream)):
            assert frame.pts is not None, 'decoded source frame has no PTS'
            absolute = float(frame.pts * frame.time_base)
            if origin is None:
                origin = absolute
            actual_time = absolute - origin
            assert actual_time > previous, 'source video PTS is not monotone'
            previous = actual_time
            entry = dict(index=index,pts=int(frame.pts),time=actual_time,absolute=absolute,
                time_base=[frame.time_base.numerator,frame.time_base.denominator],
                width=frame.width,height=frame.height)
            yield frame, entry, origin, 0 <= actual_time < duration


def acquire(row, folder):
    start = time.perf_counter()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    duration = float(row['duration'])
    targets = targets_for(duration)
    cursor = 0
    entries, selected = [], []
    saved = set()
    origin = None
    decoded = 0
    for frame, entry, origin, available in decode_entries(row):
        decoded += 1
        if not available:
            continue
        entries.append(entry)
        while cursor < len(targets) and entry['time'] >= targets[cursor]:
            selected.append(entry['index'])
            cursor += 1
            if entry['index'] not in saved:
                image = frame.to_image().convert('RGB')
                try:
                    image.save(folder / f'frame_{entry["index"]:08d}.png')
                finally:
                    image.close()
                saved.add(entry['index'])
    selected.extend([None] * (len(targets) - cursor))
    assert selected == select(entries, duration)
    result = dict(input_video=str(resolve_video(row)),manifest_video_path=row['video_path'],
        duration=duration,origin=origin,decoded_frames=decoded,entries=entries,
        source_fps=SPEC['source_fps'],targets=targets,target_indices=selected,
        selected_indices=sorted(saved),uncovered_targets=[i for i, value in enumerate(selected) if value is None],
        decode_seconds=time.perf_counter()-start)
    validate(result,row,folder)
    result['decode_seconds'] = time.perf_counter()-start
    return result


def validate(meta, row, folder):
    assert meta['manifest_video_path'] == row['video_path'] and meta['duration'] == float(row['duration'])
    assert Path(meta['input_video']).stem == row['video_id']
    assert meta['source_fps'] == SPEC['source_fps'] and meta['targets'] == targets_for(meta['duration'])
    assert meta['target_indices'] == select(meta['entries'], meta['duration'])
    assert meta['selected_indices'] == sorted(set(i for i in meta['target_indices'] if i is not None))
    assert meta['uncovered_targets'] == [i for i, value in enumerate(meta['target_indices']) if value is None]
    current, decoded, verified = [], 0, set()
    origin = None
    wanted = set(meta['selected_indices'])
    for frame, entry, origin, available in decode_entries(row):
        decoded += 1
        if available:
            current.append(entry)
        if entry['index'] not in wanted:
            continue
        actual = frame.to_image().convert('RGB')
        try:
            with Image.open(Path(folder) / f'frame_{entry["index"]:08d}.png') as image:
                assert image.format == 'PNG' and image.size == (entry['width'], entry['height'])
                assert np.array_equal(np.asarray(actual), np.asarray(image.convert('RGB'))), 'actual source pixels differ'
        finally:
            actual.close()
        verified.add(entry['index'])
    assert verified == wanted and current == meta['entries']
    assert decoded == meta['decoded_frames'] and origin == meta['origin']
    assert meta['decode_seconds'] >= 0


def owned_source_ids(meta, window):
    a, b = window
    entries = {e['index']: e for e in meta['entries']}
    return [i for i, source_index in enumerate(meta['selected_indices']) if a <= entries[source_index]['time'] < b]
