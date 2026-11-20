"""Freeze cross-camera TUM development candidates from public pose texts."""

import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path


SOURCES = {
    "freiburg1_plant": (281995, "db4b4bbea6e3a64fdcd2c161746cf39fce6076bc678cd379364ca361a457e6f7"),
    "freiburg2_dishes": (2058671, "649ec02f2deb8fa5736560e419105cb946b9025c541f9227d34ffb8d6c5e16de"),
}
WINDOWS = 12
GAPS = (0.15, 0.30, 0.45)
MAX_TIMESTAMP_ERROR = 0.02
MIN_TRANSLATION = 0.02


def parse_trajectory(raw):
    rows = []
    for line in raw.decode("utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        values = tuple(map(float, text.split()))
        if len(values) != 8 or not all(map(math.isfinite, values)):
            raise ValueError("invalid TUM pose row")
        if not 0.999 <= math.hypot(*values[4:]) <= 1.001:
            raise ValueError("invalid mocap quaternion")
        if rows and values[0] < rows[-1][0]:
            raise ValueError("TUM timestamps are not chronological")
        rows.append(values)
    accepted = []
    duplicate_identical, duplicate_conflicting = 0, 0
    for _, group in itertools.groupby(rows, key=lambda row: row[0]):
        group = list(group)
        if len(set(group)) == 1:
            accepted.append(group[0])
            duplicate_identical += len(group) - 1
        else:
            duplicate_conflicting += len(group)
    if len(accepted) < 2:
        raise ValueError("too few unambiguous poses")
    return accepted, dict(raw_pose_rows=len(rows), unambiguous_pose_rows=len(accepted),
                          identical_duplicate_rows_collapsed=duplicate_identical,
                          conflicting_timestamp_rows_excluded=duplicate_conflicting)


def nearest(rows, times, wanted):
    index = bisect_left(times, wanted)
    positions = range(max(0, index - 1), min(len(rows), index + 1))
    return rows[min(positions, key=lambda x: (abs(wanted - times[x]), x))]


def pose_angles(left, right):
    a, b = left[4:], right[4:]
    dot = abs(sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b)))
    return math.degrees(2 * math.acos(min(1.0, dot)))


def plan(rows):
    times = [row[0] for row in rows]
    width = (times[-1] - times[0]) / WINDOWS
    if width <= 2 * max(GAPS):
        raise ValueError("trajectory does not cover twelve distinct temporal windows")
    result, seen = [], set()
    for index in range(WINDOWS):
        center = times[0] + (index + 0.5) * width
        gap = GAPS[index % len(GAPS)]
        targets = (center - gap / 2, center + gap / 2)
        left, right = [nearest(rows, times, time) for time in targets]
        error = max(abs(left[0] - targets[0]), abs(right[0] - targets[1]))
        translation = math.dist(left[1:4], right[1:4])
        geometry = dict(max_pose_time_error_seconds=error, translation_m=translation,
                        relative_rotation_degrees=pose_angles(left, right))
        if error > MAX_TIMESTAMP_ERROR:
            state = "POSE_TIMING_GAP"
        elif left[0] >= right[0] or left[0] in seen or right[0] in seen:
            state = "REPEATED_POSE"
        elif translation < MIN_TRANSLATION:
            state = "LOW_TRANSLATION"
        else:
            state = "POSE_PAIR_CANDIDATE"
            seen.update((left[0], right[0]))
        result.append(dict(window=index + 1, requested_gap_seconds=gap, requested_times=targets,
                           original_pose_times=(left[0], right[0]), state=state, **geometry))
    return result


def freeze(source, output):
    result = {}
    private = {}
    for sequence, (size, expected) in SOURCES.items():
        path = source / f"{sequence}.txt"
        if path.is_symlink() or not path.is_file():
            raise ValueError("expected original trajectory text")
        with path.open("rb") as stream:
            raw = stream.read(size + 1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("original public trajectory identity differs")
        rows, report = parse_trajectory(raw)
        candidates = plan(rows)
        private[sequence] = candidates
        result[sequence] = dict(official_pose_bytes=size, official_pose_SHA256=expected,
                                duration_seconds=rows[-1][0] - rows[0][0],
                                pair_status_counts=dict(Counter(x["state"] for x in candidates)), **report)
    payload = (json.dumps(private, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(payload)
    return dict(sequences=result, planned_pose_pairs=WINDOWS * len(SOURCES),
                private_anchor_plan_SHA256=hashlib.sha256(payload).hexdigest(), private_anchor_plan_bytes=len(payload),
                actual_RGB_reads=0, actual_predictions=0, scored_effect_cases=0, formal_CO3D=False,
                previous_household_blind_data_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
