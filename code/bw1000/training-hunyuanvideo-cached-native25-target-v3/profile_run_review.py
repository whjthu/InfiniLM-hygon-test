"""Require both marked native steps alongside the trained-run result."""

from hashlib import sha256
import json
import math
import stat


PHASES = ("backward", "gradient_clip", "muon_step")


def small_json(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 8192:
        raise ValueError("Invalid T09 diagnostic receipt")
    with path.open("rb") as stream:
        data = stream.read(8193)
    if len(data) != info.st_size:
        raise ValueError("T09 diagnostic receipt extent changed")
    return json.loads(data), sha256(data).hexdigest()


def read_phases(output):
    root = output / "t09-phase-profile"
    result = {"planned_steps": [6, 7], "actual_steps": [], "passed": False}
    try:
        intent, _ = small_json(root / "intent.json")
        if intent != {"planned_global_steps": [6, 7], "phases": list(PHASES), "formal_performance_eligible": False}:
            raise ValueError("T09 profile intent differs")
        for ordinal in (6, 7):
            value, digest = small_json(root / f"step-{ordinal:02d}.json")
            if (value["actual_global_step"] != ordinal
                    or value["uses_existing_device_synchronization"] is not True
                    or value["device_time_is_stream_interval_including_possible_idle"] is not True
                    or value["formal_performance_eligible"] is not False
                    or tuple(value["phase_seconds"]) != PHASES):
                raise ValueError("Wrong T09 phase receipt identity")
            for entry in value["phase_seconds"].values():
                if set(entry) != {"host_between_marks", "device_stream_between_marks"} or not all(
                    type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in entry.values()
                ):
                    raise ValueError("Invalid T09 phase duration")
            total = value["original_backward_update_seconds_with_observer"]
            if type(total) not in (int, float) or not math.isfinite(total) or total <= 0:
                raise ValueError("Invalid native diagnostic denominator")
            result["actual_steps"].append({"SHA256": digest, **value})
        result["passed"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["error_type"] = type(error).__name__
    return result
