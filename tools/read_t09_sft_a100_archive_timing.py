"""Reconcile the original three HunyuanVideo SFT timing windows against Hygon."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile


MANIFEST_SHA = "efda8f5535c029a8fbc3b5014004bfb34632ee105f815ae378122d6dd066e085"
ARCHIVE_SHA = "92187897d860b241b60a9686636590a6623b3f9c625f11e24baba28124bee664"
LAUNCH_SHA = "3fdd29e7ea1a09fc54872c044699199cd5ef16bd73af6e7d522fe2ff1b32662f"
PREFIX = "node/results/hunyuan-cuda-sft-baseline-r2-walltime-repeat"
HYGON = Path("results/bw1000/2026-11-23/training-hunyuanvideo-original794-phase-timing-and-window-bridge.json")
HYGON_SHA = "eb52d9bb007f7a3b7e0a1daa26ba280254e5ed1d3b5177fccec5151e46e40ddb"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def member_bytes(archive, pins, name):
    pin = pins[name]
    member = archive.getmember(name)
    require(member.isfile() and 0 < member.size == pin["bytes"] <= 10000, "original evidence extent")
    with archive.extractfile(member) as stream:
        raw = stream.read(member.size + 1)
    require(len(raw) == member.size and sha256(raw) == pin["sha256"], "original evidence bytes")
    return raw


def read_repeat(archive, pins, repeat):
    prefix = f"{PREFIX}{repeat}/"
    paths = {role: prefix + name for role, name in
             (("launch", "launch.json"), ("result", "result.json"),
              ("summary", "summary.json"), ("steps", "steps.ndjson"))}
    raw = {role: member_bytes(archive, pins, name) for role, name in paths.items()}
    require(sha256(raw["launch"]) == LAUNCH_SHA, "original timing launch digest")
    config = json.loads(raw["launch"])["config"]
    result = json.loads(raw["result"])
    summary = json.loads(raw["summary"])
    steps = [json.loads(line) for line in raw["steps"].splitlines()]
    metrics = result["metrics"]
    require(config["max_steps"] == 20 and config["warmup_steps"] == 5
            and config["checkpoint_steps"] == [10, 20]
            and metrics["executed_measured_steps"] == 20 and metrics["executed_warmup_steps"] == 5
            and [step["global_step"] for step in steps] == list(range(1, 26))
            and summary["checkpoint_step_basis"] == "global-optimizer-step", "original update extent")
    started = [step["started_unix"] for step in steps]
    ended = [step["ended_unix"] for step in steps]
    require(all(math.isfinite(t) for t in started + ended)
            and all(a < b for a, b in zip(started, ended, strict=True))
            and all(a <= b for a, b in zip(ended[:-1], started[1:], strict=True)), "original clocks")
    span = ended[-1] - started[5]
    wall = metrics["measurement_wall_seconds"]
    rate = metrics["measurement_updates_per_second"]
    require(0 < span < wall and math.isclose(rate, 20 / wall, rel_tol=1e-12), "original wall timing")
    return dict(repeat=repeat, declared_boundary=metrics["timing_boundary"], measured_updates=20, raw_updates=25,
                start_update=6, end_update=25, measured_seconds=wall, measured_updates_per_second=rate,
                step_clock_span_seconds=span, wall_minus_step_clock_span_seconds=wall-span,
                gaps_after_checkpoint_seconds={str(i): started[i]-ended[i-1] for i in (10, 20)},
                checkpoint_artifact_count=sum(Path(item["path"]).name == "state.pt" for item in result["artifacts"]),
                input_pins={role: {"bytes": len(value), "SHA256": sha256(value)} for role, value in raw.items()})


def analyze(evidence, repo):
    raw = (evidence / "remote-node26-r2-manifest.json").read_bytes()
    require(sha256(raw) == MANIFEST_SHA, "original A100 manifest")
    pins = {item["path"]: item for item in json.loads(raw)["files"]}
    archive = evidence / "remote-node26-r2.tar.gz"
    require(archive.stat().st_size == 8881640 and sha256(archive.read_bytes()) == ARCHIVE_SHA,
            "original A100 archive")
    with tarfile.open(archive, "r:gz") as handle:
        runs = [read_repeat(handle, pins, repeat) for repeat in (1, 2, 3)]
    raw = (repo / HYGON).read_bytes()
    require(sha256(raw) == HYGON_SHA, "original Hygon measurement evidence")
    hygon = json.loads(raw)["original_actual_measurement"]
    require(hygon["actual_gpu_step"] == "1087656.794" and hygon["actual_cards"] == 1
            and hygon["measured_updates"] == 20 and hygon["first_measured_update"] == 6
            and hygon["last_measured_update"] == 25, "original Hygon step extent")
    boundary = hygon["native_checkpoint_inclusive_boundary"]
    require(all(run["declared_boundary"] == boundary and run["checkpoint_artifact_count"] == 3 for run in runs),
            "declared checkpoint and hash boundary differs")
    median = statistics.median(run["measured_updates_per_second"] for run in runs)
    observed = hygon["native_checkpoint_inclusive_updates_per_second"]
    stage3 = 0.9 * median
    return dict(state="T09_SFT_THREE_ORIGINAL_A100_CHECKPOINT_HASH_BOUNDARIES_MATCH_HYGON_DECLARATION",
                case="T09.sft", source_archive_SHA256=ARCHIVE_SHA, source_manifest_SHA256=MANIFEST_SHA,
                Hygon_source_SHA256=HYGON_SHA, Hygon_source_path=HYGON.as_posix(), A100=runs,
                A100_median_updates_per_second=median, Hygon_original_step=hygon["actual_gpu_step"],
                Hygon_observed_updates_per_second=observed, Hygon_observed_seconds=hygon["native_checkpoint_inclusive_seconds_from_rate"],
                Hygon_observations=1, additional_Hygon_repeats_needed=2, declared_timing_boundary=boundary,
                ratio_single_Hygon_to_A100_median=observed/median,
                Hygon_stage3_required_updates_per_second=stage3,
                Hygon_wall_seconds_savings_required_for_single_run_stage3=(
                    hygon["native_checkpoint_inclusive_seconds_from_rate"] - 20 / stage3),
                same_input_value_audit_passed=False,
                formal_performance_passed=False, new_gpu_runs=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="directory containing the verified A100 archive and manifest")
    args = parser.parse_args()
    result = json.dumps(analyze(args.evidence, Path(__file__).resolve().parent.parent),
                        ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.buffer.write((result + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
