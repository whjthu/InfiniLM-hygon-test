from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
A100 = Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline")
manifest_bytes = (A100 / "evidence/remote-node26-r2-manifest.json").read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest() == "efda8f5535c029a8fbc3b5014004bfb34632ee105f815ae378122d6dd066e085"
manifest = json.loads(manifest_bytes)
entries = {row["path"]: row for row in manifest["files"]}
paths = [f"node/results/hunyuan-cuda-rl-baseline-r1-repeat{repeat}/{name}"
         for repeat in (1, 2, 3) for name in ("launch.json", "result.json", "steps.ndjson", "summary.json")]
pins = {path: entries[path] for path in paths}
remote = r"""
from pathlib import Path
import hashlib
import json
import math
import statistics
import tarfile

home = Path("/home/whj/nvidia-baseline")
archive = Path(ARCHIVE)
assert archive.parent == home and archive.name.startswith("evidence-export-app-")
archive_reader = None

def read(rel):
    global archive_reader
    pin = PINS[rel]
    assert 0 < pin["bytes"] <= 65536
    path = Path(rel)
    for candidate in (home / path, home.joinpath(*path.parts[1:])):
        if candidate.is_file() and not candidate.is_symlink() and candidate.stat().st_size == pin["bytes"]:
            raw = candidate.read_bytes()
            if hashlib.sha256(raw).hexdigest() == pin["sha256"]:
                return raw
    if archive_reader is None:
        archive_reader = tarfile.open(archive, mode="r:gz")
    member = archive_reader.getmember(rel)
    assert member.isfile() and member.size == pin["bytes"]
    with archive_reader.extractfile(member) as stream:
        raw = stream.read(pin["bytes"] + 1)
    assert len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"]
    return raw

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def audit(repeat):
    stem = f"node/results/hunyuan-cuda-rl-baseline-r1-repeat{repeat}/"
    launch = json.loads(read(stem + "launch.json"))["config"]
    result = json.loads(read(stem + "result.json"))
    summary = json.loads(read(stem + "summary.json"))
    rows = [json.loads(line) for line in read(stem + "steps.ndjson").splitlines()]
    assert len(rows) == 25 and [row["global_step"] for row in rows] == list(range(1, 26))
    assert (launch["warmup_steps"], launch["max_steps"], launch["num_generations"], launch["sampling_steps"]) == (5,20,4,4)
    component_keys = sorted(rows[0]["reward"])
    assert component_keys
    for row in rows:
        rewards = row["reward"]
        assert sorted(rewards) == component_keys
        assert all(len(v) == 4 and all(map(finite, v)) for v in rewards.values())
        for key in ("grad_norm", "ratio_min", "ratio_mean", "ratio_max", "reward_variance", "step_seconds"):
            assert finite(row[key]) and row[key] > 0
        for key in ("log_prob_mae", "log_prob_p99"):
            assert finite(row[key]) and row[key] >= 0
        assert row["ratio_min"] <= row["ratio_mean"] <= row["ratio_max"]
        assert row["rollout_adapter_sha"] == row["actor_start_adapter_sha"]
        assert row["adapter_sha_after"] != row["actor_start_adapter_sha"]
    assert all(a["adapter_sha_after"] == b["actor_start_adapter_sha"] for a, b in zip(rows, rows[1:]))
    measured = rows[5:]
    metrics = result["metrics"]
    wall = metrics["measurement_wall_seconds"]
    throughput = 20 / wall
    assert metrics["measurement_steps"] == 20 and math.isclose(throughput, metrics["measurement_updates_per_second"], rel_tol=1e-12)
    for key in ("log_prob_mae", "log_prob_p99"):
        assert math.isclose(max(r[key] for r in measured), metrics[key + "_max"], rel_tol=1e-12)
    assert math.isclose(statistics.mean(r["ratio_mean"] for r in measured), metrics["ratio_mean"], rel_tol=1e-12)
    return dict(repeat=repeat, reward_components=component_keys, rewards_per_component_per_update=4,
                warmup_updates=5, measured_updates=20, recorded_adapter_changes=25,
                first_adapter_sha=rows[0]["actor_start_adapter_sha"], final_adapter_sha=rows[-1]["adapter_sha_after"],
                max_log_prob_mae=metrics["log_prob_mae_max"], max_log_prob_p99=metrics["log_prob_p99_max"],
                measurement_wall_seconds=wall, measurement_updates_per_second=throughput,
                sum_measured_step_seconds=sum(r["step_seconds"] for r in measured),
                step_fields=sorted(rows[0]), summary_fields=sorted(summary))
try:
    rows = [audit(repeat) for repeat in (1, 2, 3)]
    print(json.dumps(dict(runs=rows, median_updates_per_second=statistics.median(r["measurement_updates_per_second"] for r in rows))))
finally:
    if archive_reader:
        archive_reader.close()
"""
args = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-F", str(A100 / "ssh_config"),
        "a100-26", "timeout 50 python3 -B -u -"]
program = f"PINS={pins!r}\nARCHIVE={manifest['archive']!r}\n" + remote
try:
    call = subprocess.run(args, input=program.encode(), capture_output=True, timeout=60)
except subprocess.TimeoutExpired:
    raise SystemExit('{"state":"SSH_READ_TIMEOUT"}') from None
receipt = dict(rc=call.returncode, stdout_SHA256=hashlib.sha256(call.stdout).hexdigest(),
               stderr_SHA256=hashlib.sha256(call.stderr).hexdigest(), pins=pins)
if call.returncode == 0:
    receipt["audit"] = json.loads(call.stdout)
else:
    receipt["failure_types"] = [s for s in ("AssertionError", "KeyError", "Connection closed", "Permission denied")
                                if s.encode() in call.stderr]
with (ROOT / "hunyuan-sage-a100-step-audit-first.json").open("x", encoding="utf8") as output:
    json.dump(receipt, output, indent=2)
print(json.dumps({key: value for key, value in receipt.items() if key != "pins"}))
