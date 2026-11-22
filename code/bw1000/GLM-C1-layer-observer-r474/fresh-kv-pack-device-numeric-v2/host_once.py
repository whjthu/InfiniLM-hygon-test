"""Run the claimed GLM474 fresh K/V numerical gate through Slurm once."""

import argparse
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from types import ModuleType, SimpleNamespace


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
AREA = Path("code/bw1000/GLM-C1-layer-observer-r474")
DEVICE = AREA / "fresh-kv-pack-device-numeric-v2"
THIS = DEVICE / "host_once.py"
SNAPSHOT = Path("code/bw1000/training-qwen-native25-hip-v3/snapshot.py")
RESULT_READER = Path("code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-pack-device-host-result-v2/inspect_result.py")
WAITER = SNAPSHOT.with_name("slurm_accounting.py")
TRES = Path("tools/read_training_slurm_membership.py")
HOST_PY = "/usr/local/python3.12/bin/python3"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
PARENT, NODE = "1103606", "f12r2n13"
LABEL = "GLM474-fresh-kv-pack-numeric-GPU-v2"
REQUEST = "93148263aba346dda17b534d419d81c83"
WHOLE, WORK = 300, 270
COLUMNS = ("JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES")
STEP = re.compile(PARENT + r"\.\d+")
PHASES = {"pre": (0, 2, 2048), "gpu": (1, 4, 16384), "post": (0, 2, 2048)}
PINS = {
    RESULT_READER: "266449c1cabe928a33d4d6e2f46421e0c7e9969a49fd3efe33a3ad4dde2ced5c",
    DEVICE / "probe.py": "a9af94ebc3c016cd3174adbe9e48485258968ea5d021ac9dfaaa7c1f62d39818",
    DEVICE / "CLAIM.json": "c6e91a971e21c76ea8a212eea16c609d7bc360d1d9311bee3f64a4bae753f2be",
    DEVICE / "device_inside.sh": "bbb84d0e3b1dcffdfa4873130f2466ed65dabb6d12ff5eeaf6d35ed1868e9a10",
    AREA / "depth-stratified-observer-CPU-v1/ENV.sh": "147ea951bfe67a82ba39ef62e2e695e4ae7d21f8cc03f95b60054527ed960db9",
    SNAPSHOT: "2e8592caa8b573aa676884b63c65d5a378540d77c877d885424d934a5c4ebcb8",
    WAITER: "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
    TRES: "d2973a1841a0e8cc2fcf616fee383e8ea95b783d4914f203889a6dcafedc22b3",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def bounded(path, maximum=131072):
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    need(0 < len(raw) <= maximum, "unexpected byte length of " + path.name)
    return raw


def pin(root, name, expected, maximum=131072):
    path = root / name
    need(path.resolve(strict=True) == path and path.is_file(), "noncanonical staged source " + str(name))
    raw = bounded(path, maximum)
    need(sha256(raw).hexdigest() == expected, "staged source differs: " + str(name))
    return raw


def load(name, raw):
    module = ModuleType(name.stem + "_verified")
    exec(compile(raw, str(name), "exec"), module.__dict__)
    return module


def verify_stage(checkout):
    sources = {name: pin(checkout, name, expected) for name, expected in PINS.items()}
    probe = load(DEVICE / "probe.py", sources[DEVICE / "probe.py"])
    for name, expected in ((probe.PLAN, probe.PLAN_SHA), (probe.ARCHIVE, probe.ARCHIVE_SHA),
                           (probe.HELPER, probe.HELPER_SHA), (probe.KFD_HELPER, probe.KFD_HELPER_SHA)):
        pin(checkout, name, expected, 2 * 1024**2)
    return probe, load(WAITER, sources[WAITER]), load(TRES, sources[TRES])


def save(path, value):
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def label(phase):
    return LABEL if phase == "gpu" else LABEL + "-" + phase


def select_rows(raw):
    rows = {}
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        row = line.split("|")
        need(len(row) == len(COLUMNS), "incomplete Slurm accounting columns")
        for phase in PHASES:
            if row[1] == label(phase):
                need(phase not in rows and STEP.fullmatch(row[0]) and row[2].split()
                     and re.fullmatch(r"\d+:\d+", row[3]), "ambiguous or malformed original phase accounting")
                rows[phase] = row
    return rows


def accounting(timeout):
    raw = subprocess.check_output(
        ["sacct", "-n", "-P", "-j", PARENT, "-S", "2026-11-19T00:00:00", "-o", ",".join(COLUMNS)],
        timeout=timeout,
    )
    need(0 < len(raw) <= 2 * 1024**2 and any(line.startswith(PARENT.encode() + b"|") for line in raw.splitlines()),
         "complete parent accounting was not observed")
    return raw


def parent_end(fields):
    lines = subprocess.check_output(["scontrol", "show", "job", PARENT, "--oneliner"], timeout=8).decode().splitlines()
    need(len(lines) == 1, "parent allocation must be visible once")
    row = fields(lines[0])
    owner = re.fullmatch(r"[^()]+\((\d+)\)", row.get("UserId", ""))
    need(row.get("JobId") == PARENT and row.get("JobState") == "RUNNING" and owner
         and int(owner[1]) == os.getuid(), "parent allocation owner or state changed")
    return datetime.fromisoformat(row["EndTime"]).timestamp()


def slurm(phase, minutes, payload):
    cards, cpus, mem = PHASES[phase]
    return ["srun", "--jobid=" + PARENT, "--job-name=" + label(phase), "--exact", "--immediate=5",
            "--nodes=1", "--ntasks=1", f"--cpus-per-task={cpus}", "--cpu-bind=threads", f"--mem={mem}M",
            "--gres=" + ("dcu:1" if cards else "none"), "--nodelist=" + NODE, f"--time={minutes}", *payload]


def container(checkout, run):
    mounts = ((ROOT, "/refactor", "ro"), (checkout, "/source", "ro"), (run, "/scope", "rw"),
              (Path("/opt/hyhal"), "/opt/hyhal", "ro"))
    cmd = [APPTAINER, "exec", "--rocm", "--no-home", "--pwd", "/"]
    for path, container_path, mode in mounts:
        need(not any(char in str(path) for char in (":", ",", "\n")), "unsafe Apptainer source bind")
        cmd.extend(["--bind", f"{path}:{container_path}:{mode}"])
    return [*cmd, str(ROOT / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc",
            str(Path("/source") / DEVICE / "device_inside.sh")]


def own_identity(record, row, intent_sha, probe):
    need(record["state"] == "STARTED" and record["request"] == REQUEST and record["label"] == LABEL
         and record["step"] == row[0] and record["node"] == NODE and record["intent_SHA256"] == intent_sha
         and record["planned"] == probe.PLANNED and record["native_SHA256"] == probe.NATIVE_SHA
         and type(record["pid"]) is int and record["pid"] > 1
         and type(record["startticks"]) is int and record["startticks"] > 0
         and re.fullmatch(r"[0-9a-f-]{36}", record["boot_id"]), "GPU worker identity or denominator differs")
    return {key: record[key] for key in ("pid", "startticks", "boot_id")}


def owner_state(pid, start_ticks):
    if pid is None:
        return "NO_WORKER_IDENTITY"
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except FileNotFoundError:
        return "OWNER_ABSENT"
    if int(fields[19]) != start_ticks:
        return "PID_REUSED"
    return "OWNER_ZOMBIE" if fields[0] == "Z" else "OWNER_LIVE"


def snapshot_main(checkout, phase, owner):
    module = load(SNAPSHOT, pin(checkout, SNAPSHOT, PINS[SNAPSHOT]))
    module.NODE, module.STEP = NODE, STEP
    info = module.snapshot(False)
    need(info["full_fd_inventory_performed"] is True and info["node"] == NODE, "incomplete HOST device snapshot")
    if phase == "post":
        if owner is not None:
            need(owner["boot_id"] == info["boot_id"], "original worker boot changed")
        info["worker_identity"] = owner
        info["worker_state"] = owner_state(owner["pid"], owner["startticks"]) if owner else "NO_WORKER_IDENTITY"
    print(json.dumps(info, sort_keys=True), flush=True)


def snapshot_cmd(checkout, phase, owner=None):
    argv = ["/usr/bin/timeout", "--kill-after=3s", "12s", HOST_PY, "-I", "-S", "-B",
            str(checkout / THIS), "--checkout", str(checkout), "--snapshot", phase]
    if owner is not None:
        argv += ["--worker", json.dumps(owner, separators=(",", ":"))]
    return argv


def read_snapshot(path, row, probe):
    raw = bounded(path, 16384)
    snap = json.loads(raw)
    pci = snap.get("pci")
    need(snap["step"] == row[0] and snap["kind"] == "training_qwen_device_snapshot"
         and snap["node"] == NODE and snap["mode"] == "host_complete"
         and snap["full_fd_inventory_performed"] is True and type(snap["kfd_holder_count"]) is int
         and snap["kfd_holder_count"] >= 0
         and len(pci) == len(set(pci)) == 8 and all(probe.PCI.fullmatch(item) for item in pci),
         "HOST snapshot identity differs")
    return raw, snap


def inspect_probe(path, gpu_row, intent_sha, before, probe):
    result = dict(worker=None, expected=probe.PLANNED, passed={"valid": 0, "rejected": 0}, trace_pass=False)
    if not path.is_file():
        return result
    lines = []
    try:
        raw = bounded(path, 262144)
        result["trace_SHA256"] = sha256(raw).hexdigest()
        lines = raw.splitlines(keepends=True)
        need(lines[0].endswith(b"\n"), "GPU worker first receipt was interrupted")
        start = json.loads(lines[0])
        result["worker"] = own_identity(start, gpu_row, intent_sha, probe)
        need(result["worker"]["boot_id"] == before["boot_id"], "worker boot differs from fresh admission")
        need(len(lines) > 1 and lines[1].endswith(b"\n"), "GPU worker provider receipt missing")
        bound = json.loads(lines[1])
        physical = bound["physical_device"]
        need(bound["state"] == "BOUND" and bound["providers"]["count"] == 71 and bound["hip_SHA256"] == probe.HIP_SHA
             and bound["providers"]["canonical_HIP"] == str(probe.HIP_CANONICAL)
             and bound["providers"]["canonical_HIP_bytes"] == probe.HIP_BYTES
             and physical["pci"] in before["pci"] and physical["visible_devices"] == 1
             and physical["kfd"]["pci"] == physical["pci"] and physical["kfd"]["gpu_id"] > 0,
             "actual DSO provider or assigned GPU receipt differs")
        result["physical_device"] = physical
        offset = 2
        for group, ids in probe.PLANNED.items():
            for key in ids:
                need(lines[offset].endswith(b"\n"), "interrupted GPU case start")
                started = dict(group=group, id=key, state="STARTED", output_SHA256=None)
                need(json.loads(lines[offset]) == started, "GPU case start differs from frozen order")
                result["last_started"] = dict(group=group, id=key)
                need(lines[offset+1].endswith(b"\n"), "interrupted GPU case result")
                hashes = [sha256(blob).hexdigest() for blob in probe.reference(*probe.fixture(key[0]))] if group == "valid" else None
                need(json.loads(lines[offset+1]) == {**started, "state": "PASS", "output_SHA256": hashes},
                     "GPU case did not pass or an output hash differs")
                result["passed"][group] += 1
                offset += 2
        need(len(lines) == offset+1 and lines[offset].endswith(b"\n"), "GPU result is missing or extended")
        finish = json.loads(lines[offset])
        need(finish["state"] == "PASS" and finish["teardown"] == "normal"
             and all(8589934592 <= memory["free_bytes"] <= memory["total_bytes"]
                     for memory in (bound["before"], finish["during"], finish["after"])),
             "GPU teardown or memory floor differs")
        result["trace_pass"] = True
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result["error"] = f"{type(error).__name__}: {str(error)[:220]}"
        if lines and lines[-1].endswith(b"\n"):
            try:
                last = json.loads(lines[-1])
                if isinstance(last, dict) and last.get("state") == "FAIL":
                    result["worker_failure"] = {key: str(last.get(key, ""))[:1024] for key in ("type", "error")}
            except (UnicodeError, ValueError):
                pass
    return result


def launch(checkout, run, publication, admission_sha, probe, waiter, tres, *, query=accounting,
           expires=None, runner=subprocess.run, mount=container, wall=time.time, mono=time.monotonic):
    need(not run.exists() and not run.is_symlink(), "original device namespace already exists; reconcile")
    need(not select_rows(query(8)), "original Slurm phase already exists; reconcile")
    expiry = (expires or (lambda: parent_end(tres.fields)))()
    epoch, origin = wall(), mono()
    need(math.isfinite(expiry) and expiry >= epoch+WHOLE, "parent expires inside the one GPU epoch")
    run.mkdir(mode=0o700)
    scope = dict(label=LABEL, request=REQUEST, parent=PARENT, node=NODE, archive_SHA256=probe.ARCHIVE_SHA,
                 epoch=epoch, deadline=epoch+WHOLE, work_deadline=epoch+WORK, replay_allowed=False,
                 publication=publication, upstream_admission_SHA256=admission_sha, parent_end=expiry)
    save(run / "controller-intent.json", scope)
    result = dict(request=REQUEST, state="UNKNOWN", accounting_columns=COLUMNS, phases={},
                  denominator=probe.PLANNED, actual_GPU_steps_observed=0, post_node_clear=False,
                  gpu_attempt_recorded=False, post_worker_state="UNOBSERVED", cleanup_qualified=False,
                  numerical_pass=False, Model_GO=False)

    def left():
        return min(scope["deadline"] - wall(), WHOLE - (mono() - origin))

    def phase(name, payload, minutes=1):
        limits = {"pre": (22, 6), "gpu": (60*minutes+8, 76), "post": (25, 2)}
        cap, reserve = limits[name]
        need(left() >= minutes*60+5 and left() > reserve + 10, "insufficient original budget for " + name)
        report = dict(row=None, terminal_visible=False, completed_zero=False, observations=[])
        result["phases"][name] = report
        save(run / (name + "-attempt.json"), dict(request=REQUEST, phase=name, label=label(name),
             argv=slurm(name, minutes, payload), deadline=scope["deadline"], replay_allowed=False))
        if name == "gpu":
            result["gpu_attempt_recorded"] = True
        try:
            with (run / (name + ".log")).open("xb") as output:
                report["client_rc"] = runner(slurm(name, minutes, payload), stdin=subprocess.DEVNULL, stdout=output,
                                             stderr=subprocess.STDOUT, timeout=min(cap, left()-reserve), check=False).returncode
        except subprocess.TimeoutExpired:
            report["client_timeout"] = True
        except OSError as error:
            report["client_error"] = str(error)[:220]

        def lookup():
            budget = min(5, left() - (65 if name == "gpu" else 1))
            if budget <= 0:
                raise TimeoutError("original phase accounting budget exhausted")
            raw = query(budget)
            row = select_rows(raw).get(name)
            report["observations"].append(dict(accounting_SHA256=sha256(raw).hexdigest(), row=row))
            return row

        try:
            terminal = waiter.wait_terminal(lookup, mono()+min(12, max(0, left()-(65 if name == "gpu" else 1))),
                                            interval=1.5, max_observations=5)
            row = terminal["row"]
            report.update(row=row, terminal_visible=terminal["terminal_visible"])
            if row:
                report["placement_matches"] = (row[5] == NODE and tres.allocated_dcu(row[6]) == PHASES[name][0]
                                               and f"cpu={PHASES[name][1]}" in row[6].split(","))
                report["completed_zero"] = report["placement_matches"] and row[2:4] == ["COMPLETED", "0:0"]
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report["accounting_error"] = f"{type(error).__name__}: {str(error)[:220]}"
        save(run / (name + "-accounting.json"), report)
        return report

    try:
        pre = phase("pre", snapshot_cmd(checkout, "pre"))
        if not pre["completed_zero"]:
            if pre["terminal_visible"]:
                result["state"] = "PRE_TERMINAL_UNQUALIFIED"
            return result
        result["state"] = "PRE_TERMINAL_NO_GPU_ATTEMPT"
        before_raw, before = read_snapshot(run / "pre.log", pre["row"], probe)
        need(before["kfd_holder_count"] == 0, "fresh target holds a GPU process")
        with (run / "before.json").open("xb") as stream:
            stream.write(before_raw)
        scope["pre_admission_SHA256"] = sha256(before_raw).hexdigest()
        intent_sha = save(run / "intent.json", scope)
        result["intent_SHA256"] = intent_sha
        gpu_minutes = min(3, math.floor((left() - 85) / 60))
        need(gpu_minutes >= 1, "original epoch leaves no GPU plus independent post window")
        result["state"] = "UNKNOWN"
        gpu = phase("gpu", mount(checkout, run), gpu_minutes)
        result["actual_GPU_steps_observed"] = int(any(x["row"] is not None for x in gpu["observations"]))
        if not gpu["terminal_visible"]:
            return result
        result["state"] = "GPU_TERMINAL_UNVERIFIED"
        reader = load(RESULT_READER, pin(checkout, RESULT_READER, PINS[RESULT_READER]))
        host = SimpleNamespace(inspect_probe=inspect_probe, bounded=bounded, own_identity=own_identity, need=need)
        observed = reader.inspect_probe(run / "probe.jsonl", gpu["row"], intent_sha, before, probe, host)
        result["probe"] = observed
        owner = observed["worker"]
        post = phase("post", snapshot_cmd(checkout, "post", owner))
        if not post["completed_zero"]:
            return result
        raw, after = read_snapshot(run / "post.log", post["row"], probe)
        result["post_snapshot_SHA256"] = sha256(raw).hexdigest()
        result["post_worker_state"] = after["worker_state"]
        result["post_node_clear"] = (after["boot_id"] == before["boot_id"] and after["pci"] == before["pci"]
                                     and after["kfd_holder_count"] == 0)
        result["cleanup_qualified"] = (gpu.get("placement_matches") is True and result["post_node_clear"] and owner is not None
                                       and after["worker_identity"] == owner
                                       and after["worker_state"] in {"OWNER_ABSENT", "PID_REUSED", "OWNER_ZOMBIE"})
        result["numerical_pass"] = gpu["completed_zero"] and observed["trace_pass"] and result["cleanup_qualified"]
        if result["numerical_pass"]:
            result["state"] = "GPU_NUMERICAL_PASS_POST_CLEAN"
        elif result["cleanup_qualified"]:
            result["state"] = "GPU_TERMINAL_NUMERICAL_UNQUALIFIED_POST_CLEAN"
        return result
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result["error"] = f"{type(error).__name__}: {str(error)[:220]}"
        return result
    finally:
        save(run / "outcome.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--publication")
    parser.add_argument("--admission-sha256")
    parser.add_argument("--snapshot", choices=("pre", "post"))
    parser.add_argument("--worker")
    args = parser.parse_args()
    checkout = args.checkout
    need(checkout.is_relative_to(ROOT / "incoming") and checkout.resolve(strict=True) == checkout
         and Path(__file__).resolve() == checkout / THIS, "wrong canonical reviewed checkout")
    if args.snapshot:
        snapshot_main(checkout, args.snapshot, json.loads(args.worker) if args.worker else None)
        return 0
    need(re.fullmatch(r"[0-9a-f]{40}", args.publication or "")
         and re.fullmatch(r"[0-9a-f]{64}", args.admission_sha256 or ""), "publication and actual root admission pin required")
    probe, waiter, tres = verify_stage(checkout)
    outcome = launch(checkout, ROOT / "results" / (LABEL + "-controller"), args.publication,
                     args.admission_sha256, probe, waiter, tres)
    print(json.dumps(outcome, sort_keys=True), flush=True)
    return 0 if outcome["numerical_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
