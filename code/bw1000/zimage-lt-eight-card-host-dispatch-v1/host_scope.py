"""Validate and render the existing O01 octet handoff. This module runs no scheduler commands."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time


REQUEST = "e23d24ea0c9c4d00af06154d82dc22e8"
LABEL = "O01-zimage-lt-eight-card-supervisor-v1"
PRE_REQUEST = "f0674bbd407942a4a2455634c8cb44c4"
POST_REQUEST = "a57632cbc5534cf781abbde00746a7a9"
PRE_LABEL = "O01-zimage-hybrid-pre-" + PRE_REQUEST[:12]
POST_LABEL = "O01-zimage-hybrid-post-" + POST_REQUEST[:12]
PARENT, NODE = "1103606", "f12r2n11"
OLD_STEP = "1103606.2075"
OLD_LABEL = "training80-dcu-standby-3228d0f2042f-f12r2n11"
PARENT_UID = r"\(([0-9]+)\)$"
STEP = re.compile(PARENT + r"\.[0-9]+")
PCI = frozenset(f"0000:{bus}:00.0" for bus in ("09", "36", "55", "77", "85", "b5", "d5", "f5"))
SHA = re.compile("[0-9a-f]{64}")
REQUEST_ID = re.compile("[0-9a-f]{32}")
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "DEADLINE", "BOOT_FAIL", "REVOKED"}
PROJECT = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = PROJECT / "results"
RUN = RESULTS / (LABEL + "-run")
STAGE = PROJECT / "incoming" / (LABEL + "-host-stage")
POST = RESULTS / POST_LABEL
PRE = RESULTS / PRE_LABEL
LOCK = RESULTS / ".training80-dcu-standby.lock"
PYTHON = "/usr/local/python3.12/bin/python3"
NATIVE = RESULTS / "training-zimage-lt-native-build-fd31777395fb474f86dedacef84dcce9/libzimage_lt_probe.so"
NATIVE_SHA = "f735cc99e10d0c958b3736fde852bbf9e1f9747827225f1375f0214726279a07"
ATTESTED_SHA = "c7f13669918f720ea195cb49d0be843681d3155b025520caa8b1887babbd6ee6"
DEVICE_HOST_SHA = "402fce88ebf96b57ed474059e7bc0c080dc377abfebb52dce6447c85336a4259"
INVENTORY_SHA = "dfb047f40a1f21e9ff606afeb95e60026f9d350b5bb35e3b447d8e5b86fa7efc"
RUNTIME_SHA = "7ba1d9355838c09d41176f67708d892528832971f83f036be63d6ddacbbdb35d"
HOST2105_SHA = "10a0070a7f5759dee3bc8f463a336ef6140908e600e6e0c66b644d3d7313fda0"
OWNER2108_SHA = "4d26986955950b073ec03d4227f3895e3a693802eb885aca1b4dc729cbb84b37"


def require(ok, gate):
    if not ok:
        raise ValueError(gate)


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def encoded(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode()


def digest(data):
    return sha256(data).hexdigest()


def read_exact(path, expected, cap=65536):
    require(path.is_file() and not path.is_symlink(), "EXACT_REGULAR_FILE_REQUIRED")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(0 < len(raw) <= cap and digest(raw) == expected, "ARTIFACT_BYTES_OR_SHA")
    return raw


def save_new(path, value):
    raw = encoded(value)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return digest(raw)


def rows(scheduler):
    require(scheduler["accounting_columns"] == COLUMNS and scheduler["accounting_complete"] is True,
            "ACCOUNTING_COLUMNS_OR_COMPLETENESS")
    records = scheduler["accounting_rows"]
    require(type(records) is list and all(type(row) is list and len(row) == len(COLUMNS)
                                         and all(type(field) is str for field in row) for row in records),
            "ACCOUNTING_ROW_SHAPE")
    return records


def exact_row(records, label):
    matching = [row for row in records if row[1] == label]
    require(len(matching) == 1 and STEP.fullmatch(matching[0][0]) and matching[0][5] == NODE,
            "EXACT_NUMBERED_ACCOUNTING")
    row = matching[0]
    require(sum(other[0] == row[0] for other in records) == 1 and row[4].isdecimal(), "ACCOUNTING_ID_DUPLICATE")
    return row


def check_resources(tres, cpus, dcu, memories):
    entries = tres.split(",")
    resources = dict(item.split("=", 1) for item in entries)
    expected = {"cpu", "mem", "node"} | ({"gres/dcu"} if dcu else set())
    require(len(resources) == len(entries) and set(resources) == expected and resources["cpu"] == str(cpus)
            and resources["node"] == "1" and resources["mem"] in memories
            and resources.get("gres/dcu") == (str(dcu) if dcu else None), "ACCOUNTED_RESOURCES")


def scheduler_scope(scheduler, now, uid):
    observed = scheduler["epoch"]
    require(is_number(observed) and 0 <= now - observed <= 30, "STALE_SCHEDULER_OBSERVATION")
    parent = scheduler["parent"]
    uid_suffix = re.search(PARENT_UID, parent["UserId"])
    membership = scheduler["parent_nodes"]
    require(parent["JobId"] == PARENT and parent["JobState"] == "RUNNING"
            and uid_suffix and int(uid_suffix[1]) == uid and len(membership) == len(set(membership)) == 40
            and NODE in membership and "gres/dcu=320" in parent["AllocTRES"].split(","), "PARENT_OWNERSHIP")
    require(scheduler["numbered_steps_complete"] is True, "INCOMPLETE_NUMBERED_STEPS")
    records = rows(scheduler)
    target_active = []
    for step in scheduler["numbered_steps"]:
        require(STEP.fullmatch(step["StepId"]) and step["State"] and step["Name"]
                and step["nodes"] and set(step["nodes"]) <= set(membership), "UNRESOLVED_NUMBERED_STEP")
        if NODE in step["nodes"] and step["State"] not in TERMINAL:
            target_active.append(step)
    ids = [step["StepId"] for step in scheduler["numbered_steps"]]
    require(len(set(ids)) == len(ids), "DUPLICATE_NUMBERED_STEP")
    return records, target_active


def old_standby(records):
    row = exact_row(records, OLD_LABEL)
    require(row[0] == OLD_STEP and row[2:4] == ["COMPLETED", "0:0"], "PREDECESSOR_STILL_PROTECTED")
    check_resources(row[6], 8, 8, {"16G", "16384M"})
    return row


def clean_inventory(records, phase, proof, *, observed, after, boot):
    label, role = (PRE_LABEL, PRE_REQUEST) if phase == "pre" else (POST_LABEL, POST_REQUEST)
    row = exact_row(records, label)
    require(row[2:4] == ["COMPLETED", "0:0"], "ZERO_GRES_INVENTORY_TERMINAL")
    check_resources(row[6], 2, 0, {"4G", "4096M"})
    require(proof["request"] == role and proof["source_SHA256"] == INVENTORY_SHA, "INVENTORY_PRODUCER")
    raw = proof["stdout"].encode("utf8")
    require(0 < len(raw) <= 16384 and SHA.fullmatch(proof["stdout_SHA256"])
            and digest(raw) == proof["stdout_SHA256"], "INVENTORY_RECEIPT_SHA")
    inventory = json.loads(raw)
    require(inventory["step"] == row[0] and inventory["node"] == NODE and row[0] != OLD_STEP
            and len(inventory["PCIs"]) == 8 and set(inventory["PCIs"]) == PCI
            and inventory["boot_id_SHA256"] == boot
            and type(inventory["kfd_process_count"]) is int and inventory["kfd_process_count"] == 0
            and type(inventory["own_device_fd_process_count"]) is int and inventory["own_device_fd_process_count"] == 0,
            "EXACT_SAME_BOOT_EIGHT_PCI_CLEANUP")
    start, end = proof["start_epoch"], proof["observed_epoch"]
    require(is_number(start) and is_number(end) and after <= start <= end <= observed and observed - end <= 30,
            "INDEPENDENT_CLEANUP_ORDER_OR_FRESHNESS")
    return row, proof["stdout_SHA256"]


def validate_admission(evidence, now, uid):
    require(evidence["request"] == REQUEST and evidence["label"] == LABEL
            and evidence["transport_disposition"] == "NEVER_ATTEMPTED", "ORIGINAL_REQUEST_STATE")
    scheduler = evidence["scheduler"]
    records, active = scheduler_scope(scheduler, now, uid)
    predecessor = old_standby(records)
    require(not active and not any(row[1] in (LABEL, POST_LABEL) for row in records)
            and not any(step["Name"] in (LABEL, POST_LABEL) for step in scheduler["numbered_steps"]),
            "TARGET_BUSY_OR_ORIGINAL_ALREADY_VISIBLE")
    boot = evidence["boot_id_SHA256"]
    earlier = evidence["predecessor_terminal_observed_epoch"]
    require(SHA.fullmatch(boot) and is_number(earlier), "BOOT_OR_PREDECESSOR_OBSERVATION")
    pre, pin = clean_inventory(records, "pre", evidence["independent_pre"], observed=scheduler["epoch"],
                               after=earlier, boot=boot)
    return {"request": REQUEST, "label": LABEL, "parent": PARENT, "node": NODE, "fresh_root_admitted": True,
            "allocated_DCU": 8, "owner_uid": uid, "PCI": sorted(PCI), "boot_id_SHA256": boot,
            "independent_pre_step": pre[0], "independent_pre_SHA256": pin, "predecessor_step": predecessor[0],
            "epoch": now, "deadline_epoch": now + 240, "attested_entry_SHA256": ATTESTED_SHA,
            "runtime_SHA256": RUNTIME_SHA, "replay_allowed": False}


def srun_base(label, cpu, gres, memory, seconds, out):
    return ["srun", "--jobid=" + PARENT, "--job-name=" + label, "--exact", "--immediate=5",
            "--nodes=1", "--ntasks=1", "--nodelist=" + NODE, "--cpus-per-task=" + str(cpu),
            "--threads-per-core=1", "--cpu-bind=threads", "--gres=" + gres, "--mem=" + memory,
            "--time=00:{:02d}:{:02d}".format(*divmod(seconds, 60)), "--kill-on-bad-exit=1",
            "--output=" + str(out / "slurm.stdout"), "--error=" + str(out / "slurm.stderr")]


def render_gpu(scope, scope_sha, entry_sha, launcher_sha):
    require(scope["request"] == REQUEST and scope["label"] == LABEL and scope["fresh_root_admitted"] is True
            and scope["deadline_epoch"] == scope["epoch"] + 240 and scope["attested_entry_SHA256"] == entry_sha == ATTESTED_SHA
            and scope["runtime_SHA256"] == RUNTIME_SHA and launcher_sha == DEVICE_HOST_SHA
            and scope_sha == digest(encoded(scope)), "EXACT_GPU_RENDER_SCOPE")
    return [*srun_base(LABEL, 12, "dcu:8", "24576M", 240, RUN),
            PYTHON, "-I", "-S", "-B", str(STAGE / "device_host.py"), "--launcher-sha256", launcher_sha,
            "--entry-sha256", entry_sha, "--scope-sha256", scope_sha, "--run", str(RUN),
            "--runtime", str(STAGE / "runtime")]


def render_inventory(phase):
    # The caller owns the fresh phase directory and retains the allocation lock through submission.
    label, output = {"pre": (PRE_LABEL, PRE), "post": (POST_LABEL, POST)}[phase]
    return [*srun_base(label, 2, "none", "4096M", 60, output),
            PYTHON, "-I", "-S", "-B", str(STAGE / "node_inventory.py"), PARENT, NODE]


def adopt_bound_owner(receipt, scope):
    owner, live = receipt["owner"], receipt["live_slurm"]
    require(receipt["request"] == REQUEST and receipt["scope_SHA256"] == digest(encoded(scope))
            and receipt["entry_SHA256"] == ATTESTED_SHA and receipt["runtime_SHA256"] == RUNTIME_SHA
            and receipt["slurm_listpids_matched"] is True and live["Name"] == LABEL and live["State"] == "RUNNING"
            and live["StepId"] == owner["step"] and STEP.fullmatch(owner["step"]) and owner["step"] != OLD_STEP
            and live["NodeList"] == owner["node"] == NODE and owner["boot_SHA256"] == scope["boot_id_SHA256"]
            and type(owner["uid"]) is int and owner["uid"] == scope["owner_uid"]
            and owner["cgroup_names_step"] is True and SHA.fullmatch(owner["cgroup_SHA256"])
            and type(owner["pid"]) is int and owner["pid"] > 1 and type(owner["start_ticks"]) is int
            and owner["start_ticks"] > 0 and len(owner["affinity"]) == len(set(owner["affinity"])) == 12
            and scope["epoch"] <= owner["epoch"] < scope["deadline_epoch"] - 185, "ATTESTED_KERNEL_OWNER")
    check_resources(live["TRES"], 12, 8, {"24G", "24576M"})
    return owner["step"]


def adopt_container_owner(host_raw, container_raw, scope):
    require(all(0 < len(raw) <= 16384 for raw in (host_raw, container_raw)), "ATTESTED_RECEIPT_SIZE")
    host, container = json.loads(host_raw), json.loads(container_raw)
    step = adopt_bound_owner(host, scope)
    before, inside = host["owner"], container["owner"]
    require(container["request"] == REQUEST and container["host_owner_SHA256"] == digest(host_raw)
            and all(inside[key] == before[key] for key in ("step", "node", "boot_SHA256", "uid", "affinity",
                                                          "cgroup_SHA256", "cgroup_names_step"))
            and inside["step"] == step and type(inside["pid"]) is int and inside["pid"] > 1
            and type(inside["start_ticks"]) is int and inside["start_ticks"] > 0
            and before["epoch"] <= inside["epoch"] < scope["deadline_epoch"] - 185, "CONTAINER_OWNER_DIFFERS")
    return {"step": step, "host_owner_SHA256": digest(host_raw), "container_owner_SHA256": digest(container_raw)}


def post_after_terminal(scheduler, scope, now, uid, *, attested=None):
    records, active = scheduler_scope(scheduler, now, uid)
    gpu = exact_row(records, LABEL)
    require(not active and gpu[0] not in (OLD_STEP, scope["independent_pre_step"])
            and gpu[2].rstrip("+") in TERMINAL, "GPU_TERMINAL_OR_TARGET_BUSY")
    check_resources(gpu[6], 12, 8, {"24G", "24576M"})
    if attested is not None:
        require(adopt_bound_owner(attested, scope) == gpu[0], "GPU_TERMINAL_DIFFERS_FROM_ADOPTED_OWNER")
    require(not any(row[1] == POST_LABEL for row in records)
            and not any(row["Name"] == POST_LABEL for row in scheduler["numbered_steps"]), "POST_ALREADY_ATTEMPTED")
    return gpu, render_inventory("post")


def refill_after_clean_post(scheduler, scope, now, uid, evidence):
    records, active = scheduler_scope(scheduler, now, uid)
    gpu = exact_row(records, LABEL)
    require(not active and gpu == evidence["adopted_gpu_accounting"] and gpu[2].rstrip("+") in TERMINAL,
            "ACTUAL_GPU_TERMINAL_REQUIRED")
    check_resources(gpu[6], 12, 8, {"24G", "24576M"})
    row, pin = clean_inventory(records, "post", evidence["independent_post"], observed=scheduler["epoch"],
                               after=evidence["gpu_terminal_observed_epoch"], boot=scope["boot_id_SHA256"])
    require(row[0] not in (OLD_STEP, scope["independent_pre_step"], gpu[0]), "POST_NOT_INDEPENDENT")
    return {"root_refill_eligibility": True, "post_step": row[0], "post_SHA256": pin,
            "post_observed_epoch": evidence["independent_post"]["observed_epoch"], "GPU_accounting": gpu,
            "boot_id_SHA256": scope["boot_id_SHA256"], "PCI": sorted(PCI)}


def consume_refill(scheduler, now, uid, eligibility, evidence):
    records, active = scheduler_scope(scheduler, now, uid)
    request = evidence["request"]
    require(REQUEST_ID.fullmatch(request) and request not in (REQUEST, PRE_REQUEST, POST_REQUEST,
                                                             "3228d0f2042f4224bb72bdf2313617c5"),
            "NEW_REFILL_REQUEST")
    label = "training80-dcu-standby-" + request[:12] + "-" + NODE
    row = exact_row(records, label)
    require(len(active) == 1 and active[0]["StepId"] == row[0] and active[0]["Name"] == label
            and active[0]["State"] == row[2] == "RUNNING" and row[0] not in (OLD_STEP, eligibility["post_step"],
                                                                           eligibility["GPU_accounting"][0])
            and eligibility["root_refill_eligibility"] is True, "REFILL_LIVE_SCOPE")
    check_resources(row[6], 8, 8, {"16G", "16384M"})
    first, second = evidence["first"], evidence["second"]
    for point in (first, second):
        require(point["step"] == row[0] and point["node"] == NODE and set(point["rows"]) == PCI
                and point["boot_id_SHA256"] == eligibility["boot_id_SHA256"]
                and is_number(point["epoch"]) and is_number(point["compute_observed_epoch"]),
                "REFILL_SNAPSHOT_IDENTITY")
    require(first["worker_PID"] == second["worker_PID"] > 1
            and first["worker_start_ticks"] == second["worker_start_ticks"] > 0
            and eligibility["post_observed_epoch"] < first["epoch"] < second["epoch"] <= scheduler["epoch"]
            and first["compute_observed_epoch"] < second["compute_observed_epoch"]
            and now - second["epoch"] <= 18, "REFILL_TWO_POINT_OWNER_OR_CLOCK")
    for pci in PCI:
        a, b = first["rows"][pci], second["rows"][pci]
        require(a["pci"] == b["pci"] == pci and a["active"] is b["active"] is True
                and type(a["calls"]) is type(b["calls"]) is int and 0 < a["calls"] < b["calls"]
                and a["calls"] % 8 == b["calls"] % 8 == 0
                and a["epoch"] < b["epoch"] and -1 <= first["compute_observed_epoch"] - a["epoch"] <= 18
                and -1 <= second["compute_observed_epoch"] - b["epoch"] <= 18,
                "REFILL_EIGHT_CARD_GEMM_GROWTH")
    return {"refill_step": row[0], "actual_computing_cards": 8, "request": request,
            "operator_performance_verified": False}


@contextmanager
def standby_lock():
    import fcntl

    with Path(LOCK).open("ab") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield held


def require_lease(held):
    # Only accept the shared lock hand-off from standby_lock(), never a Boolean admission.
    require(not held.closed and Path(held.name).resolve(strict=True) == Path(LOCK).resolve(strict=True),
            "SHARED_STANDBY_LOCK_REQUIRED")
    import fcntl

    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)


def stage_pins(launcher_sha):
    stage = Path(STAGE)
    require(launcher_sha == DEVICE_HOST_SHA, "EXACT_DEVICE_HOST_SOURCE")
    launcher = read_exact(stage / "device_host.py", launcher_sha)
    read_exact(stage / "node_entry.py", ATTESTED_SHA)
    read_exact(stage / "node_inventory.py", INVENTORY_SHA)
    read_exact(stage / "host2105.json", HOST2105_SHA)
    read_exact(stage / "owner2108.json", OWNER2108_SHA)
    require(len(read_exact(Path(NATIVE), NATIVE_SHA, 17760)) == 17760, "NATIVE_SIZE")
    runtime = stage / "runtime"
    manifest = json.loads(read_exact(runtime / "RUNTIME_FILES.json", RUNTIME_SHA))
    for name, row in manifest["files"].items():
        require(len(read_exact(runtime / name, row["sha256"])) == row["bytes"], "RUNTIME_MEMBER_SIZE")
    return digest(launcher)


def prepare_pre_locked(held, collect):
    require_lease(held)
    read_exact(Path(STAGE / "node_inventory.py"), INVENTORY_SHA)
    require(not any(os.path.lexists(path) for path in (PRE, RUN, POST)), "NAMESPACE_ALREADY_USED_RECONCILE")
    scheduler = collect()
    records, active = scheduler_scope(scheduler, time.time(), os.getuid())
    previous = old_standby(records)
    names = (PRE_LABEL, LABEL, POST_LABEL)
    require(not active and not any(row[1] in names for row in records)
            and not any(row["Name"] in names for row in scheduler["numbered_steps"]), "TARGET_BUSY_OR_ROLE_USED")
    argv = render_inventory("pre")
    path = Path(PRE)
    path.mkdir(mode=0o700)
    intent = {"request": PRE_REQUEST, "gpu_request": REQUEST, "predecessor_accounting": previous,
              "predecessor_terminal_observed_epoch": scheduler["epoch"], "argv_SHA256": digest(encoded(argv)),
              "replay_allowed": False}
    intent_sha = save_new(path / "intent.json", intent)
    return intent, intent_sha, argv


def prepare_gpu_locked(held, collect, launcher_sha):
    """Collect actual evidence while locked, persist the original attempt, and return argv once."""
    require_lease(held)
    stage_pins(launcher_sha)
    run = Path(RUN)
    require(not os.path.lexists(run) and not os.path.lexists(POST), "NAMESPACE_ALREADY_USED_RECONCILE")
    evidence = collect()
    pre_proof = evidence["independent_pre"]
    pre_intent = json.loads(read_exact(Path(PRE / "intent.json"), pre_proof["intent_SHA256"]))
    require(pre_intent["request"] == PRE_REQUEST and pre_intent["gpu_request"] == REQUEST
            and pre_intent["argv_SHA256"] == digest(encoded(render_inventory("pre")))
            and pre_intent["predecessor_accounting"] == old_standby(rows(evidence["scheduler"]))
            and pre_intent["predecessor_terminal_observed_epoch"] == evidence["predecessor_terminal_observed_epoch"],
            "PRE_INTENT_OR_OWNERSHIP_DIFFERS")
    scope = validate_admission(evidence, time.time(), os.getuid())
    run.mkdir(mode=0o700)
    scope_sha = save_new(run / "scope.json", scope)
    admission_sha = save_new(run / "admission.json", evidence)
    argv = render_gpu(scope, scope_sha, ATTESTED_SHA, launcher_sha)
    save_new(run / "gpu-attempt.json", {"request": REQUEST, "label": LABEL, "scope_SHA256": scope_sha,
                                       "admission_SHA256": admission_sha, "argv_SHA256": digest(encoded(argv)),
                                       "epoch": scope["epoch"], "deadline_epoch": scope["deadline_epoch"],
                                       "state": "ORIGINAL_ATTEMPT_RESERVED", "replay_allowed": False})
    return scope, argv


def prepare_post_locked(held, collect, scope, *, attested=None):
    require_lease(held)
    read_exact(Path(STAGE / "node_inventory.py"), INVENTORY_SHA)
    require(not os.path.lexists(POST), "POST_NAMESPACE_ALREADY_USED_RECONCILE")
    scope_sha = digest(encoded(scope))
    read_exact(Path(RUN / "scope.json"), scope_sha)
    intent_path = Path(RUN / "gpu-attempt.json")
    require(intent_path.is_file() and not intent_path.is_symlink() and intent_path.stat().st_size < 4096,
            "ORIGINAL_GPU_ATTEMPT_REQUIRED")
    attempt = json.loads(intent_path.read_bytes())
    require(attempt["request"] == REQUEST and attempt["scope_SHA256"] == scope_sha
            and attempt["epoch"] == scope["epoch"] and attempt["deadline_epoch"] == scope["deadline_epoch"]
            and attempt["state"] == "ORIGINAL_ATTEMPT_RESERVED" and attempt["replay_allowed"] is False,
            "ORIGINAL_GPU_ATTEMPT_DIFFERS")
    scheduler = collect()
    row, argv = post_after_terminal(scheduler, scope, time.time(), os.getuid(), attested=attested)
    path = Path(POST)
    path.mkdir(mode=0o700)
    save_new(path / "intent.json", {"request": POST_REQUEST, "gpu_request": REQUEST, "gpu_accounting": row,
                                   "gpu_terminal_observed_epoch": scheduler["epoch"], "argv_SHA256": digest(encoded(argv)),
                                   "replay_allowed": False})
    return row, argv


def refill_eligibility_locked(held, collect, scope, evidence):
    require_lease(held)
    post_proof = evidence["independent_post"]
    intent = json.loads(read_exact(Path(POST / "intent.json"), post_proof["intent_SHA256"]))
    require(intent["request"] == POST_REQUEST and intent["gpu_request"] == REQUEST
            and intent["argv_SHA256"] == digest(encoded(render_inventory("post")))
            and intent["gpu_accounting"] == evidence["adopted_gpu_accounting"]
            and intent["gpu_terminal_observed_epoch"] == evidence["gpu_terminal_observed_epoch"],
            "ORIGINAL_POST_INTENT_DIFFERS")
    scheduler = collect()
    return refill_after_clean_post(scheduler, scope, time.time(), os.getuid(), evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--evidence-sha256", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--uid", required=True, type=int)
    args = parser.parse_args()
    raw = read_exact(args.evidence, args.evidence_sha256, 262144)
    scope = validate_admission(json.loads(raw), time.time(), args.uid)
    argv = render_gpu(scope, digest(encoded(scope)), ATTESTED_SHA, args.launcher_sha256)
    print(json.dumps({"validation_only": True, "GPU_admitted": False, "scope_preview": scope, "argv_preview": argv}))


if __name__ == "__main__":
    main()
