"""One exact O01 action per invocation. Unknown submissions retain their original intent."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_scope as spec
import pending
import scheduler


GUARDED_BRIDGE_SHA = "785ed7e948c0484243c7c325eb470d4555b6304e24759667f7b1289f9d67676f"


def bounded(path, cap=16384):
    spec.require(path.is_file() and not path.is_symlink(), "EXPECTED_REGULAR_FILE")
    with path.open("rb") as file:
        data = file.read(cap + 1)
    spec.require(0 < len(data) <= cap, "UNEXPECTED_FILE_SIZE")
    return data


def json_file(path, cap=16384):
    return json.loads(bounded(path, cap))


def safe_environment():
    # This HOST process starts exact explicit steps in the owned parent allocation.
    return {name: value for name, value in os.environ.items()
            if not name.startswith(("SLURM_", "SBATCH_", "SRUN_"))}


def marker(expected):
    spec.require(spec.SHA.fullmatch(expected), "RESERVATION_SHA_REQUIRED")
    spec.require(pending.marker_document(Path(spec.RESULTS)) == expected,
                 "DURABLE_RESERVATION_DIFFERS")


def intent(phase):
    return Path((spec.PRE if phase == "pre" else spec.POST) / "intent.json")


def inventory_proof(phase):
    path = intent(phase)
    document = json_file(path)
    observation_path = path.with_name("submit-observed.json")
    observation = json_file(observation_path)
    raw = bounded(path.with_name("slurm.stdout"))
    role = spec.PRE_REQUEST if phase == "pre" else spec.POST_REQUEST
    spec.require(observation["request"] == document["request"] == role and observation["srun_rc"] == 0
                 and observation["stdout_SHA256"] == sha256(raw).hexdigest()
                 and observation["intent_SHA256"] == sha256(bounded(path)).hexdigest(),
                 "ORIGINAL_INVENTORY_OBSERVATION_DIFFERS")
    return {"request": role, "source_SHA256": spec.INVENTORY_SHA, "stdout": raw.decode("utf8"),
            "stdout_SHA256": sha256(raw).hexdigest(), "intent_SHA256": observation["intent_SHA256"],
            "start_epoch": observation["start_epoch"], "observed_epoch": observation["observed_epoch"]}


def run_inventory(phase, argv):
    path = intent(phase)
    role = json_file(path)["request"]
    start = time.time()
    with path.with_name("launcher.stderr").open("xb") as error:
        try:
            result = subprocess.run(argv, env=safe_environment(), stdout=subprocess.DEVNULL,
                                    stderr=error, stdin=subprocess.DEVNULL, timeout=28, check=False)
        except subprocess.TimeoutExpired:
            spec.save_new(path.with_name("submit-observed.json"), {
                "request": role, "start_epoch": start, "observed_epoch": time.time(),
                "intent_SHA256": sha256(bounded(path)).hexdigest(), "srun_rc": None,
                "remote_outcome": "UNKNOWN", "replay_allowed": False})
            raise
    end = time.time()
    output = path.with_name("slurm.stdout")
    pin = sha256(bounded(output)).hexdigest() if output.is_file() and output.stat().st_size else None
    evidence = {"request": role, "start_epoch": start, "observed_epoch": end,
                "intent_SHA256": sha256(bounded(path)).hexdigest(), "srun_rc": result.returncode,
                "stdout_SHA256": pin, "remote_outcome": "OBSERVED_LOCAL_LAUNCHER_RETURN",
                "replay_allowed": False}
    spec.save_new(path.with_name("submit-observed.json"), evidence)
    evidence["independent_scheduler"] = relevant(scheduler.collect(), role)
    return evidence


def relevant(snapshot, phase_request=None):
    labels = {spec.OLD_LABEL, spec.PRE_LABEL, spec.LABEL, spec.POST_LABEL}
    return {"epoch": snapshot["epoch"], "request": phase_request,
            "accounting": [row for row in snapshot["accounting_rows"] if row[1] in labels],
            "numbered_target_steps": [row for row in snapshot["numbered_steps"] if spec.NODE in row["nodes"]],
            "target_configured": snapshot["target_configured"], "actual_parent_nodes": len(snapshot["parent_nodes"])}


def gpu_evidence(boot):
    pre = inventory_proof("pre")
    return {"request": spec.REQUEST, "label": spec.LABEL, "transport_disposition": "NEVER_ATTEMPTED",
            "independent_pre": pre, "boot_id_SHA256": boot,
            "predecessor_terminal_observed_epoch": json_file(intent("pre"))["predecessor_terminal_observed_epoch"],
            "scheduler": scheduler.collect()}


def run_gpu(scope, argv):
    path = Path(spec.RUN)
    allowed_until = min(time.monotonic() + 16, time.monotonic() + scope["deadline_epoch"] - time.time() - 185)
    spec.require(allowed_until - time.monotonic() >= 4, "INSUFFICIENT_ORIGINAL_GPU_EPOCH")
    with (path / "launcher.stderr").open("xb") as error:
        proc = subprocess.Popen(argv, env=safe_environment(), stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=error, start_new_session=True)
    spec.save_new(path / "launcher-pid.json", {"request": spec.REQUEST, "Popen_PID": proc.pid,
                                              "start_epoch": time.time(), "replay_allowed": False})
    while time.monotonic() < allowed_until:
        steps = scheduler.command("scontrol", "-o", "show", "step", spec.PARENT)
        matching = []
        for line in steps.splitlines():
            if not line.strip():
                continue
            row = scheduler.single(line, "StepId")
            if row.get("Name") == spec.LABEL:
                matching.append(row)
        spec.require(len(matching) <= 1, "O01_NUMBERED_STEP_DUPLICATE")
        if matching:
            match = matching[0]
            spec.require(spec.STEP.fullmatch(match["StepId"]) and match["StepId"] != spec.OLD_STEP
                         and match["NodeList"] == spec.NODE, "O01_NUMBERED_STEP_PLACEMENT")
            spec.check_resources(match["TRES"], 12, 8, {"24G", "24576M"})
            result = {"remote_outcome": "NUMBERED_STEP_OBSERVED", "step": match["StepId"],
                      "state": match["State"], "TRES": match["TRES"]}
            break
        rc = proc.poll()
        if rc is not None:
            result = {"remote_outcome": "UNKNOWN", "srun_rc": rc}
            break
        time.sleep(min(0.75, max(0, allowed_until - time.monotonic())))
    else:
        result = {"remote_outcome": "UNKNOWN", "srun_rc": proc.poll()}
    result.update(request=spec.REQUEST, scope_SHA256=spec.digest(spec.encoded(scope)),
                  observed_epoch=time.time(), replay_allowed=False)
    spec.save_new(path / "launcher-observation.json", result)
    return result


def adopt_owner_if_materialized():
    path = Path(spec.RUN)
    host = path / "attested-host-owner.json"
    container = path / "attested-container-owner.json"
    if host.is_file() and container.is_file():
        scope = json_file(path / "scope.json")
        return spec.adopt_container_owner(bounded(host), bounded(container), scope)
    return None


def terminal_summary(owner):
    path = Path(spec.RUN / "supervised/terminal.json")
    if not path.is_file():
        return None
    raw = bounded(path, 262144)
    record = json.loads(raw)
    scope = bounded(Path(spec.RUN / "scope.json"))
    spec.require(owner is not None and record["request"] == spec.REQUEST and record["step"] == owner["step"]
                 and record["scope_SHA256"] == sha256(scope).hexdigest() and record["runtime_SHA256"] == spec.RUNTIME_SHA
                 and record["formal_performance"] is False, "SUPERVISOR_TERMINAL_IDENTITY")
    witness = record.get("witness", {})
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest(),
            "local_process_cleanup": record["local_process_cleanup"],
            "restored_eight_proved": record["restored_eight_proved"],
            "qualified_operator_route": witness.get("qualified_operator_route"),
            "supervisor_error_type": record.get("error", {}).get("type"),
            "formal_performance": False}


def grant(held, scope, refill_request):
    path = intent("post")
    doc = json_file(path)
    evidence = {"independent_post": inventory_proof("post"), "adopted_gpu_accounting": doc["gpu_accounting"],
                "gpu_terminal_observed_epoch": doc["gpu_terminal_observed_epoch"]}
    return pending.authorize_refill_locked(Path(spec.RESULTS), held, Path(spec.STAGE / "host_scope.py"),
                                           scheduler.collect, scope, evidence, refill_request)


def start_gpu(held, boot):
    spec.require(spec.SHA.fullmatch(boot), "EXACT_TARGET_BOOT_SHA_REQUIRED")
    scope, argv = spec.prepare_gpu_locked(held, lambda: gpu_evidence(boot), spec.DEVICE_HOST_SHA)
    return run_gpu(scope, argv)


def action(args):
    spec.stage_pins(spec.DEVICE_HOST_SHA)
    spec.read_exact(Path(spec.STAGE / "guarded-standby-bridge.py"), GUARDED_BRIDGE_SHA)
    if args.phase == "observe":
        snapshot = scheduler.collect()
        spec.scheduler_scope(snapshot, time.time(), os.getuid())
        guard = Path(spec.RESULTS / pending.MARKER)
        owner = adopt_owner_if_materialized()
        return relevant(snapshot) | {
            "observed_attested_owner": owner,
            "supervisor_terminal": terminal_summary(owner),
            "reservation_SHA256": pending.marker_document(Path(spec.RESULTS)) if pending.present(guard) else None,
        }
    with spec.standby_lock() as held:
        if args.phase == "reserve":
            snapshot = scheduler.collect()
            records, active = spec.scheduler_scope(snapshot, time.time(), os.getuid())
            previous = spec.exact_row(records, spec.OLD_LABEL)
            spec.check_resources(previous[6], 8, 8, {"16G", "16384M"})
            labels = (spec.PRE_LABEL, spec.LABEL, spec.POST_LABEL)
            spec.require(previous[0] == spec.OLD_STEP and previous[2] in ("RUNNING", "COMPLETED")
                         and previous[3] == "0:0" and all(row["StepId"] == spec.OLD_STEP for row in active)
                         and not any(row[1] in labels for row in records)
                         and not any(row["Name"] in labels for row in snapshot["numbered_steps"]),
                         "ORIGINAL_RESERVATION_COLLISION")
            pin = pending.reserve_locked(Path(spec.RESULTS), held)
            return relevant(snapshot) | {"reservation_SHA256": pin, "replay_allowed": False}
        marker(args.reservation_SHA256)
        if args.phase in {"pre", "handoff"}:
            if args.phase == "handoff":
                spec.require(spec.SHA.fullmatch(args.boot_SHA256), "EXACT_TARGET_BOOT_SHA_REQUIRED")
            _, _, argv = spec.prepare_pre_locked(held, scheduler.collect)
            evidence = run_inventory("pre", argv)
            if args.phase == "pre":
                return evidence
            return {"fresh_pre_observation": evidence, "actual_gpu_submission": start_gpu(held, args.boot_SHA256)}
        if args.phase == "gpu":
            return start_gpu(held, args.boot_SHA256)
        scope = json_file(Path(spec.RUN / "scope.json"))
        spec.require(pending.refill_request(args.refill_request), "DISTINCT_REFILL_REQUEST_REQUIRED")
        if args.phase == "grant":
            return {"refill_request": args.refill_request, "grant_SHA256": grant(held, scope, args.refill_request)}
        private = Path(spec.RUN / "attested-host-owner.json")
        attested = json_file(private) if private.is_file() else None
        _, argv = spec.prepare_post_locked(held, scheduler.collect, scope, attested=attested)
        result = run_inventory("post", argv)
        result.update(refill_request=args.refill_request, grant_SHA256=grant(held, scope, args.refill_request))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("observe", "reserve", "pre", "handoff", "gpu", "post", "grant"))
    parser.add_argument("--reservation-SHA256", default="")
    parser.add_argument("--boot-SHA256", default="")
    parser.add_argument("--refill-request", default="")
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(action(args), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
