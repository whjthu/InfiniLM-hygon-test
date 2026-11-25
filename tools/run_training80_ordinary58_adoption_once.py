"""Bind the actual local handoff and the fixed vNext ledger for ordinary58."""

import argparse
from contextlib import closing
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import sqlite3
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "local-only"
MODULE = ROOT / "code/bw1000/training80-ordinary-refill-vnext"
COORDINATOR = MODULE / "coordinator.py"
COORDINATOR_SHA = "bb1e1203dd92070ed782807e05df8a5b6c17c31eb67c7d5b5141dbec16092961"
CALENDAR_MODULE_SHA = "f3c09fe2afb75701398a469f8bda7d0a302ec3dbc6d6579ee041a2f928ce770e"
CALENDAR = LOCAL / "training80-0544-ordinary-58-node-verified-standby-v1.json"
CALENDAR_SHA = "56da79542a9bb1181385e66f68ac9523f32c323b3cd31cb73a530283a329c83f"
PLAN = LOCAL / "training80-vnext-20261125-0841-ordinary58-group-request-plan.json"
PLAN_SHA = "96908d6b7b25e75e57a6756a12da273fe252a15e1935f18d6eace6c5d403152d"
ROSTER_SHA = "a9d84c05150dc61f1751e904cceb99c05672e7dfbefa9981fde683aff14b4c5e"
WAVE = "training80-vnext-20261125-0841-ordinary58"
STEM = "training80-vnext-0841-ordinary58"
OLD_OWNER = LOCAL / "training80-vnext-0841-legacy-owner-exit-20261125-v1.json"
OLD_OWNER_SHA = "9d37a77db68c2162170aa871da58daa146dd703b6bb5b9e7152a829dc562c1c7"
HISTORY = LOCAL / "training80-vnext-0841-ordinary58-local-validation-final-20261125.json"
HISTORY_SHA = "fd3f7aec2b60e5efe544a3aa24bb9a659091a1731edd63a95cf73bfa8f86aa63"
HANDOFF = LOCAL / (STEM + "-operator-handoff-v1.json")
INTENT = LOCAL / (STEM + "-adoption-intent-v1.json")
ADOPTION = LOCAL / (STEM + "-sole-adoption-v1.json")
LEDGER = LOCAL / "training80-ordinary-refill-vnext/claim-ledger-v1.sqlite3"
HOUR_REQUEST = "0e1a294fabfc4854854c09ed6b4d6f83"
HOUR_STEPS = {"1103606": "1103606.3088", "1194900": "1194900.1182"}
HOUR_READY = "44e1d5737c57f7f126993971da76d37c6c1d80103ca52339cdb148fb5d2cab63"
HOUR_INSTALL = "fdaf7cd6990707735315d9d1b22e9c389a10855531866f04e7a81eecadadeab6"
HOUR_DIR = LOCAL / ("training80-fixed-hour-transport-" + HOUR_REQUEST)
OLD_PROCESS_NAMES = (
    "run_training80_0544_ordinary_refill_once.py", "run-training80-0544-ordinary-refill-once.py",
    "run_training80_2347_ordinary_refill_once.py", "run_standby_renewal_with_pinned_publication.py",
    "run-dcu-standby-cycle-20261123.ps1",
)
TERMINAL = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def hex_pin(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def read_path(path, pin, limit=262144):
    require(path.is_file() and path.resolve(strict=True) == path, "EXACT_CANONICAL_ARTIFACT_REQUIRED")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and (pin is None or (hex_pin(pin) and digest(raw) == pin)), "PINNED_ARTIFACT_CHANGED")
    return raw


def load_json(path, pin, limit=262144):
    return json.loads(read_path(path, pin, limit))


def checked_plan():
    read_path(COORDINATOR, COORDINATOR_SHA)
    read_path(MODULE / "calendar58.py", CALENDAR_MODULE_SHA)
    sys.path.insert(0, str(MODULE))
    api = runpy.run_path(str(COORDINATOR))
    original = api["legacy"]()
    _, scope = api["calendar_scope"](original, CALENDAR, CALENDAR_SHA)
    plan = load_json(PLAN, PLAN_SHA)
    require(plan["wave"] == WAVE and scope["wave"] == WAVE and plan["group_roster_SHA256"] == scope["group_roster_SHA256"] == ROSTER_SHA
            and plan["calendar_SHA256"] == scope["calendar_SHA256"] == CALENDAR_SHA
            and plan["state"] == "PREPARED_REQUEST_IDS_ONLY_NOT_ADOPTED" and plan["initialized_ledger"] is False
            and plan["operator_adoption_receipt"] is None and plan["actual_submissions"] == 0,
            "PREPARED_EXACT_REQUEST_PLAN_REQUIRED")
    require(scope["ordinary_candidates"] == plan["ordinary_nodes"] == 58 and scope["excluded_nodes"] == plan["excluded_nodes"] == 22
            and scope["classification"] == {"PRE_BRIDGE": 7, "BRIDGE_UNKNOWN": 3, "APPROVAL_EXCLUDED": 2,
                    "EXISTING_PROTECTION": 5, "SINGLE_NODE_SCOPE": 5, "VERIFIED_ORDINARY_SUCCESSOR": 58},
            "KEEP_ORIGINAL_22_EXCLUSIONS")
    requests = set()
    require(len(plan["groups"]) == len(scope["groups"]) == 16 and LEDGER == api["AUDIT_ROOT"] / "claim-ledger-v1.sqlite3",
            "FIXED_LEDGER_AND_16_GROUPS_REQUIRED")
    for candidate, verified in zip(plan["groups"], scope["groups"], strict=True):
        row = dict(candidate)
        request = row.pop("request")
        require(row.pop("submission_attempts") == 0 and row == verified and api["hex_id"](request, 32) and request not in requests,
                "EXACT_GROUP_REQUEST_BINDING_CHANGED")
        requests.add(request)
    return api, scope, plan


def old_owner_receipt():
    info = load_json(OLD_OWNER, OLD_OWNER_SHA)
    census = info["local_process_inventory"]
    require(info["prior_wave"] == "training80-0544-ordinary" and info["prior_owner"] == "/root/standby_resource_monitor"
            and info["handed_to"] == "/root" and info["previous_ordinary_controller_released"] is True
            and info["original_wrapper_sessions_remaining"] == 0 and info["old_group_replay_allowed"] is False
            and info["original_58_result_SHA256"] == CALENDAR_SHA and info["vnext_adopted_by_this_record"] is False
            and census["provider"] == "Win32_Process" and census["read_rc"] == 0 and census["matching_processes"] == []
            and census["matching_exact_launcher_names"] == list(OLD_PROCESS_NAMES), "FORMAL_PRIOR_OWNER_EXIT_REQUIRED")
    return info


def hour_release(document):
    report = document.get("response") or {}
    require(document.get("rc") == 0 and document.get("request") == report.get("request") == HOUR_REQUEST
            and document.get("replay_allowed") is False and report.get("replay_allowed") is False
            and report.get("state") == "ACTUAL_TERMINAL_AND_INDEPENDENT_MEASUREMENT"
            and report.get("released_steps") == HOUR_STEPS and report.get("ready_SHA256") == HOUR_READY
            and report.get("install_SHA256") == HOUR_INSTALL and hex_pin(report.get("original_host_outcome_SHA256")),
            "FIXED_HOUR_INDEPENDENT_RELEASE_REQUIRED")
    require(set(report["accounting"]) == set(HOUR_STEPS), "BOTH_ORIGINAL_HOUR_ACCOUNTING_ROWS_REQUIRED")
    for parent, step in HOUR_STEPS.items():
        row = report["accounting"][parent]
        require(isinstance(row, list) and len(row) == 7 and row[:2] == [step, f"training80-long-busy-{HOUR_REQUEST}-{parent}"]
                and isinstance(row[2], str) and bool(row[2].split()) and row[2].split()[0] in TERMINAL
                and row[5] and row[6] in {"cpu=40,mem=10G,node=40", "cpu=40,mem=10240M,node=40"},
                "EXACT_TERMINAL_ZERO_GRES_OBSERVER_REQUIRED")
    return report["released_steps"]


def local_census():
    require(os.name == "nt", "LOCAL_WIN32_PROCESS_AUTHORITY_REQUIRED")
    names = OLD_PROCESS_NAMES + ("training80-ordinary-refill-vnext/coordinator.py",)
    literals = ",".join("'" + value + "'" for value in names)
    command = "$ErrorActionPreference='Stop'; $needles=" + literals + r""";
$matches = @(Get-CimInstance -ClassName Win32_Process | Where-Object {
    if($_.ProcessId -eq $PID -or -not $_.CommandLine){ return $false }
    $line=$_.CommandLine.Replace('\','/')
    @($needles | Where-Object { $line.IndexOf($_,[StringComparison]::OrdinalIgnoreCase) -ge 0 }).Count -gt 0
} | Select-Object ProcessId,ParentProcessId,Name)
[pscustomobject]@{read_rc=0; matches=$matches} | ConvertTo-Json -Depth 3 -Compress
"""
    exe = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    result = subprocess.run([str(exe), "-NoProfile", "-NonInteractive", "-Command", command],
                            stdin=subprocess.DEVNULL, capture_output=True, timeout=18)
    require(result.returncode == 0 and 0 < len(result.stdout) <= 8192, "FRESH_LOCAL_PROCESS_INVENTORY_UNAVAILABLE")
    census = json.loads(result.stdout)
    require(census == {"read_rc": 0, "matches": []}, "ORIGINAL_ORDINARY_OR_STANDBY_LAUNCHER_STILL_PRESENT")
    return dict(provider="Win32_Process", matching_processes=0, local_epoch=time.time())


def ledger_identity(path):
    require(path.is_file() and path.resolve(strict=True) == path and not path.is_symlink(), "SAME_EXISTING_VNEXT_LEDGER_REQUIRED")
    sidecars = [path.with_name(path.name + ending) for ending in ("-wal", "-shm", "-journal")]
    require(not any(map(file_present, sidecars)), "ACTIVE_OR_UNRECOVERED_LEDGER_REQUIRES_RECONCILIATION")
    before = path.stat()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=5)) as db:
        db.execute("PRAGMA query_only=ON")
        require(db.execute("PRAGMA user_version").fetchone() == (1,)
                and db.execute("PRAGMA quick_check(1)").fetchone() == ("ok",), "FIXED_LEDGER_SCHEMA_OR_INTEGRITY_CHANGED")
        require({row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {"claims", "targets", "events"},
                "FIXED_LEDGER_SCHEMA_OR_INTEGRITY_CHANGED")
        claims, = db.execute("SELECT count(*) FROM claims").fetchone()
        wave_claims, = db.execute("SELECT count(*) FROM claims WHERE wave=?", (WAVE,)).fetchone()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) and after.st_ino > 0
            and not any(map(file_present, sidecars)),
            "LEDGER_CHANGED_DURING_READ")
    return dict(device=after.st_dev, inode=after.st_ino, claims=claims, wave_claims=wave_claims)


def file_present(path):
    return path.exists() or path.is_symlink()


def initial_policy(custody, existing, occupied):
    require(not occupied, "ADOPTION_ALREADY_STARTED_OR_PRESENT_RECONCILE")
    if existing is None:
        require(custody == {
            "state": "NEVER_INITIALIZED_AND_NEVER_ADOPTED", "history_confirmed_by": "/root",
            "historical_local_validation_SHA256": HISTORY_SHA, "fixed_ledger_absolute": LEDGER.as_posix(),
            "affirmative_owner_history_reviewed": True}, "ABSENCE_IS_NOT_PROOF_OF_NEVER_ADOPTED")
        return "INITIALIZE_FIXED_LEDGER_ONCE"
    require(custody == {
        "state": "REUSE_EXISTING_FIXED_LEDGER", "history_confirmed_by": "/root", "fixed_ledger_absolute": LEDGER.as_posix(),
        "ledger_device": existing["device"], "ledger_inode": existing["inode"], "current_wave_never_adopted": True},
        "EXISTING_LEDGER_CUSTODY_OR_INODE_DIFFERS")
    require(existing["wave_claims"] == 0, "EXISTING_WAVE_CLAIMS_MUST_RECONCILE")
    return "REUSE_FIXED_LEDGER"


def handoff(pin):
    require(hex_pin(pin), "ACTUAL_ROOT_HANDOFF_PIN_REQUIRED")
    old = old_owner_receipt()
    doc = load_json(HANDOFF, pin, 16384)
    require(doc.get("schema") == "training80.ordinary.vnext-operator-handoff.v1" and doc.get("owner") == "/root"
            and doc.get("current_sole_coordinator") in {"/root", "/root/standby_resource_monitor"}
            and doc.get("previous_ordinary_controller_released") is True and doc.get("sole_current_coordinator_authorized") is True
            and doc.get("wave") == WAVE and doc.get("calendar_SHA256") == CALENDAR_SHA and doc.get("group_roster_SHA256") == ROSTER_SHA
            and doc.get("group_request_plan_SHA256") == PLAN_SHA and doc.get("legacy_owner_exit_SHA256") == OLD_OWNER_SHA,
            "EXACT_ROOT_SINGLE_COORDINATOR_HANDOFF_REQUIRED")
    issued = datetime.fromisoformat(doc["issued_Beijing"])
    cutoff = datetime.fromisoformat(old["observed_Beijing"])
    require(issued.tzinfo is not None and issued >= cutoff and issued.timestamp() <= time.time(),
            "CURRENT_FORMAL_HANDOFF_REQUIRED")
    observation = doc["observer_terminal"]
    name = observation["file"]
    require(re.fullmatch(r"independent-terminal-[a-f0-9]{32}\.json", name) and hex_pin(observation["SHA256"]),
            "EXACT_FIXED_HOUR_TERMINAL_ARTIFACT_REQUIRED")
    released = hour_release(load_json(HOUR_DIR / name, observation["SHA256"], 32768))
    history = load_json(HISTORY, HISTORY_SHA)
    require(history["production_adoption_created"] is False and history["production_ledger_changed"] is False,
            "PRIOR_LOCAL_VALIDATION_RECORD_CHANGED")
    existing = ledger_identity(LEDGER) if file_present(LEDGER) else None
    if existing is None and file_present(LEDGER.parent):
        require(LEDGER.parent.is_dir() and not LEDGER.parent.is_symlink()
                and not any(LEDGER.parent.iterdir()), "NONEMPTY_FIXED_AUDIT_DIRECTORY_MUST_RECONCILE")
    choice = initial_policy(doc["ledger_custody"], existing, file_present(INTENT) or file_present(ADOPTION))
    return dict(wave=WAVE, custody_action=choice, ledger_existing_identity=existing, handoff_SHA256=pin,
                sole_current_coordinator=doc["current_sole_coordinator"], independent_hour_release=released,
                observer_terminal_file=name, observer_terminal_SHA256=observation["SHA256"], original_ordinary_owner_exit_SHA256=OLD_OWNER_SHA,
                process_census=local_census())


def exclusive_save(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return digest(raw)


def production_argv(*args):
    return [sys.executable, "-B", "-X", "utf8", str(COORDINATOR), *args]


def dispatch_argv(api, scope, plan, group, adoption_sha):
    require(type(group) is int and 1 <= group <= 16, "EXACT_GROUP_REQUIRED")
    api["calendar58"].require_adoption(ADOPTION, adoption_sha, WAVE, LEDGER, scope, LOCAL)
    intent = load_json(INTENT, None, 16384)
    require(intent.get("schema") == "training80.ordinary.vnext-adoption-intent.v1" and intent.get("wave") == WAVE
            and intent.get("request_plan_SHA256") == PLAN_SHA and intent.get("roster_SHA256") == ROSTER_SHA
            and intent.get("original_ordinary_owner_exit_SHA256") == OLD_OWNER_SHA
            and intent.get("independent_hour_release") == HOUR_STEPS and intent.get("replay_allowed") is False,
            "EXACT_ONE_SHOT_ADOPTION_INTENT_REQUIRED")
    name = intent["observer_terminal_file"]
    require(re.fullmatch(r"independent-terminal-[a-f0-9]{32}\.json", name) and hex_pin(intent["observer_terminal_SHA256"]),
            "EXACT_FIXED_HOUR_TERMINAL_ARTIFACT_REQUIRED")
    hour_release(load_json(HOUR_DIR / name, intent["observer_terminal_SHA256"], 32768))
    request = plan["groups"][group - 1]["request"]
    return production_argv("dispatch", "--calendar", str(CALENDAR), "--calendar-sha", CALENDAR_SHA,
                           "--wave", WAVE, "--group", str(group), "--request", request, "--adoption", str(ADOPTION),
                           "--adoption-sha", adoption_sha)


def activate(api, scope, pin):
    evidence = handoff(pin)
    exclusive_save(INTENT, dict(schema="training80.ordinary.vnext-adoption-intent.v1", **evidence,
                               request_plan_SHA256=PLAN_SHA, roster_SHA256=ROSTER_SHA, replay_allowed=False))
    created = evidence["custody_action"] == "INITIALIZE_FIXED_LEDGER_ONCE"
    if created:
        require(not file_present(LEDGER), "FIXED_LEDGER_APPEARED_AFTER_PINNED_HANDOFF")
        result = subprocess.run(production_argv("init-ledger"), cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, timeout=15)
        require(result.returncode == 0 and 0 < len(result.stdout) <= 1024 and json.loads(result.stdout) == {
                    "state": "EXPLICIT_NEW_LEDGER_CREATED", "replay_allowed": False}, "CANONICAL_INIT_UNKNOWN_RECONCILE")
    real = ledger_identity(LEDGER)
    require(real["wave_claims"] == 0 and (real["claims"] == 0 if created else real == evidence["ledger_existing_identity"]),
            "FIXED_LEDGER_CHANGED_AFTER_AUTHORITY_CHECK")
    data = dict(schema="training80.ordinary.vnext-sole-adoption.v1", wave=WAVE, calendar_SHA256=CALENDAR_SHA,
                group_roster_SHA256=ROSTER_SHA, ledger_absolute=LEDGER.as_posix(), ledger_device=real["device"],
                ledger_inode=real["inode"], previous_ordinary_controller_released=True, sole_current_coordinator_authorized=True)
    pin = exclusive_save(ADOPTION, data)
    api["calendar58"].require_adoption(ADOPTION, pin, WAVE, LEDGER, scope, LOCAL)
    return dict(state="ADOPTION_SEALED", adoption=ADOPTION.as_posix(), adoption_SHA256=pin, initialized_ledger=created,
                ledger_device=real["device"], ledger_inode=real["inode"], fresh_claims=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "check-handoff", "activate", "show-dispatch-argv"))
    parser.add_argument("--handoff-sha")
    parser.add_argument("--adoption-sha")
    parser.add_argument("--group", type=int)
    args = parser.parse_args()
    api, scope, plan = checked_plan()
    if args.mode == "check":
        old_owner_receipt()
        result = dict(state="PREPARED_ONLY", wave=WAVE, groups=scope["groups"], roster_SHA256=ROSTER_SHA,
                      request_plan_SHA256=PLAN_SHA, candidates=58, excluded=22, first_preparation_Beijing=scope["first_preparation_Beijing"],
                      ledger_exists=file_present(LEDGER), handoff_exists=file_present(HANDOFF), adoption_exists=file_present(ADOPTION),
                      absence_proves_history=False, adoption_created_in_this_call=False, initialized_ledger_in_this_call=False)
    elif args.mode == "show-dispatch-argv":
        result = dict(state="ARGV_ONLY", command=dispatch_argv(api, scope, plan, args.group, args.adoption_sha), executed=False)
    elif args.mode == "activate":
        result = activate(api, scope, args.handoff_sha)
    else:
        result = dict(state="HANDOFF_VALIDATED_ONLY", **handoff(args.handoff_sha))
    print(json.dumps(dict(result, SCNet_calls=0, Git_calls=0, scheduled_steps=0, GPU_requests=0), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, IndexError, sqlite3.Error, subprocess.SubprocessError) as error:
        reason = str(error) if type(error) is ValueError and re.fullmatch(r"[A-Z0-9_]+", str(error)) else type(error).__name__
        print(json.dumps(dict(state="REFUSED_OR_RECONCILE", reason=reason)))
        raise SystemExit(1)
