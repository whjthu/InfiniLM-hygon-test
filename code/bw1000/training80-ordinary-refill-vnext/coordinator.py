"""Ordinary refill coordinator with atomic candidate claims and one durable bridge boundary."""

from __future__ import annotations

import argparse
import ast
from contextlib import closing, contextmanager
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import sqlite3
import subprocess
import time
from types import FunctionType

import calendar58


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tools/run_training80_0544_ordinary_refill_once.py"
SOURCE_SHA = "2fb4649087e1e2162f706fe61b095cc434c120758bff4e60f843ebc973de7a0b"
PUBLICATION_SHA = "e900334ac5be3a74fbde60f5c99457706dd9e802b7f749f0c375a1ecca48424e"
RUNNER_SHA = "04f2bef937f98b35a29c0bfce7c199e56e049e001a0a5b92ecbf12878a358c73"
OLD_CALENDAR_SHA = "0730b099d674de7b195d825f5f81c9a12b656e50d79e4e2c1117c1156828863e"
EVIDENCE = ROOT / "results/bw1000/2026-11-25/training80-refill-admission-and-bridge-255-independent-classification.json"
EVIDENCE_SHA = "e2d9019aa0f510535b75608393583f01aeab6eb8ea684ab7b617c1c97b9477b8"
AUDIT_ROOT = ROOT / "local-only/training80-ordinary-refill-vnext"
SCHEMA = """
CREATE TABLE claims(
  request TEXT PRIMARY KEY, wave TEXT NOT NULL, group_id INTEGER NOT NULL,
  calendar_sha TEXT NOT NULL, publication TEXT NOT NULL, observation_sha TEXT NOT NULL,
  roster_count INTEGER NOT NULL, epoch REAL NOT NULL,
  UNIQUE(wave, group_id, calendar_sha)
);
CREATE TABLE targets(
  parent TEXT NOT NULL, node TEXT NOT NULL, predecessor TEXT NOT NULL,
  request TEXT NOT NULL REFERENCES claims(request),
  PRIMARY KEY(parent, node, predecessor)
);
CREATE TABLE events(
  sequence INTEGER PRIMARY KEY, request TEXT NOT NULL REFERENCES claims(request),
  state TEXT NOT NULL, detail TEXT NOT NULL, epoch REAL NOT NULL,
  UNIQUE(request, state)
);
PRAGMA user_version=1;
"""


def need(condition, error):
    if not condition:
        raise ValueError(error)


def digest(data):
    return sha256(data).hexdigest()


def encode(data):
    return (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()


def pinned(path, expected):
    need(path.is_file() and path.resolve(strict=True) == path.absolute(), "CANONICAL_SOURCE")
    raw = path.read_bytes()
    need(digest(raw) == expected, "SOURCE_SHA_CHANGED")
    return raw


def hex_id(value, length):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


def legacy():
    pinned(SOURCE, SOURCE_SHA)
    frozen = runpy.run_path(str(SOURCE))
    env = {"__name__": "vnext_scope", "__file__": str(SOURCE)}
    exec(frozen["scope_source"](), env)
    pinned(env["PUBLISHER"], PUBLICATION_SHA)
    return env


class Conflict(Exception):
    """A request or candidate key is already in the shared vNext ledger."""


class Ledger:
    def __init__(self, path):
        self.path = path

    @classmethod
    def create(cls, path):
        with path.open("xb"):
            pass
        with closing(sqlite3.connect(path)) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.executescript(SCHEMA)
        return cls(path)

    def connect(self):
        need(self.path.is_file() and self.path.resolve(strict=True) == self.path.absolute(), "EXISTING_CANONICAL_LEDGER")
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=8)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            need(db.execute("PRAGMA user_version").fetchone() == (1,), "EXACT_LEDGER_VERSION")
        except BaseException:
            db.close()
            raise
        return db

    @contextmanager
    def transaction(self):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    def claim(self, request, wave, group, calendar_sha, publication, observation_sha, roster):
        require_roster(roster)
        need(hex_id(request, 32) and hex_id(calendar_sha, 64) and hex_id(publication, 40)
             and hex_id(observation_sha, 64) and type(group) is int and group > 0
             and re.fullmatch(r"[a-z0-9][a-z0-9-]{4,63}", wave), "EXACT_GROUP_CLAIM_SCOPE")
        try:
            with self.transaction() as db:
                db.execute("INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (request, wave, group, calendar_sha, publication, observation_sha, len(roster), time.time()))
                db.executemany("INSERT INTO targets VALUES (?, ?, ?, ?)", ((*row, request) for row in roster))
                self._insert(db, request, "CLAIMED", {"roster_count": len(roster), "replay_allowed": False})
        except sqlite3.IntegrityError as exc:
            raise Conflict("GROUP_ATOMIC_CONFLICT") from exc

    @staticmethod
    def _insert(db, request, state, detail):
        db.execute("INSERT INTO events(request, state, detail, epoch) VALUES (?, ?, ?, ?)",
                   (request, state, encode(detail).decode(), time.time()))

    def advance(self, request, previous, state, detail):
        with self.transaction() as db:
            head, = db.execute("SELECT state FROM events WHERE request=? ORDER BY sequence DESC LIMIT 1",
                               (request,)).fetchone()
            need(head == previous, "ONE_WAY_REQUEST_TRANSITION")
            expected, = db.execute("SELECT roster_count FROM claims WHERE request=?", (request,)).fetchone()
            actual, = db.execute("SELECT count(*) FROM targets WHERE request=?", (request,)).fetchone()
            need(actual == expected, "ATOMIC_ROSTER_EXTENT")
            self._insert(db, request, state, detail)

    def status(self, request):
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            claim = db.execute("SELECT wave,group_id,calendar_sha,publication,observation_sha,roster_count FROM claims "
                               "WHERE request=?", (request,)).fetchone()
            events = db.execute("SELECT state,detail FROM events WHERE request=? ORDER BY sequence", (request,)).fetchall()
            actual, = db.execute("SELECT count(*) FROM targets WHERE request=?", (request,)).fetchone()
        scope = None
        if claim is not None:
            need(claim[-1] == actual and bool(events), "ATOMIC_ROSTER_EXTENT")
            keys = ("wave", "group", "calendar_SHA256", "publication", "terminal_observation_SHA256")
            scope = dict(zip(keys, claim[:-1], strict=True))
        state = events[-1][0] if events else "ABSENT"
        state = {"CLAIMED": "CLAIMED_UNRESOLVED", "BRIDGE_INTENT": "BRIDGE_UNKNOWN"}.get(state, state)
        return {"request": request, "state": state, "events": [name for name, _ in events],
                "scope": scope, "evidence": json.loads(events[-1][1]) if events else None,
                "claimed": actual, "replay_allowed": False}


def require_roster(roster):
    need(1 <= len(roster) <= 8 and len({row[0] for row in roster}) == 1
         and len({row[1] for row in roster}) == len({row[2] for row in roster}) == len(roster)
         and all(parent in ("1103606", "1194900") and re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}", node)
                 and re.fullmatch(parent + r"\.[0-9]+", previous) for parent, node, previous in roster),
         "EXACT_PARENT_LOCAL_ROSTER")


def hook(source):
    return ast.parse(source).body[0]


def assigned(stmt, name):
    return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name)


def compiled(parts, name):
    return compile(ast.fix_missing_locations(ast.Module(body=parts, type_ignores=[])), name, "exec")


def frozen_action(publisher):
    original = publisher["render"]
    context = original.__globals__
    path = context["REFILL"]
    pinned(path, RUNNER_SHA)
    capture = dict(context, compile=lambda source, filename, mode: source)
    source = FunctionType(original.__code__, capture)()
    statements = ast.parse(source).body
    probe, = (index for index, stmt in enumerate(statements) if assigned(stmt, "t0"))
    send, = (index for index, stmt in enumerate(statements) if isinstance(stmt, ast.Try)
             and any(assigned(child, "q") for child in stmt.body))
    require_call = lambda call, timeout: (ast.unparse(call.func) == "subprocess.run"
                                         and next(item.value.value for item in call.keywords if item.arg == "timeout") == timeout)
    charge = statements[probe + 1]
    fail = statements[probe + 3]
    outbound = statements[send].body[0]
    need(assigned(charge, "a") and require_call(charge.value, 60)
         and isinstance(fail, ast.If) and ast.unparse(fail.test) == "a.returncode"
         and assigned(outbound, "q") and require_call(outbound.value, 120), "ACTUAL_FROZEN_ADMISSION_AND_BRIDGE")
    charge.value.args.insert(0, charge.value.func)
    charge.value.func = ast.Name(id="vnext_candidate_admission", ctx=ast.Load())
    fail.body.insert(0, hook("vnext_admission_failed(req, a)"))
    outbound.value.args.insert(0, outbound.value.func)
    outbound.value.func = ast.Name(id="vnext_bridge_send", ctx=ast.Load())
    outbound.value.keywords.append(ast.keyword(arg="vnext_intent", value=ast.Name(id="intent", ctx=ast.Load())))
    outbound.value.keywords.append(ast.keyword(arg="vnext_request", value=ast.Name(id="request", ctx=ast.Load())))
    ack, = (stmt for stmt in statements[send:] if isinstance(stmt, ast.If)
            and "bridge_launch_pending" in ast.unparse(stmt.test))
    need("d.get('request') != request" in ast.unparse(ack.body[0])
         and "d.get('planned_DCU') != 0" in ast.unparse(ack.body[0]), "FROZEN_VALIDATED_ACK")
    ack.body.insert(1, hook("vnext_bridge_ack(request, d, intent)"))
    replacements = {"root": "vnext_root", "via_compute": "True", "priority": "vnext_priorities",
                    "dest": "vnext_audit_root / ('admission-' + req)"}
    for stmt in statements[:probe]:
        key = next((name for name in replacements if assigned(stmt, name)), None)
        if key is not None:
            stmt.value = ast.parse(replacements.pop(key), mode="eval").body
    need(not replacements, "EXACT_FROZEN_ENVIRONMENT")
    return tuple(compiled(part, name) for part, name in (
        (statements[:probe], "<vnext exact preparation>"),
        (statements[probe:probe + 4], "<vnext exact candidate admission>"),
        (statements[probe + 4:send], "<vnext exact device and publication gates>"),
        (statements[send:], "<vnext exact compute bridge>"),
    ))


def execute(ledger, request, roster, publication, action, publisher, state, audit, *, fixture=None):
    priorities = sorted(row[1] for row in roster)
    admitted = None

    def admission(run, *args, **kwargs):
        nonlocal admitted
        need(admitted is None, "ONE_CANDIDATE_ADMISSION")
        admitted = run(*args, **kwargs)
        return admitted

    def failed(probe_request, reply):
        need(reply is admitted and type(reply.returncode) is int and reply.returncode != 0
             and hex_id(probe_request, 32), "EXACT_ADMISSION_FAILURE_BRANCH")
        ledger.advance(request, "CLAIMED", "ADMISSION_FAILED_BEFORE_BRIDGE",
                       {"probe_request": probe_request, "rc": reply.returncode,
                        "stdout_SHA256": digest(reply.stdout), "stderr_SHA256": digest(reply.stderr),
                        "bridge_intent_written": False, "no_bridge_send_proven": True})

    def send(run, *args, vnext_intent, vnext_request, **kwargs):
        need(admitted is not None and admitted.returncode == 0
             and vnext_intent["request"] == vnext_request and hex_id(vnext_request, 32)
             and vnext_intent["publication"] == publication and vnext_intent["replay_allowed"] is False
             and vnext_intent["receipt_request"] == namespace["req"] and vnext_intent["nodes"] == priorities
             and vnext_intent["requested_cards"] == 8 * len(roster)
             and all(namespace["receipt"]["nodes"][node]["parent"] == parent for parent, node, _ in roster)
             and hex_id(vnext_intent["payload_SHA256"], 64), "WHOLE_GROUP_REQUIRED_AT_BRIDGE")
        command = args[0]
        body = kwargs["input"]
        ledger.advance(request, "CLAIMED", "BRIDGE_INTENT",
                       {"bridge_request": vnext_request, "admission_request": namespace["req"],
                        "controller_SHA256": vnext_intent["payload_SHA256"], "wire_SHA256": digest(body),
                        "argv_SHA256": digest(encode(command)), "roster_count": len(roster), "replay_allowed": False})
        need(time.monotonic() < namespace["cutoff"], "ADMISSION_EXPIRED_DURING_DURABLE_INTENT")
        return run(*args, **kwargs)

    def ack(bridge_request, observed, intent):
        need(observed.get("bridge_launch_pending") is True and observed["request"] == bridge_request
             and observed["planned_DCU"] == 0 and isinstance(observed.get("control_label"), str),
             "EXACT_AUTHENTICATED_BRIDGE_ACK")
        ledger.advance(request, "BRIDGE_INTENT", "BRIDGE_ACKNOWLEDGED",
                       {"bridge_request": bridge_request, "control_label": observed["control_label"],
                        "requested_cards": intent["requested_cards"], "actual_GPU_steps_verified": 0})

    namespace = {"__name__": "__main__", "certified_tip": publication, "vnext_root": ROOT.parent,
                 "vnext_priorities": priorities, "vnext_audit_root": audit,
                 "vnext_candidate_admission": admission, "vnext_admission_failed": failed,
                 "vnext_bridge_send": send, "vnext_bridge_ack": ack,
                 "check_preverified_publication": lambda repo, tip, paths: publisher["verify_bound"](state, repo, tip, paths),
                 "pinned_members": publisher["ALL"]}
    before, probe, middle, bridge = action
    try:
        trace = audit / f"{request}-action.stdout.private"
        with trace.open("x", encoding="utf-8", newline="\n") as stdout:
            namespace["print"] = partial(print, file=stdout)
            try:
                if fixture is None:
                    exec(before, namespace)
                else:
                    fixture.prepare(namespace)
                exec(probe, namespace)
                if fixture is None:
                    exec(middle, namespace)
                else:
                    fixture.bridge_inputs(namespace)
                exec(bridge, namespace)
            finally:
                stdout.flush()
                os.fsync(stdout.fileno())
    except (SystemExit, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        evidence = {"error_type": type(exc).__name__}
        if isinstance(exc, SystemExit) and type(exc.code) is int:
            evidence["rc"] = exc.code
    else:
        evidence = {"action_returned": True}
    current = ledger.status(request)
    if current["state"] == "CLAIMED_UNRESOLVED":
        ledger.advance(request, "CLAIMED", "STOPPED_BEFORE_BRIDGE", evidence)
    elif current["state"] == "BRIDGE_UNKNOWN":
        ledger.advance(request, "BRIDGE_INTENT", "BRIDGE_UNKNOWN", evidence)
    return ledger.status(request)


def run_group(ledger, *, request, wave, group, calendar_sha, publication, observation_sha, roster,
              action, publisher, state, audit, fixture=None):
    ledger.claim(request, wave, group, calendar_sha, publication, observation_sha, roster)
    return execute(ledger, request, roster, publication, action, publisher, state, audit, fixture=fixture)


def calendar_scope(coordinator, calendar, calendar_sha):
    need(calendar is not None and calendar.is_absolute() and calendar.is_relative_to(ROOT / "local-only")
         and hex_id(calendar_sha, 64) and calendar_sha != OLD_CALENDAR_SHA, "NEW_EXPLICIT_CALENDAR_REQUIRED")
    if calendar_sha == calendar58.VERIFIED_SHA or calendar.name == calendar58.VERIFIED_FILE:
        return calendar58.validate(calendar, calendar_sha, coordinator)
    pinned(calendar, calendar_sha)
    coordinator.update(CALENDAR=calendar, CALENDAR_SHA=calendar_sha)
    return coordinator["groups"](), None


def dispatch_group(ledger, group, rows, request, wave, calendar_sha, coordinator, base, publisher, action,
                   *, require_all=False, clock=time.time, audit=AUDIT_ROOT, fixture=None):
    prior = ledger.status(request)
    if prior["state"] != "ABSENT":
        return dict(prior, new_claims=0)
    need(rows[0]["planned_end_epoch"] - 65 <= clock() <= rows[-1]["planned_end_epoch"] + 180,
         "FRESH_FUTURE_WAVE_ADMISSION_WINDOW")
    state = publisher["precheck"]()
    receipt, _, released = coordinator["observe"](group, rows, base)
    roster = tuple((row["parent"], row["node"], row["previous"]) for row in released)
    if not roster:
        return {"state": "NO_RELEASED_CANDIDATES", "new_claims": 0}
    if require_all and released != rows:
        return {"state": "EXACT_GROUP_AWAITING_ALL_ORIGINAL_TERMINALS", "released": len(roster),
                "planned": len(rows), "new_claims": 0}
    try:
        return run_group(ledger, request=request, wave=wave, group=group,
                         calendar_sha=calendar_sha, publication=state["head"], observation_sha=digest(receipt.read_bytes()),
                         roster=roster, action=action, publisher=publisher, state=state, audit=audit, fixture=fixture)
    except Conflict:
        return {"state": "GROUP_ATOMIC_CONFLICT", "new_claims": 0, "replay_allowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "check-calendar", "init-ledger", "status", "dispatch"))
    parser.add_argument("--calendar", type=Path)
    parser.add_argument("--calendar-sha")
    parser.add_argument("--wave")
    parser.add_argument("--group", type=int)
    parser.add_argument("--request")
    parser.add_argument("--adoption", type=Path)
    parser.add_argument("--adoption-sha")
    args = parser.parse_args()
    database = AUDIT_ROOT / "claim-ledger-v1.sqlite3"
    if args.mode == "init-ledger":
        AUDIT_ROOT.mkdir(mode=0o700, exist_ok=True)
        Ledger.create(database)
        print(json.dumps({"state": "EXPLICIT_NEW_LEDGER_CREATED", "replay_allowed": False}))
        return
    if args.mode == "status":
        need(hex_id(args.request, 32), "EXACT_REQUEST")
        print(json.dumps(Ledger(database).status(args.request), sort_keys=True))
        return
    coordinator = legacy()
    if args.mode == "check-calendar":
        groups, scope = calendar_scope(coordinator, args.calendar, args.calendar_sha)
        if scope is None:
            scope = {"state": "EXACT_FUTURE_CALENDAR_CHECKED", "candidates": sum(map(len, groups)), "remote_calls": 0}
        print(json.dumps(scope, sort_keys=True))
        return
    base, publisher = coordinator["modules"]()
    action = frozen_action(publisher)
    if args.mode == "check":
        pinned(EVIDENCE, EVIDENCE_SHA)
        print(json.dumps({"state": "EXACT_SOURCE_ADMISSION_AND_BRIDGE_COMPILED", "remote_calls": 0}))
        return
    need(hex_id(args.request, 32)
         and args.wave is not None and re.fullmatch(r"training80-vnext-[a-z0-9-]{3,46}", args.wave)
         and type(args.group) is int and args.group >= 1, "NEW_EXPLICIT_WAVE_REQUIRED")
    groups, scope = calendar_scope(coordinator, args.calendar, args.calendar_sha)
    coordinator.update(WAVE=args.wave)
    need((scope is not None) == (args.wave == calendar58.WAVE), "EXACT_58_WAVE_IDENTITY")
    if scope is not None:
        calendar58.require_adoption(args.adoption, args.adoption_sha, args.wave, database, scope, coordinator["LOCAL"])
    need(args.group <= len(groups), "EXACT_FUTURE_GROUP")
    rows = groups[args.group - 1]
    result = dispatch_group(Ledger(database), args.group, rows, args.request, args.wave, args.calendar_sha,
                            coordinator, base, publisher, action, require_all=scope is not None)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
