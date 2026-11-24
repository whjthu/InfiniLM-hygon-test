"""Use the vNext entry and the exact frozen admission/bridge branches with captured transports."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier
import time
from types import SimpleNamespace

import coordinator as co


ORIGINALS = {4: "edf616f6543e4dafa7ead6fe9394e695", 5: "6ce9954640f743f6a29432e72c2fc3ba",
             13: "e0efd68bbc4d429eb87cf3ad255972cd"}
TEST_WAVE = "training80-vnext-local"
PUBLICATION = "1" * 40
OLD_JOURNAL = co.ROOT / "local-only/training80-1738-ordinary-renewal-dispatch"


def args(group, roster, request, calendar=co.OLD_CALENDAR_SHA):
    return dict(request=request, wave=TEST_WAVE, group=group, calendar_sha=calendar, publication=PUBLICATION,
                observation_sha="8" * 64, roster=roster)


def exact_groups(legacy, base):
    evidence = json.loads(co.pinned(co.EVIDENCE, co.EVIDENCE_SHA))
    groups = legacy["groups"]()
    historical = {}
    for group, invocation in ORIGINALS.items():
        claim = evidence["groups"][str(group)]
        outcome_path = OLD_JOURNAL / (invocation + "-outcome.json")
        outcome = json.loads(co.pinned(outcome_path, claim["outcome_SHA256"]))
        public_stdout = co.pinned(OLD_JOURNAL / (invocation + "-stdout.private"), claim["stdout_SHA256"])
        terminal = json.loads(co.pinned(co.ROOT / "local-only" / outcome["terminal_receipt"],
                                        outcome["terminal_receipt_SHA256"]))
        assert outcome["batch"] == group and outcome["refill_rc"] == claim["rc"] == 255
        assert outcome["stdout_SHA256"] == claim["stdout_SHA256"]
        released = {row["node"]: row for row in base["released_nodes"](groups[group - 1], terminal)}
        assert set(outcome["terminal_ready_nodes"]) <= set(released)
        roster = tuple((r["parent"], r["node"], r["previous"])
                       for node in outcome["terminal_ready_nodes"] for r in (released[node],))
        assert len(roster) == claim["node_count"]
        if claim["observed_branch"] == "ADMISSION_PRECEDES_BRIDGE":
            message, rc, stderr_hash = public_stdout.strip().rsplit(b" ", 2)
            assert message == b"Bounded admission did not complete" and int(rc) == claim["rc"]
            observation = {"stdout": b"", "stderr": b"", "admission_rc": int(rc), "bridge_rc": None,
                           "historical_stderr_SHA256": stderr_hash.decode()}
        else:
            status, = [json.loads(line) for line in public_stdout.splitlines()
                       if line.startswith(b"{") and json.loads(line).get("remote_outcome") == "UNKNOWN"]
            assert status["rc"] == claim["rc"] and co.hex_id(status["request"], 32)
            observation = {"stdout": b"", "stderr": b"", "admission_rc": 0, "bridge_rc": status["rc"],
                           "bridge_request": status["request"]}
        historical[group] = (roster, observation, outcome["terminal_receipt_SHA256"], claim)
    return historical


class Fixture:
    def __init__(self, ledger, request, audit, roster, observation, *, crash=False, shrink=False, stale=False):
        self.ledger, self.request, self.audit, self.roster = ledger, request, audit, roster
        self.observation, self.crash, self.shrink, self.stale = observation, crash, shrink, stale
        self.calls = []

    def process(self, argv, *, input, capture_output, timeout):
        assert capture_output and argv[0] == "capture-only"
        self.calls.append(timeout)
        if timeout == 60:
            assert self.ledger.status(self.request)["events"] == ["CLAIMED"]
            return SimpleNamespace(returncode=self.observation["admission_rc"],
                                   stdout=self.observation["stdout"], stderr=self.observation["stderr"])
        assert timeout == 120
        row = self.ledger.status(self.request)
        assert row["events"] == ["CLAIMED", "BRIDGE_INTENT"] and row["claimed"] == len(self.roster)
        assert row["evidence"]["wire_SHA256"] == co.digest(input)
        assert row["evidence"]["argv_SHA256"] == co.digest(co.encode(argv))
        if self.crash:
            os._exit(71)
        rc = self.observation["bridge_rc"]
        reply = b"" if rc else co.encode({"request": self.bridge_request, "planned_DCU": 0,
                                          "bridge_launch_pending": True, "control_label": "fixture-ack"})
        return SimpleNamespace(returncode=rc, stdout=reply, stderr=b"")

    def prepare(self, scope):
        dest = self.audit / self.request
        dest.mkdir()
        scope.update(req=self.request, inner=["capture-only"], probe_command=["capture-admission"],
                     subprocess=SimpleNamespace(run=self.process, TimeoutExpired=subprocess.TimeoutExpired),
                     time=time, shlex=__import__("shlex"), base64=__import__("base64"), bundle={},
                     sha256=co.sha256, json=json, sys=sys, dest=dest)

    def bridge_inputs(self, scope):
        self.bridge_request = self.observation.get("bridge_request", "9" * 32)
        nodes = sorted(row[1] for row in self.roster)
        scope.update(command="capture-bridge", transport_body=b"captured-wire", request=self.bridge_request,
                     via_compute=True, cutoff=time.monotonic() + (-1 if self.stale else 30),
                     receipt={"nodes": {node: {"parent": parent} for parent, node, _ in self.roster}},
                     intent={"request": self.bridge_request, "publication": PUBLICATION, "replay_allowed": False,
                             "receipt_request": self.request, "nodes": nodes[1:] if self.shrink else nodes,
                             "requested_cards": len(nodes) * 8, "payload_SHA256": "7" * 64, "ready_training": 0})


def blocked(ledger, setup, parameters, fixture):
    try:
        co.run_group(ledger, **setup, **parameters, fixture=fixture)
    except co.Conflict:
        assert not fixture.calls
        return ledger.status(parameters["request"])
    raise AssertionError("conflicting cohort was not rejected")


def setup_at(path):
    legacy = co.legacy()
    base, publisher = legacy["modules"]()
    setup = dict(action=co.frozen_action(publisher), publisher=publisher, state={"head": PUBLICATION}, audit=path)
    return legacy, base, setup


def crash_mode(db, audit):
    ledger = co.Ledger(db)
    _, _, setup = setup_at(audit)
    roster = (("1103606", "f13r2n04", "1103606.99101"),)
    request = "c" * 32
    capture = Fixture(ledger, request, audit, roster, {"admission_rc": 0, "bridge_rc": 0, "stdout": b"", "stderr": b""},
                      crash=True)
    co.run_group(ledger, **setup, **args(308, roster, request), fixture=capture)
    raise AssertionError("crash injection never reached bridge")


def main(output):
    report = {}
    with TemporaryDirectory(prefix="ordinary-refill-vnext-", dir=co.ROOT / "local-only") as scratch:
        tmp = Path(scratch).resolve()
        database = tmp / "atomic.sqlite3"
        audit = tmp / "evidence"
        audit.mkdir()
        ledger = co.Ledger.create(database)
        historical_scope, base, setup = setup_at(audit)
        historical = exact_groups(historical_scope, base)
        original = {}
        bridge_rc = 0
        for group, (roster, observation, terminal_sha, claim) in historical.items():
            request = f"{group:032x}"
            fixture = Fixture(ledger, request, audit, roster, observation)
            parameters = args(group, roster, request)
            parameters["observation_sha"] = terminal_sha
            state = co.run_group(ledger, **setup, **parameters, fixture=fixture)
            bridge_unknown = claim["observed_branch"] == "BRIDGE_UNKNOWN"
            assert state["state"] == ("BRIDGE_UNKNOWN" if bridge_unknown else "ADMISSION_FAILED_BEFORE_BRIDGE")
            assert fixture.calls == ([60, 120] if bridge_unknown else [60])
            assert state["claimed"] == claim["node_count"]
            if bridge_unknown:
                assert state["evidence"]["rc"] == claim["rc"]
            else:
                assert state["evidence"]["rc"] == claim["rc"] and state["evidence"]["no_bridge_send_proven"]
            bridge_rc += fixture.calls.count(120)
            replay = Fixture(ledger, request, audit, roster, observation)
            blocked(ledger, setup, parameters, replay)
            original[str(group)] = {"state": state["state"], "claimed": state["claimed"], "captured_transports": fixture.calls,
                                    "original_outcome_SHA256": claim["outcome_SHA256"],
                                    "original_stdout_SHA256": claim["stdout_SHA256"]}
        assert bridge_rc == 1
        old_again = exact_groups(historical_scope, base)
        assert old_again == historical
        report["exact_history"] = original

        good = {"admission_rc": 0, "bridge_rc": 0, "stdout": b"", "stderr": b""}
        p = "1103606"
        overlap = (p, "f13r2n12", p + ".99093")
        first = ((p, "f13r2n10", p + ".99091"), overlap)
        second = ((p, "f13r2n11", p + ".99092"), overlap)
        third = ((p, "f13r2n13", p + ".99094"),)
        race = Barrier(3)

        def actor(request, group, roster):
            capture = Fixture(ledger, request, audit, roster, good)
            race.wait(timeout=8)
            try:
                info = co.run_group(ledger, **setup, **args(group, roster, request, f"{group:064x}"), fixture=capture)
                assert info["state"] == "BRIDGE_ACKNOWLEDGED" and info["claimed"] == len(roster)
            except co.Conflict:
                assert ledger.status(request)["state"] == "ABSENT"
            return capture.calls

        with ThreadPoolExecutor(max_workers=3) as pool:
            calls = [pool.submit(actor, str(n) * 32, n + 100, roster)
                     for n, roster in zip((1, 2, 3), (first, second, third))]
            a, b, c = [future.result(timeout=12) for future in calls]
        assert sorted((a, b)) == [[], [60, 120]] and c == [60, 120]
        assert sorted(ledger.status(str(n) * 32)["claimed"] for n in (1, 2, 3)) == [0, 1, 2]
        report["concurrent_groups"] = {"attempted": 3, "full_claimed": 2, "zero_claimed": 1, "partial": 0,
                                       "captured_bridge_calls": 2, "shared_key_sends": 1}

        partial = Fixture(ledger, "a" * 32, audit, ((p, "f13r2n01", p + ".99121"), (p, "f13r2n02", p + ".99122")),
                          good, shrink=True)
        stopped = co.run_group(ledger, **setup, **args(232, partial.roster, partial.request), fixture=partial)
        assert stopped["state"] == "STOPPED_BEFORE_BRIDGE" and partial.calls == [60]
        report["partial_device_selection"] = {"state": stopped["state"], "claimed": stopped["claimed"],
                                                "bridge_calls": 0}

        result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
                                 "--crash-child", str(database), str(audit)], capture_output=True, timeout=25)
        assert result.returncode == 71
        new_connection = co.Ledger(database)
        crash_request = "c" * 32
        crash = new_connection.status(crash_request)
        assert crash["events"] == ["CLAIMED", "BRIDGE_INTENT"] and crash["state"] == "BRIDGE_UNKNOWN"
        crash_roster = ((p, "f13r2n04", p + ".99101"),)
        replacement_request = "d" * 32
        blocked(new_connection, setup, args(309, crash_roster, replacement_request),
                Fixture(new_connection, replacement_request, audit, crash_roster, good))
        assert new_connection.status(replacement_request)["state"] == "ABSENT"
        report["hard_process_exit_after_committed_intent"] = {"before_exit": crash["events"],
            "reopened_state": crash["state"], "replacement_rows_committed": 0, "automatic_replays": 0}

        stale = Fixture(ledger, "e"*32, audit, ((p, "f13r2n03", p + ".99123"),), good, stale=True)
        stale_result = co.run_group(ledger, **setup, **args(310, stale.roster, stale.request), fixture=stale)
        assert stale_result["state"] == "BRIDGE_UNKNOWN" and stale.calls == [60]
        report["expiry_after_commit"] = {"state": stale_result["state"], "bridge_calls": 0}

    report.update(schema="training80.ordinary-refill-vnext.local-v1", classification_SHA256=co.EVIDENCE_SHA,
                  launcher_SHA256=co.SOURCE_SHA, actual_admission_bridge_source_SHA256=co.RUNNER_SHA,
                  coordinator_SHA256=co.digest(Path(co.__file__).read_bytes()),
                  full_real_frozen_admission_and_bridge_statements_executed=True, remote_transports=0, Slurm=0, GPU=0,
                  old_state_changed=False, historical_direct_automatic_replays=0)
    raw = co.encode(report)
    with output.open("xb") as handle:
        handle.write(raw)
    print(json.dumps({"bytes": len(raw), "SHA256": co.digest(raw)}))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--crash-child"]:
        crash_mode(*(Path(name) for name in sys.argv[2:]))
    else:
        main(Path(sys.argv[1]))
