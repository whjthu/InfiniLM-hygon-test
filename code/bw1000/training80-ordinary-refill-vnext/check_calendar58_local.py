"""Read the exact 58 source and exercise the coordinator's production group boundary locally."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import calendar58 as cal
import check_local as original_test
import coordinator as co


def expect_error(reason, action):
    try:
        action()
    except ValueError as ex:
        assert str(ex) == reason, str(ex)
        return reason
    raise AssertionError(reason)


def terminal_fixture(base, group, rows, ready):
    evidence = {"batch": group, "rc": 0, "observed": {
        "plans_only": True, "requires_fresh_independent_device_and_dispatch_checks": True,
        "targets": [
            {"node": row["node"], "step": row["previous"], "request": row["request"], "terminal_ready": ready[i],
             "worker_SHA256": row["worker_SHA256"], "expected_SHA256": row["expected_SHA256"], "terminal_SHA256": "6"*64,
             "accounting": [row["previous"], "captured-fixture", "COMPLETED", "0:0", str(row["worker_seconds"]),
                            row["node"], "cpu=8,gres/dcu=8,mem=16G,node=1", "2000-01-01T00:00:00",
                            "2000-01-01T03:00:00"]}
            for i, row in enumerate(rows)
        ]}}
    return evidence, base["released_nodes"](rows, evidence)


def main(destination):
    legacy = co.legacy()
    base, pub_source = legacy["modules"]()
    action = co.frozen_action(pub_source)
    local = legacy["LOCAL"]
    calendar = local / cal.VERIFIED_FILE
    groups, summary = co.calendar_scope(legacy, calendar, cal.VERIFIED_SHA)
    assert len(groups) == 16 and summary["ordinary_candidates"] == 58 and summary["excluded_nodes"] == 22
    assert summary["parent_candidates"] == {"1103606": 24, "1194900": 34}
    assert summary["whole_pool_DCU"] == 640 and summary["net_new_DCU"] == 0
    assert len({(r["parent"], r["node"], r["previous"]) for rows in groups for r in rows}) == 58
    assert all(r["previous_worker_pid"] == r["worker_PID"] for rows in groups for r in rows)
    results = {"verified_calendar": summary, "dispatch_test_inputs": "captured only; original current successors were running"}
    results["historical_70_cannot_dispatch"] = expect_error("NEW_EXPLICIT_CALENDAR_REQUIRED",
                                    lambda: co.calendar_scope(legacy, legacy["CALENDAR"], co.OLD_CALENDAR_SHA))
    results["correct_path_wrong_sha_rejected"] = expect_error("EXACT_58_CALENDAR_ARGUMENT",
                                    lambda: co.calendar_scope(legacy, calendar, "1"*64))

    with TemporaryDirectory(prefix="training80-ordinary58-proof-", dir=local) as scratch:
        tmp = Path(scratch).resolve()
        dbpath = tmp / "claim-ledger-v1.sqlite3"
        ledger = co.Ledger.create(dbpath)
        adoption = tmp / "adoption.json"
        provision = {"schema": "training80.ordinary.vnext-sole-adoption.v1", "wave": cal.WAVE,
                     "calendar_SHA256": cal.VERIFIED_SHA, "group_roster_SHA256": summary["group_roster_SHA256"],
                     "ledger_absolute": dbpath.as_posix(), "ledger_device": dbpath.stat().st_dev,
                     "ledger_inode": dbpath.stat().st_ino, "previous_ordinary_controller_released": True,
                     "sole_current_coordinator_authorized": True}
        body = cal.encoded(provision)
        adoption.write_bytes(body)
        cal.require_adoption(adoption, cal.pin(body), cal.WAVE, dbpath, summary, local)
        bad_adoption = tmp / "wrong-adoption.json"
        bad = cal.encoded(dict(provision, ledger_inode=provision["ledger_inode"]+1))
        bad_adoption.write_bytes(bad)
        results["different_ledger_identity_rejected"] = expect_error(
            "ADOPTION_MUST_BIND_SAME_EXISTING_LEDGER_AND_CALENDAR",
            lambda: cal.require_adoption(bad_adoption, cal.pin(bad), cal.WAVE, dbpath, summary, local))

        calls = []
        partial = True
        publisher = dict(pub_source)
        def precheck():
            calls.append("publication")
            return {"head": original_test.PUBLICATION}
        publisher["precheck"] = precheck

        def observe(group, rows, real_base):
            calls.append("observation")
            evidence, released = terminal_fixture(real_base, group, rows, [True] + [not partial] * (len(rows) - 1))
            receipt = tmp / ("observation-" + str(calls.count("observation")) + ".json")
            with receipt.open("xb") as output:
                output.write(cal.encoded(evidence))
            return receipt, evidence, released

        integration = dict(legacy, observe=observe)
        good = {"admission_rc": 0, "bridge_rc": 0, "stdout": b"", "stderr": b""}

        def capture(request, group=1):
            rows = groups[group-1]
            roster = tuple((r["parent"], r["node"], r["previous"]) for r in rows)
            driver = original_test.Fixture(ledger, request, tmp, roster, good)
            outcome = co.dispatch_group(ledger, group, rows, request, cal.WAVE, cal.VERIFIED_SHA, integration,
                                        base, publisher, action, require_all=True, fixture=driver, audit=tmp,
                                        clock=lambda: rows[-1]["planned_end_epoch"] + 1)
            return outcome, driver.calls

        request = "a"*32
        result, sends = capture(request)
        assert result["state"] == "EXACT_GROUP_AWAITING_ALL_ORIGINAL_TERMINALS" and result["released"] == 1
        assert sends == [] and ledger.status(request)["state"] == "ABSENT"
        results["partial_terminal"] = dict(result, bridge_calls=0)

        partial = False
        row = groups[0][-1]
        prior = "b"*32
        ledger.claim(prior, "training80-vnext-prior-test", 99, "9"*64, original_test.PUBLICATION, "8"*64,
                     ((row["parent"], row["node"], row["previous"]),))
        ledger.advance(prior, "CLAIMED", "BRIDGE_INTENT", {"replay_allowed": False})
        result, sends = capture(request)
        assert result["state"] == "GROUP_ATOMIC_CONFLICT" and sends == []
        assert ledger.status(request)["claimed"] == 0
        results["shared_ledger_group_conflict"] = dict(result, bridge_calls=0, unchanged_prior_claim=1)

        before = len(calls)
        duplicate, sends = capture(prior)
        assert duplicate["state"] == "BRIDGE_UNKNOWN" and duplicate["new_claims"] == 0 and sends == []
        assert len(calls) == before
        results["original_unknown_repeat"] = {"state": duplicate["state"], "new_claims": 0,
                                              "publication_calls": 0, "observer_calls": 0, "bridge_calls": 0}

        completed, sends = capture("c"*32, 2)
        assert completed["state"] == "BRIDGE_ACKNOWLEDGED" and completed["claimed"] == 1 and sends == [60, 120]
        results["exact_other_group_fixture"] = {"state": completed["state"], "claimed": 1, "bridge_capture": 1}

    results.update(schema="training80.ordinary58.vnext.local-validation-v1", remote_calls=0, Slurm=0, GPU=0,
                   production_ledger_changed=False, production_adoption_created=False)
    raw = cal.encoded(results)
    with destination.open("xb") as output:
        output.write(raw)
    print(json.dumps({"bytes": len(raw), "SHA256": cal.pin(raw)}, sort_keys=True))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
