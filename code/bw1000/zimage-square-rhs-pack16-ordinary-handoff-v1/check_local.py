"""Capture the real coordinator and frozen Pack16 grant paths without submission."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import arbiter
import ordinary
import pack16_host


NOW = 1795560000
FIXTURE_OUTPUT = str(arbiter.RESULTS / "O01-pack16-2930-0000000000000001")
CALENDAR = ordinary.ROOT / "local-only/training80-0544-ordinary-70-exact-predecessor-calendar-preparation-v1.json"


def rejected(call, gate):
    try:
        call()
    except ValueError as error:
        assert str(error) == gate, error
    else:
        raise AssertionError("unexpected acceptance: " + gate)


def define(path, extra_scope=None):
    journal = path / "original-journal"
    journal.mkdir(parents=True)
    namespace, ledger, group = ordinary.original(journal)
    members = namespace["groups"]()[group - 1]
    (target,) = [row for row in members if row["node"] == arbiter.NODE]
    scope = {"request": arbiter.REQUEST, "label": arbiter.LABEL, "parent": arbiter.PARENT,
             "node": arbiter.NODE, "root_admitted": True, "epoch": NOW, "deadline_epoch": NOW + 240,
             "gfx": "gfx936", "host_output": FIXTURE_OUTPUT}
    adoption = {"schema": arbiter.SCOPE_SCHEMA, "parent": arbiter.PARENT, "node": arbiter.NODE,
                "predecessor": arbiter.PREDECESSOR, "calendar_SHA256": arbiter.CALENDAR_SHA, "group": group,
                "sole_coordinator_confirmed": True, "legacy_target_dispatch_drained": True,
                "coordinator_SHA256": arbiter.signature(Path(ordinary.__file__).read_bytes()),
                "observed_epoch": NOW - 1, "valid_until_epoch": NOW + 45}
    scope.update(extra_scope or {})
    scope_sha = arbiter.publish(path / "scope.json", scope)
    adoption_sha = arbiter.publish(path / "adoption.fixture.json", adoption)
    def accept():
        return ledger.accept(scope, scope_sha, adoption, adoption_sha, group, NOW)
    return journal, namespace, ledger, group, target, scope, scope_sha, accept


def capture_refill(namespace, group, ready, directory, *, rows=None):
    directory.mkdir(exist_ok=True)
    seen = []
    receipt = directory / "terminal-fixture.json"
    receipt.write_bytes(b'{"local_fixture":true}\n')
    namespace["observe"] = lambda group, rows, base: (receipt, {"rc": 0}, ready)
    namespace["time"] = lambda: max(row["planned_end_epoch"] for row in namespace["groups"]()[group - 1])
    def capture(argv, **kwargs):
        child_nodes = kwargs["env"]["HYGON_STANDBY_ONLY_NODES"].split()
        assert argv.count("--child") == 1 and argv[-4] == "--child"
        assert re.fullmatch(r"[a-f0-9]{32}", argv[-3])
        assert argv[-2:] == ["--group", str(group)]
        assert Path(argv[4]) == Path(ordinary.__file__).resolve()
        seen.append({"nodes": child_nodes, "child_argv": argv[:-3] + ["<fixture UUID>"] + argv[-2:]})
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
    namespace["subprocess"] = SimpleNamespace(run=capture, SubprocessError=subprocess.SubprocessError,
                                               TimeoutExpired=subprocess.TimeoutExpired)
    base = {"store": arbiter.publish, "did_not_send": lambda stdout, rc: rc == 0}
    # Preserve original ordinary exclusive-file semantics in this local fixture.
    def original_store(path, value):
        with path.open("xb") as handle:
            handle.write(arbiter.encode(value))
    base["store"] = original_store
    publisher = {"precheck": lambda: {"head": "<fixture publication>"}}
    source_rows = namespace["groups"]()[group - 1] if rows is None else rows
    result = namespace["refill"](group, source_rows, base, publisher)
    assert len(seen) <= 1
    if seen:
        assert result["terminal_ready_nodes"] == seen[0]["nodes"]
    return seen[0] if seen else {"nodes": []}


def check(path):
    records = {}
    fixture = path / "ordinary-first"
    journal, ns, ledger, group, target, _, _, accept = define(fixture)
    original_candidates = ns["groups"]()
    assert len({row["node"] for group_rows in original_candidates for row in group_rows}) == 70
    assert any(row["node"] == "f13r1n12" for row in original_candidates[10])
    first = capture_refill(ns, group, [target], fixture / "capture")
    assert first["nodes"] == [arbiter.NODE]
    rejected(accept, "EXISTING_ORDINARY_TICKET")
    records["ordinary_already_has_ticket"] = {"real_reserve_capture": first, "business": "REFUSED"}

    fixture = path / "business-unsent"
    journal, ns, ledger, group, target, scope, scope_sha, accept = define(fixture)
    accepted, receipt_sha = accept()
    ready = ns["groups"]()[group - 1]
    held = capture_refill(ns, group, ready, fixture / "capture")
    expected_other = [row["node"] for row in ready if row["node"] != arbiter.NODE]
    assert held["nodes"] == expected_other and len(expected_other) == 7
    assert not any(journal.glob(arbiter.NODE + "-" + arbiter.PREDECESSOR + "-[012].json"))
    value, _ = ledger.release_unsent(receipt_sha, NOW + 1)
    assert value["state"] == "RELEASED_POSITIVE_UNSENT"
    source_rows = ordinary.resume_inputs(ns, ledger)
    assert source_rows == [target]
    reopened = capture_refill(ns, group, [target], fixture / "reopened", rows=source_rows)
    assert reopened["nodes"] == [arbiter.NODE] and ordinary.refill_argv(group)[-1] == "--resume-pack16"
    records["business_positive_never_sent"] = {"before_release": held, "after_release": reopened,
                                                "released": value["state"]}

    fixture = path / "business-unknown"
    _, ns, ledger, group, target, scope, scope_sha, accept = define(fixture)
    _, receipt_sha = accept()
    _, grant_sha = ledger.grant(receipt_sha, NOW + 1)
    rejected(lambda: ledger.grant(receipt_sha, NOW + 2), "NO_SEND_GRANT")
    rejected(lambda: ledger.release_unsent(receipt_sha, NOW + 2), "SEND_POSSIBLE_OR_NO_ACCEPTANCE")
    assert capture_refill(ns, group, [target], fixture / "capture")["nodes"] == []
    assert ledger.get("release") == (None, None)
    rejected(lambda: ordinary.resume_inputs(ns, ledger), "PACK16_REFILL_NOT_RELEASED")
    assert capture_refill(ns, group, [target], fixture / "late-capture")["nodes"] == []
    records["business_unknown"] = {"ordinary_nodes": [], "second_business_grant": "REFUSED",
                                    "expiry_does_not_reopen": True}

    fixture = path / "business-final"
    _, ns, ledger, group, target, scope, scope_sha, accept = define(fixture)
    _, receipt_sha = accept()
    _, grant_sha = ledger.grant(receipt_sha, NOW + 1)
    host = pack16_host.load_baseline()
    rejected(lambda: host.ordinary_exclusion(ordinary.ROOT, CALENDAR, arbiter.NODE),
             "EXACT_ORDINARY_REFILL_EXCLUSION_UNAVAILABLE")
    rejected(lambda: pack16_host.configure(host, ledger.directory / "grant.json", grant_sha,
                                            fixture / "scope.json", scope_sha, FIXTURE_OUTPUT + "0",
                                            clock=lambda: NOW + 2), "PACK16_EXCLUSIVE_OUTPUT")
    pack16_host.configure(host, ledger.directory / "grant.json", grant_sha, fixture / "scope.json",
                          scope_sha, FIXTURE_OUTPUT, clock=lambda: NOW + 2)
    host.ordinary_exclusion(ordinary.ROOT, CALENDAR, arbiter.NODE)
    row = ["1103606.3099", arbiter.LABEL, "COMPLETED", "0:0", "11", arbiter.NODE,
           "cpu=4,gres/dcu=1,mem=8192M,node=1"]
    host_document = {"state": "TERMINAL", "actual_resources_match": True, "scope_SHA256": scope_sha, "accounting": [row]}
    out = fixture / "host-terminal.json"
    host_pin = host.exclusive(out, host_document)
    host_document, host_sha = arbiter.read(out, host_pin["SHA256"])
    assert host_document["business_grant_SHA256"] == grant_sha
    final = {"request": arbiter.REQUEST, "actual_gpu_step": row[0], "independent_post": "CLEAN",
             "raw_worker_audit": {"actual_accounting": row, "scope_SHA256": scope_sha, "accounting_SHA256": host_sha}}
    bad = dict(final, independent_post="UNVERIFIED")
    rejected(lambda: ledger.release_post(receipt_sha, host_document, host_sha, bad,
                                        arbiter.signature(arbiter.encode(bad)), NOW + 3), "INDEPENDENT_POST_REQUIRED")
    assert capture_refill(ns, group, [target], fixture / "blocked")["nodes"] == []
    final_value, _ = ledger.release_post(receipt_sha, host_document, host_sha, final,
                                        arbiter.signature(arbiter.encode(final)), NOW + 4)
    assert final_value["state"] == "RELEASED_AFTER_TERMINAL_AND_POST"
    source_rows = ordinary.resume_inputs(ns, ledger)
    reopened = capture_refill(ns, group, [target], fixture / "reopened", rows=source_rows)
    assert source_rows == [target] and reopened["nodes"] == [arbiter.NODE]
    records["business_terminal_and_clean_post"] = {"before_post": "HELD", "after_post": reopened,
                                                    "grant_passed_original_host_gate": True}
    return records


def check_missing_worker_cleanup(path):
    baseline = pack16_host.load_baseline()
    baseline.time = SimpleNamespace(time=lambda: NOW + 6)
    boot = "00000000-0000-4000-8000-000000000001"
    pcis = [f"0000:{slot:02x}:00.0" for slot in range(8)]
    pre_step, actual_step, post_step = "1103606.3097", "1103606.3099", "1103606.3100"

    def inventory(step):
        return {"node": arbiter.NODE, "step": step, "boot_id_SHA256": arbiter.signature((boot + "\n").encode()),
                "PCIs": pcis, "kfd_process_count": 0, "own_device_fd_process_count": 0, "occupied_requested_ports": []}

    # These receipts model a real allocated step ending before the worker writes a result.
    pre = {"phase": "pre", "observed_epoch": NOW - 2, "inventory": inventory(pre_step),
           "accounting": [pre_step, "fixture-pre", "COMPLETED", "0:0", "1", arbiter.NODE, "cpu=1,mem=256M,node=1"],
           "assigned_PCI": pcis[0], "gfx": "gfx936", "exclusive_1_DCU_lease": True}
    pre_sha = arbiter.signature(arbiter.encode(pre))
    extra = {"source_publication": baseline.PUBLICATION, "runtime_SHA256": baseline.RUNTIME_SHA,
             "torch_version": "2.10.0", "torch_hip": "6.3.26113", "protected_nodes": ["f12r2n11"],
             "PCI": pcis[0], "boot_id": boot, "pre_step": pre_step, "pre_SHA256": pre_sha,
             "protection_authority_SHA256": "a" * 64}
    _, ns, ledger, group, target, scope, scope_sha, accept = define(path, extra)
    _, accepted_sha = accept()
    _, grant_sha = ledger.grant(accepted_sha, NOW + 1)
    row = [actual_step, arbiter.LABEL, "FAILED", "1:0", "2", arbiter.NODE, "cpu=4,gres/dcu=1,mem=8G,node=1"]
    host = {"request": arbiter.REQUEST, "state": "TERMINAL", "actual_resources_match": True,
            "scope_SHA256": scope_sha, "business_grant_SHA256": grant_sha, "accounting": [row],
            "terminal_observed_epoch": NOW + 4, "accounting_columns": list(baseline.COLUMNS),
            "planned": baseline.DENOMINATOR}
    host_path = path / "host-terminal.json"
    host_sha = arbiter.publish(host_path, host)
    post = {"phase": "post", "observed_epoch": NOW + 5, "inventory": inventory(post_step),
            "accounting": [post_step, "fixture-post", "COMPLETED", "0:0", "1", arbiter.NODE, "cpu=1,mem=256M,node=1"],
            "after_gpu_step": actual_step, "after_terminal": True,
            "terminal_row_SHA256": arbiter.signature(arbiter.encode(row)), "independent_gpu_accounting": row}
    runtime = ordinary.ROOT / "local-only/O01-pack16-preflight-v3-c0243062d1f24b70aee7e3092c24e8f0/runtime"

    def finish(proof):
        return baseline.independently_finish(ordinary.ROOT, scope, scope_sha, pre, proof, host,
                                             host_path, host_sha, runtime, path / "scope.json", path)

    dirty = dict(post, inventory=dict(post["inventory"], kfd_process_count=1))
    rebooted = dict(post, inventory=dict(post["inventory"], boot_id_SHA256="b" * 64))
    moved_pci = dict(post, inventory=dict(post["inventory"], PCIs=[*pcis[:-1], "0000:08:00.0"]))
    for evidence, gate in (
        (dirty, "INDEPENDENT_CLEAN_INVENTORY"),
        (rebooted, "INDEPENDENT_CLEAN_INVENTORY"),
        (dict(post, observed_epoch=NOW + 3), "POST_BEFORE_TERMINAL"),
        (moved_pci, "PCI_INVENTORY_MOVED"),
    ):
        rejected(lambda: finish(evidence), gate)
        assert ledger.get("release") == (None, None)
    assert capture_refill(ns, group, [target], path / "blocked")["nodes"] == []
    final = finish(post)
    audit = final["raw_worker_audit"]
    assert audit["real_GPU_terminal"] and not audit["worker_result_passed"] and "worker_terminal_SHA256" not in audit
    assert final["independent_post"] == "CLEAN" and final["planned"] == baseline.DENOMINATOR
    assert not final["component_timing_candidate"] and not final["promotion_allowed"]

    worker = path / "worker"
    worker.mkdir()
    (worker / "terminal.json").write_bytes(b"invalid JSON")
    malformed = finish(post)
    assert "reader_failure" in malformed["raw_worker_audit"]
    assert not malformed["raw_worker_audit"]["worker_result_passed"] and not malformed["component_timing_candidate"]

    final_path = path / "independent-final.json"
    final_sha = arbiter.publish(final_path, final)
    released, _ = ledger.release_post(accepted_sha, host, host_sha, final, final_sha, NOW + 6)
    assert released["state"] == "RELEASED_AFTER_TERMINAL_AND_POST"
    resumed = capture_refill(ns, group, [target], path / "reopened", rows=ordinary.resume_inputs(ns, ledger))
    assert resumed["nodes"] == [arbiter.NODE]
    return {"missing_worker_report": True, "malformed_worker_audit_negative": True,
            "reopened": resumed, "rejected": ["dirty", "rebooted", "early_post", "moved_PCI"],
            "component_pass": False, "planned": final["planned"]}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("expected one absent local private receipt path")
    path = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="pack16-ordinary-capture-") as temp:
        root = Path(temp).resolve()
        cases = check(root)
        cases["failed_step_without_worker_result"] = check_missing_worker_cleanup(root / "cleanup-only")
    sources = {name: arbiter.signature((arbiter.PACKAGE / name).read_bytes())
               for name in ("arbiter.py", "ordinary.py", "pack16_host.py", "check_local.py")}
    receipt = {"schema": arbiter.SCOPE_SCHEMA + "-local", "cases": cases, "source_SHA256": sources,
               "ordinary_original_SHA256": ordinary.ENTRY_SHA, "calendar_SHA256": arbiter.CALENDAR_SHA,
               "original_host_SHA256": arbiter.BASELINE_SHA, "target_group": ordinary.GROUP,
               "planned_ordinary_nodes": 70, "other_nodes_unchanged": 69, "gemma_group": 11,
               "remote_calls": 0, "actual_Slurm_calls": 0, "business_launches": 0, "real_claims_made": 0}
    print(json.dumps({"receipt_SHA256": arbiter.publish(path, receipt), "cases": len(cases)}))


if __name__ == "__main__":
    main()
