"""Exercise the actual source, readback, launch and terminal gates with real local train streams."""

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from prepare_packet import ROOT, REF, expected_members, packet
import runner as run


def reviewed_train_bytes():
    prior = ROOT / "code/bw1000/training-gemma-vizwiz-reviewed-train-stage-v1"
    module_spec = importlib.util.spec_from_file_location("pinned_vizwiz_seven_stage", prior / "launch_once.py")
    module = importlib.util.module_from_spec(module_spec)
    sys.path.insert(0, str(prior))
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop("receive_once", None)
    claim = json.loads((prior / "claim.json").read_bytes())
    release = ROOT / "local-only/specialized06-vizwiz-native-cpu-pilot-8-image-release-v1.json"
    identities, body = module.prepare_packet(release, claim["quality_receipt_SHA256"])
    assert all(claim[key] == value for key, value in identities.items())
    start = claim["binding"]["bytes"]
    return body[start:start + claim["rows"]["bytes"]]


def rejected(reason, action):
    try:
        action()
    except ValueError as error:
        assert str(error) == reason, (reason, str(error))
        return str(error)
    raise AssertionError(reason)


def digest(raw):
    return run.pin(raw)["SHA256"]


def stamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def main():
    assert len(sys.argv) == 2
    receipt = Path(sys.argv[1]).resolve()
    body = packet(ROOT)
    _, _, closure = run.decode_packet(body, digest(body))
    request = "0a" * 16
    proof = run.encode({"schema": run.SCHEMA + "-publication", "request": request, "ref": REF,
                        "commit": "1" * 40, "tree": "2" * 40, "source_closure_SHA256": run.CLOSURE_SHA,
                        "commit_verified_in_ref": True, "members": expected_members(ROOT, closure)})
    malformed = json.loads(body)
    del malformed["files"][next(iter(malformed["files"]))]
    malformed = run.encode(malformed)
    denied = {}
    with TemporaryDirectory(prefix="gemma-index-target-run-local-", dir=ROOT / "local-only") as scratch:
        root = Path(scratch).resolve()
        stage = root / run.scope_name(request)
        view = root / (stage.name + "-view") / "train800-swift.jsonl"
        out = root / (stage.name + "-result")

        def install(raw, sha):
            return run.install_sources(root, request, raw, sha, proof, digest(proof))

        denied["missing_source"] = rejected("EXACT_SOURCE_MEMBER_SET", lambda: install(malformed, digest(malformed)))
        denied["wrong_packet_sha"] = rejected("PACKET_DIGEST", lambda: install(body, "0" * 64))
        assert not stage.exists()
        installed = install(body, digest(body))
        denied["duplicate_install"] = rejected("REQUEST_ALREADY_INSTALLED", lambda: install(body, digest(body)))
        five = ROOT / "local-only/specialized06-vizwiz-native-target-binding-v1/train-swift.jsonl"
        reviewed = root / "reviewed.jsonl"
        run.write(reviewed, reviewed_train_bytes())
        original_paths = {"five": five, "reviewed": reviewed}
        view_report = run.build_private_view(stage, request, original_paths)
        denied["duplicate_private_view"] = rejected("REQUEST_ALREADY_INSTALLED",
                                                    lambda: run.build_private_view(stage, request, original_paths))
        epoch = 1795560000
        readback = run.independent_readback(stage, request, view, original_paths, epoch)
        target = run.PROJECT / "incoming"
        parent, node = "900001", "f13r1n12"
        scope = {"schema": run.SCHEMA + "-scope", "request": request, "root_admitted": True, "replay_allowed": False,
                 "parent": parent, "node": node, "resources": {"CPU": 4, "memory_MiB": 8192, "GPU": 0},
                 "view_SHA256": run.VIEW_SHA, "placement_receipt_SHA256": "8" * 64, "epoch": epoch,
                 "deadline": epoch + run.EPOCH_SECONDS, "target_stage": str(target / stage.name),
                 "target_view": str(target / view.parent.name / view.name), "target_output": str(target / out.name),
                 "readback_SHA256": digest(run.encode(readback)), "publication_SHA256": digest(proof)}
        renders = []
        admissions = []

        def placement(scope_observed, label):
            assert scope_observed == scope and not out.exists()
            admissions.append(label)

        def submit(argv, scoped_output, deadline):
            intent = json.loads((scoped_output / "SUBMISSION_INTENT.json").read_bytes())
            assert intent["numeric_step"] is None and intent["argv_SHA256"] == digest(run.encode(argv))
            assert deadline == scope["deadline"]
            renders.append(argv)
            return 1

        def dispatch():
            return run.submit_once(stage, view, original_paths, scope, request, readback, out, placement=placement,
                                   submit=submit, clock=lambda: epoch + 1)

        damaged = stage / run.DEPLOYMENT / "index_entry.py"
        exact = damaged.read_bytes()
        damaged.unlink()
        denied["missing_installed_source_before_dispatch"] = rejected("EXACT_MEMBER_PATH", dispatch)
        damaged.write_bytes(exact + b"\n")
        denied["wrong_installed_source_before_dispatch"] = rejected("EXACT_MEMBER_BYTES", dispatch)
        damaged.write_bytes(exact)
        reviewed_exact = reviewed.read_bytes()
        reviewed.write_bytes(reviewed_exact + b"\n")
        denied["wrong_original_stream_before_dispatch"] = rejected("EXACT_MEMBER_BYTES", dispatch)
        reviewed.write_bytes(reviewed_exact)
        assert not admissions and not renders and not out.exists()
        submission = dispatch()
        denied["duplicate_dispatch"] = rejected("REQUEST_ALREADY_SUBMITTED", dispatch)
        assert len(renders) == len(admissions) == 1 and submission["qualified_index_rows"] == 0
        assert readback["compute_node_checked"] is False
        intent = json.loads((out / "SUBMISSION_INTENT.json").read_bytes())
        step = parent + ".401"
        row = [step, admissions[0], "FAILED", "1:0", "2", node, "cpu=4,mem=8G,node=1",
               stamp(epoch + 3), stamp(epoch + 5)]

        def terminal(parts, step_id=step, evidence=intent):
            return run.independent_terminal(stage, request, scope, evidence, "|".join(parts), step_id)

        actual_terminal = terminal(row)
        late = row[:]
        late[4], late[8] = "121", stamp(epoch + 124)
        denied["exceeds_original_120_second_budget"] = rejected("ACTUAL_CPU_ELAPSED_OR_DEADLINE", lambda: terminal(late))
        early = row[:]
        early[4], early[7] = "5", stamp(epoch)
        denied["predates_durable_intent"] = rejected("ACTUAL_CPU_ELAPSED_OR_DEADLINE", lambda: terminal(early))
        dcu = row[:]
        dcu[6] += ",gres/dcu=1"
        denied["nonzero_dcu_accounting"] = rejected("ACTUAL_ZERO_GRES_ACCOUNTING", lambda: terminal(dcu))
        moved = row[:]
        moved[1] += "-wrong"
        denied["different_job_name"] = rejected("ACTUAL_ZERO_GRES_ACCOUNTING", lambda: terminal(moved))
        denied["non_numeric_or_other_parent"] = rejected("ACTUAL_NUMERIC_STEP", lambda: terminal(row, "900002.401"))
        denied["different_start_claim"] = rejected("ACTUAL_REQUEST_EPOCH_BINDING",
                                                  lambda: terminal(row, evidence=dict(intent, scope_SHA256="0"*64)))
        assert actual_terminal["planned_train_rows"] == 12 and actual_terminal["qualified_index_rows"] == 0
        report = {"schema": run.SCHEMA + "-local", "source_stage": installed["state"],
                  "source_closure_SHA256": run.CLOSURE_SHA, "source_packet": run.pin(body),
                  "packet_members": len(closure["members"]), "private_view": view_report["view"],
                  "private_view_rows": 800, "local_actual_train_stream_rows": [5, 7],
                  "captured_argv_SHA256": digest(run.encode(renders[0])),
                  "private_view_readback": readback["view"] == view_report["view"],
                  "original_readonly_train_sources": readback["original_readonly_train_sources"],
                  "actual_runner_preflight_calls_with_captured_submission": len(renders),
                  "synthetic_publication_placement_and_accounting_for_local_control_flow": True,
                  "captured_zero_gres_and_scoped_mounts": True, "local_terminal_gate": actual_terminal["state"],
                  "planned_train_rows": 12, "qualified_native_index_rows": 0,
                  "native_CPU_calls": 0, "Slurm_calls": 0, "GPU": 0, "target_staged": False,
                  "source_publication_verified": False, "protected_private_rows_persisted_by_test": False,
                  "rejections": denied}
    raw = run.encode(report)
    run.write(receipt, raw)
    print(json.dumps(run.pin(raw), sort_keys=True))


if __name__ == "__main__":
    main()
