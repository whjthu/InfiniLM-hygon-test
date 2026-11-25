"""Local proof of the real provenance, frozen Dev24 reader and captured GPU-owner geometry."""

from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory

import audit_terminal as independent
import authority as auth
import launch


ROOT = launch.REPO
PRIVATE = ROOT / "local-only"
CPU = PRIVATE / ("vggt-original1318-dev24-r16-real-admission-preparation-v2-20261125/"
                 "root-independent-result-99b05aafc5ca43e9a734f27f715f89f4.json")
HEAD = PRIVATE / ("training80-fixed-hour-transport-0e1a294fabfc4854854c09ed6b4d6f83/"
                  "vggt-3090-original-head-authority-only-20261125.json")


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def no(reason, call):
    try:
        call()
    except ValueError as error:
        assert str(error) == reason, (reason, str(error))
        return reason
    raise AssertionError(reason)


def main(path):
    scope = {"request": "b"*32, "parent": "1103606", "node": "f13r1n09", "only_exclusive_node": True,
             "approved_operational_scope": True}
    raw = auth.read(CPU, 2100, auth.AUDIT3090_SHA)
    readback = auth.cpu_evidence(launch.read_local(auth.PUBLIC_3090, auth.PUBLIC_3090_SHA), raw)
    true_head = auth.js(HEAD, auth.HEAD_READER_SHA)["response"]
    assert true_head["provider_SHA256"] == auth.PROVIDER_SHA and true_head["result_SHA256"] == auth.CPU_RESULT_SHA
    assert true_head["intent_SHA256"] == readback["intent_SHA256"] and true_head["previous_same_input_head_identities"] == 5
    real_head = true_head["authority"]
    assert real_head["file_SHA256"] == auth.HEAD_FILE_SHA and real_head["module_SHA256"] == auth.HEAD_MODULE_SHA
    source, old, label = launch.render(scope)
    score = source[auth.SCORER]
    execution = launch.execution_identity(score)
    assert all(row["execution_SHA256"] == execution and row["base"] == auth.BASE for row in old.values())
    canonical = launch.original_module(launch.AUDIT, launch.AUDIT_SHA)
    actual = canonical["local"](canonical["public"](), score)
    assert actual["common_pose_input_windows"] == 22 and actual["original_dev24"] == 24
    assert actual["missing_original_windows"] == 2 and actual["train_recording_overlap_windows"] == 0
    p1506 = dict(file_SHA256=auth.OLD_RUNS["1506"]["file"], module_SHA256=auth.OLD_RUNS["1506"]["module"],
                 scorer_execution_SHA256=execution, inputs_SHA256=auth.INPUT_SHA, base=auth.BASE)
    assert not auth.compatible(real_head, p1506, execution)
    same = dict(p1506, module_SHA256=real_head["module_SHA256"], file_SHA256=real_head["file_SHA256"])
    assert auth.compatible(real_head, same, execution)
    assert not auth.compatible(real_head, dict(same, inputs_SHA256="0"*64), execution)
    assert not auth.compatible(real_head, dict(same, scorer_execution_SHA256="0"*64), execution)
    assert not auth.compatible(real_head, dict(same, base={}), execution)
    original_n09 = PRIVATE / "training80-1738-ordinary-renewal-dispatch/f13r1n09-1103606.2933-0.json"
    marker_pin = "a6d1b68104d181a2057ef7662a236ec78d2c0d8297946d03570d9d501ed75111"
    marker = auth.js(original_n09, marker_pin)
    archived = auth.js(PRIVATE / "training80-0544-ordinary-58-verified-source-manifest-v1.json",
                       "7eaf3eb6245278d776fbad5e2c51af5c272ece6a755588268c95879817d50e54")
    assert marker["batch"] == 4 and marker["predecessor"] == "1103606.2933" and marker["replay_allowed"] is False
    assert scope["node"] in archived["missing_classification"]["admission_process_prebridge_exit_rc255_original_journal_preserved"]
    assert all(not original_n09.with_name(f"f13r1n09-1103606.2933-{slot}.json").exists() for slot in (1, 2))
    refuse = [no("CANDIDATE_STEP_IS_NEVER_A_PRECEDENT_ADAPTER", lambda: auth.compatible(
        dict(real_head, gpu_step="1103606.498"), p1506, execution)),
        no("EXPLICIT_ONE_EXCLUSIVE_APPROVED_PLACEMENT", lambda: launch.render(dict(scope, only_exclusive_node=False)))]

    begin = 1795569000
    body, offer = launch.frozen(scope, "c"*40, begin, raw)
    preflight, no_offer = launch.frozen(scope, "c"*40, begin, raw, preflight=True)
    assert no_offer is None and b"'independent_preflight_only': True" in preflight
    with TemporaryDirectory(prefix="vggt-1318-paired-source-", dir=PRIVATE) as directory:
        t = Path(directory).resolve()
        for file, data in source.items():
            out = t / file
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        sys.path.insert(0, str(t / "tools"))
        try:
            rendered = runpy.run_path(str(t / launch.CONTROLLER), run_name="frozen_1506_exec_boundary")
            args = rendered["step_command"]("gpu", rendered["gpu_command"](t))
        finally:
            sys.path.pop(0)
        assert rendered["STAGES"]["gpu"] == (1, 4, 65536, 1200)
        assert rendered["PARENT"] == scope["parent"] and rendered["NODE"] == scope["node"]
        assert b"['I02'][NODE]['state']" in source[launch.CONTROLLER]
        assert [a for a in args if a.startswith("--gres=")] == ["--gres=dcu:1"]
        assert "-c4" in args and "--mem=65536M" in args and "/candidate:ro" in " ".join(args)
        assert "/dev24:ro" in " ".join(args) and auth.TRAIN_DIR in " ".join(args)
        assert all("/alternate:" not in a and "/lowlr:" not in a and "training-vggt-tum-bidirectional30-v1" not in a for a in args)
        normalized = [arg.replace(str(t), "<request-root>").replace(t.as_posix(), "<request-root>") for arg in args]
    intent = {"request": scope["request"], "publication": "c"*40, "parent": scope["parent"], "node": scope["node"],
              "max_seconds": 1500, "epoch": begin, "deadline": begin+1800}
    counts = {"pre": ("201", "cpu=2,mem=2G,node=1", 5, 7),
              "gpu": ("202", "cpu=4,gres/dcu=1,mem=64G,node=1", 10, 135),
              "post": ("203", "cpu=2,mem=2G,node=1", 137, 140)}
    rows = {phase: ["1103606."+step, label+"-"+phase, "COMPLETED", "0:0", str(end-start), scope["node"],
                    res, stamp(begin+start), stamp(begin+end)] for phase, (step, res, start, end) in counts.items()}
    outcome = {"request": scope["request"], "publication": "c"*40, "stages": {key: {"row": row[:7]} for key, row in rows.items()}}
    assert len(independent.timeline(intent, outcome, rows)) == 3
    invalid = dict(rows)
    invalid["gpu"] = [*rows["gpu"][:7], stamp(begin+10),stamp(begin+1211)]
    invalid["gpu"][4] = "1201"
    mod = dict(outcome, stages=dict(outcome["stages"], gpu={"row": invalid["gpu"][:7]}))
    refuse.append(no("EXACT_PHASE_EPOCH_ELAPSED_AND_CHRONOLOGY", lambda: independent.timeline(intent, mod, invalid)))
    refuse.append(no("INDEPENDENT_EXACT_CPU_MEMORY_NODE",
        lambda: independent.resources("cpu=4,gres/dcu=2,mem=64G,node=1,billing=4", 1,4,65536)))
    pci = [f"0000:0{i}:00.0" for i in range(1,9)]
    dev = {"node": scope["node"], "full_fd_inventory_performed": True, "kfd_holder_count": 0, "boot_id":"current-boot", "pci": pci}
    before, after = dict(dev, step=rows["pre"][0]), dict(dev, step=rows["post"][0])
    assignment = {"step": rows["gpu"][0], "node": scope["node"], "pci": pci[1],
                  "kfd":{"pci": pci[1], "gpu_id": 17, "properties_SHA256": "4"*64}}
    provider = {"path":"/runtime/libamdhip64.so", "bytes":64, "SHA256":"5"*64}
    probe = {"observed_epoch": begin+142, "nodes": {scope["node"]: {"device_probe_step":"1103606.204",
        "parent":scope["parent"], "device_probe_observed":True, "KFD_clean":True, "PID_clean":True, "active_steps":[], "PCIs":pci}}}
    args = (before, after, assignment, provider, probe, rows["post"], rows["gpu"], intent)
    assert independent.verify_post(*args) == "1103606.204"
    refuse.append(no("FRESH_INDEPENDENT_POST_PCI_KFD_AND_PID", lambda: independent.verify_post(
        *args[:4], dict(probe, observed_epoch=begin+136), *args[5:])))
    refuse.append(no("EXACT_SAME_BOOT_8_PCI_AND_CLEAN_FDS", lambda: independent.verify_post(
        before, dict(after, boot_id="rebooted"), *args[2:])))
    report = {"state": "REAL_LOCAL_3090_HEAD_AND_22_DEV24_INPUTS_FROZEN", "actual_3090": readback["accounting_nine_columns"],
              "target_CPU_provider_version": readback["provider"]["version"], "true_head": real_head,
              "local_same_input_windows_tested": actual["common_pose_input_windows"], "prior_binary_scorer_identities": old,
              "scope_fixture_only": scope, "budget_and_denominators": launch.PLAN,
              "actual_candidate_scorer_SHA256": auth.sha(score), "normalized_math_identity": execution,
              "rendered_source_members": len(source), "bounded_body_sha_and_bytes": [auth.sha(body),len(body)],
              "bounded_offer_sha_and_bytes": [auth.sha(offer),len(offer)],
              "gpu_argv_captured_sha": auth.sha(auth.encode(normalized)),
              "local_target_precondition_checks": refuse, "actual_target_GPU": 0, "SCNet":0, "Slurm":0, "Git":0,
              "actual_preds":0, "actual_optimizer_updates":0, "reservation_created":False}
    report["n09_original_ordinary_reservation_preserved"] = {"predecessor": marker["predecessor"], "slot0_SHA256": marker_pin,
                                                              "slots1_and_2_present": 0, "proves_current_admission": False}
    raw = auth.encode(report)
    with path.open("xb") as out:
        out.write(raw)
    print(json.dumps({"bytes":len(raw), "SHA256":auth.sha(raw)}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
