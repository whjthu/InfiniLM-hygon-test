"""Build the private source packet and exercise the real stage, CPU boundary and auditors locally."""

import argparse
import base64
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import runpy
import shlex
import sys
from tempfile import TemporaryDirectory

import host
import install_bootstrap
import verify_terminal as terminal
import worker


ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "local-only"
OLD = LOCAL / "vggt-original1318-dev24-target-source-handoff-20261125/request-e31048eef0024ce1b13725d222ddd62a/source-packet.json"
NUMPY173 = ROOT / "results/bw1000/2026-11-12/training-vggt-original173-full-framework-native-HIP-provider-CPU.json"
NUMPY173_SHA = "90cf4af6550fc0954407501a21027f3f08f4a84794bd8bbf24a7aa24aa94fda5"
ORIGINAL_CPU = ROOT / "code/bw1000/training-vggt-native-provider-HIP-CPU-v2/receive_once.py"
CPU1509 = ROOT / "results/bw1000/2026-11-23/training-vggt-two-scene-dev24-v2-original1506-terminal.json"
CPU1509_SHA = "5eca19c83c42ade3a61836285f4f4c5fdfb07621673d0de8a4b83bb24724f952"
HOST_FAILURE = ROOT / "results/bw1000/2026-11-25/vggt-1318-dev24-original-host-numpy-prerequisite-failure.json"
HOST_FAILED_SHA = "661d8e526691a92736b7886c13827066c135364acfb0c6f70dc5b535651c4b5a"


def b64(raw):
    return base64.b64encode(raw).decode("ascii")


def precedent():
    first = json.loads(host.read(NUMPY173, 256 * 1024, NUMPY173_SHA))
    first_command = host.read(ORIGINAL_CPU, 65536, host.SITE_PRECEDENT)
    score = json.loads(host.read(CPU1509, 65536, CPU1509_SHA))
    stopped = json.loads(host.read(HOST_FAILURE, 65536, HOST_FAILED_SHA))
    assert first["accounting"][0][0] == "1087656.173" and first["CPU_result"]["GPU_cards"] == 0
    assert first["CPU_result"]["versions"]["numpy"] == "1.25.0"
    assert all(anchor in first_command for anchor in (b"training-t06-sdxl-runtime-v1", b"dtk26-sandbox-r1",
                b"/training-runtime/site", b"/public/software/apps/apptainer/1.3.4/bin/apptainer", b"--gres=none"))
    assert score["cpu_audit_actual_accounting"][0] == "1103606.1509" and score["cpu_audit_private_bytes"] == 4661
    assert score["request"] == "6bd97bc61f0249c0a255ad26fe55a162" and score["cpu_audit_actual_accounting"][6] == "cpu=2,mem=2G,node=1"
    assert stopped["target_result"]["rc"] == 2 and stopped["target_result"]["reason"] == "HOST_NUMPY_UNAVAILABLE"
    assert stopped["request"] == "e31048eef0024ce1b13725d222ddd62a" and stopped["new_shared_host_source_installs"] == 0
    return {"historical_cpu173_actual_numpy_version": "1.25.0", "historical_cpu173_command_SHA256": host.SITE_PRECEDENT,
            "historical_cpu173_receipt_SHA256": NUMPY173_SHA, "historical_1509_accounting": score["cpu_audit_actual_accounting"],
            "historical_1509_audit": {"bytes": score["cpu_audit_private_bytes"], "SHA256": score["private_audit_SHA256"]},
            "historical_1509_exact_launcher_or_numpy_ABI_available": False, "previous_HOST_failure_SHA256": HOST_FAILED_SHA,
            "old_HOST_retry": False}


def build():
    source = host.read(OLD, 200000, host.SOURCE_PACKET)
    new = {name: b64((Path(__file__).resolve().parent / name).read_bytes()) for name in host.BOOT_NAMES}
    raw = worker.encode({"old_source_packet": b64(source), "cpu_sources": new})
    return raw, host.boot_identities()


def reject(reason, action):
    try:
        action()
    except ValueError as ex:
        assert str(ex) == reason, (reason, str(ex))
        return reason
    raise AssertionError(reason)


def day(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def replay(value, boot):
    request, parent, node = "46"*16, "1194900", "f14r3n06"
    epoch = 1795559200
    env = {"SLURM_JOB_ID": parent, "SLURM_STEP_ID": "991", "SLURM_CPUS_PER_TASK": "2", "SLURM_JOB_GPUS": "0,1,2,3",
           "CUDA_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1"}
    deny = {}
    real = importlib.util.find_spec("numpy")
    roots = (Path(real.origin).resolve().parent.parent,)
    locate = lambda: worker.numpy_provider(roots)
    with TemporaryDirectory(prefix="vggt-1318-cpu-readback-local-", dir=LOCAL) as scratch:
        incoming = Path(scratch).resolve()
        packet_sha = worker.pin(value)["SHA256"]
        scoped = incoming / host.scope_name(request)
        deny["bad_source_sha"] = reject("EXACT_SOURCE_ONLY_OFFER", lambda: host.install(incoming, request, value, "0"*64, boot))
        install = install_bootstrap.load_install(value, packet_sha, boot, incoming, request)
        records = host.installed(scoped, request, install["SHA256"], boot)
        assert len(records["members"]) == 21
        deny["duplicate_stage"] = reject("SCOPED_INSTALL_ABSENT", lambda: host.install(incoming, request, value, packet_sha, boot))

        actual_audit_entry = scoped / "checkout" / worker.ENTRY
        recipe = runpy.run_path(str(actual_audit_entry), run_name="actual_staged_frozen_1318_audit")
        proof = recipe["public"]()
        stored = recipe["pinned_scorer"]()
        real_scorer = recipe["candidate_source"](proof, stored[recipe["SCORER"]])
        compile(real_scorer, "actual .1318 unchanged scorer", "exec")
        assert worker.pin(real_scorer)["SHA256"] == "953ca764a83dfa607f5f0f8396d5f3c975fdc2711b9be1ba21f4ed8210cf0220"

        placement = {"request": request, "parent": parent, "node": node, "epoch": epoch, "deadline": epoch + worker.LIMIT,
                     "node_observed_epoch": epoch, "CPU_only_admitted": True,
                     "resources": {"nodes": 1, "CPU": 2, "memory_MiB": 2048, "GPU": 0}}
        built = []
        def live(p, n, r):
            assert (p, n, r) == (parent, node, request)
        def captured(argv, output, deadline):
            assert deadline == epoch + worker.LIMIT + 5 and (output/"INTENT.json").exists()
            assert all(flag in argv for flag in ("--gres=none", "--cpus-per-task=2", "--mem=2048M", "--ntasks=1", "--nodes=1"))
            assert not any(flag in argv for flag in ("--rocm", "--nv"))
            assert sum(item.endswith(":rw") for item in argv) == 1 and "source /opt/dtk/env.sh" in argv[-1]
            built.append(argv)
            return 1
        def attempt():
            return host.once(scoped, request, placement, "4"*64, install["SHA256"], dispatch=captured, verify_live=live,
                             clock=lambda: epoch + 1)
        result = attempt()
        deny["duplicate_submission"] = reject("SINGLE_CPU_SUBMISSION_ONLY", attempt)
        assert len(built) == 1 and result["srun_rc"] == 1

        bind = worker.binding(request, parent, node, epoch, epoch + 2, env, node, 2)
        device = dict(env, SLURM_TRES_PER_TASK="cpu=2,gres/dcu=1")
        deny["compute_dcu"] = reject("EXACT_ZERO_GRES_WORKER_ENVIRONMENT",
                                    lambda: worker.binding(request, parent, node, epoch, epoch + 2, device, node, 2))

        def absent():
            raise FileNotFoundError("fixture missing target artifact")
        output = scoped / "evidence"
        observed, _ = worker.execute(output, bind, locate, absent)
        assert observed["state"] == "STOP_AUDIT_FAILED" and observed["actual_provider_succeeded"] is True
        assert observed["actual_target_inputs_qualified"] == 0 and observed["local_reference_available_inputs"] == 22
        provider_raw = (output/"PROVIDER.json").read_bytes()
        result_raw = (output/"RESULT.json").read_bytes()
        intent = json.loads((output/"INTENT.json").read_bytes())
        row = [bind["step"], intent["label"], "FAILED", "1:0", "2", node, "cpu=2,mem=2G,node=1",
               day(epoch + 2), day(epoch + 4)]
        actual = terminal.verify(intent, bind["step"], "|".join(row), result_raw, provider_raw)
        assert actual["independent_provider_identity_bound"] and actual["actual_target_inputs_qualified"] == 0
        assert actual["frozen_dev24_windows"] == 24 and actual["original_missing_windows"] == 2
        assert terminal.verify(intent, bind["step"], "|".join(row), None, provider_raw)["independent_provider_identity_bound"]
        bad = list(row)
        bad[6] += ",gres/dcu=1"
        deny["accounting_dcu"] = reject("EXACT_INDEPENDENT_ZERO_GRES_2CPU_2G",
            lambda: terminal.verify(intent, bind["step"], "|".join(bad), result_raw, provider_raw))
        delayed = list(row)
        delayed[-1] = day(epoch + 107)
        delayed[4] = "105"
        deny["accounting_late"] = reject("EXACT_ORIGINAL_EPOCH_TERMINAL_BUDGET",
            lambda: terminal.verify(intent, bind["step"], "|".join(delayed), result_raw, provider_raw))
        deny["accounting_wrong_step"] = reject("EXACT_NINE_COLUMN_NUMERIC_STEP",
            lambda: terminal.verify(intent, parent + ".990", "|".join(row), result_raw, provider_raw))
        local_provider = json.loads(provider_raw)["provider"]
        provider_flags = {"version": local_provider["version"], "core_extension": local_provider["core_extension"],
                          "module": local_provider["module"], "target_ABI_claimed": False, "same_process_actual_import": True}
    return {"schema": "vggt.1318.cpu-readback.local-v1", "source_offer": worker.pin(value), "source_members": 21,
            "original_scorer": worker.pin(real_scorer), "local_provider": provider_flags,
            "frozen_dev24_windows": 24, "historically_verified_inputs": 22, "target_inputs_qualified": 0,
            "missing_new_target_fixture": observed["state"], "provider_receipt_survives_audit_failure": True,
            "independent_zero_gres_terminal_fixture": True, "captured_argv": worker.pin(worker.encode(built[0])),
            "actual_install_bootstrap_fixture_pass": True,
            "rejections": deny, "precedent": precedent(), "remote_calls": 0, "Slurm": 0, "GPU": 0,
            "model_loads": 0, "predictions": 0, "optimizer_updates": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("packet", "check", "install-command"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--request")
    a = p.parse_args()
    assert a.output.parent.resolve(strict=True).is_relative_to(LOCAL)
    value, boot = build()
    if a.mode == "packet":
        raw = value
    elif a.mode == "check":
        raw = worker.encode(replay(value, boot))
    else:
        assert host.REQUEST.fullmatch(a.request or "")
        code = Path(install_bootstrap.__file__).read_text(encoding="utf-8")
        argv = ["/usr/bin/timeout", "--kill-after=5s", "45s", "/usr/local/python3.12/bin/python3", "-I", "-S",
                "-B", "-u", "-c", code, a.request, worker.pin(value)["SHA256"], worker.encode(boot).decode().strip()]
        raw = (shlex.join(argv) + "\n").encode()
    host.save_file(a.output, raw)
    print(json.dumps(worker.pin(raw), sort_keys=True))


if __name__ == "__main__":
    main()
