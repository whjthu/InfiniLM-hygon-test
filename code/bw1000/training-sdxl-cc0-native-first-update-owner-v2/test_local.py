"""Exercise the new guard on published helper bytes and fixed scheduler fixtures."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import host


def reject(fn, code):
    try:
        fn()
    except ValueError as error:
        assert str(error) == code, str(error)
    else:
        raise AssertionError(code)


def main():
    repo = Path(__file__).resolve().parents[3]
    home = repo / "code/bw1000"
    common = home / "training-llava-clevr-original5000-val16-owner-v1/host.py"
    support = home / "training-llava-clevr-first2-owner-v1"
    pilot = home / "fleet-dcu-standby-control-node-pilot-v2/pilot.py"
    legacy = home / "fleet-dcu-standby-f13r4n03-v1/shared_standby.py"
    request = "9bbc653559a04c62b863baae9567218f"
    with TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        (root / "incoming").mkdir()
        manifest = host.HERE / "SOURCE_FILES.json"
        from hashlib import sha256
        args = SimpleNamespace(request=request, manifest_sha=sha256(manifest.read_bytes()).hexdigest(),
                               common=common, support=support, pilot=pilot, legacy=legacy)
        with patch.multiple(host, ROOT=root, RESULTS=root / "results"), patch.object(host, "assets"):
            ready = host.stage(args)
            target, work = host.paths(request)
            base, old, snap, accounting, _ = host.common(target)
            host.staged(base, target, request, ready["stage_SHA256"])
            assert snap.NODE == old.NODE == host.NODE and snap.STEP.fullmatch("1194900.378")
            assert not base.STEP.fullmatch("1103606.378")
            reject(lambda: host.label("c5eda9cfa6ad42ac933670d8f353064c"), "NEW_REQUEST_ID")
            pre = ["1194900.9001", host.label(request) + "-pre", "COMPLETED", "0:0", "1", host.NODE,
                   "cpu=2,mem=2G,node=1", "2026-11-28T07:00:00", "2026-11-28T07:00:01"]
            recorded = []
            def sacct(argv):
                assert argv[0] == "sacct"
                recorded.append((argv[argv.index("-j") + 1], argv[argv.index("-S") + 1]))
                return ["|".join(pre)] if recorded[-1][0] in ("1194900", pre[0]) else []

            with patch.object(old, "query", sacct):
                assert base.role_row(old, request, "pre", terminal=True) == pre
                assert base.wait_role(old, accounting, request, "pre") == pre
                override_since = "2026-11-28T06:59:00"
                assert old.account(job=pre[0], since=override_since) == [pre]
            assert len(recorded) == 4 and all(job == host.PARENT for job, _ in recorded[:-1])
            assert recorded[-1] == (pre[0], override_since)
            try:
                host.stage(args)
            except FileExistsError:
                pass
            else:
                raise AssertionError("duplicate stage")
            p = host.isolated(pilot, "exact_n03_test_pilot", host.PILOT)
            l = host.isolated(legacy, "exact_n03_test_legacy", host.LEGACY)
            holder = (l.OLD, l.OLD_LABEL, "RUNNING", "0:0", "1", host.NODE, l.ZERO_GRES)
            finished = (l.ORIGINAL, l.ORIGINAL_LABEL, "COMPLETED", "0:0", "1", host.NODE, l.ZERO_GRES)
            role = ("1194900.9000", host.label(request) + "-gpu")
            legacy_controller = f"StepId={l.OLD} Name={l.OLD_LABEL} State=TIMEOUT NodeList={host.NODE} TRES={l.ZERO_GRES}"
            gpu = f"StepId={role[0]} Name={role[1]} State=RUNNING NodeList={host.NODE} TRES=cpu=8,mem=62.50G,gres/dcu=1,node=1"
            def queries(extra=(), queue=(), stats=()):
                def answer(argv):
                    if argv[0:3] == ["scontrol", "show", "job"]:
                        return [f"JobId={host.PARENT} UserId=user({host.UID}) JobState=RUNNING "
                                f"NodeList={host.NODE} AllocTRES=cpu=5120,gres/dcu=320"]
                    if argv[0:3] == ["scontrol", "show", "step"]:
                        return [legacy_controller, *extra]
                    if argv[0:2] == ["squeue", "--steps"]:
                        return list(queue)
                    if argv[0] == "squeue":
                        return [f"{host.PARENT}|RUNNING|{host.NODE}"]
                    assert argv[0] == "sstat"
                    return list(stats)
                return answer

            with patch.object(p, "rows", lambda step: [list(finished if step == l.ORIGINAL else holder)]):
                with patch.object(old, "query", queries()):
                    host.scheduler_guard(target, old, providers=(p, l))
                with patch.object(old, "query", queries(extra=[gpu], queue=[f"{role[0]}|{role[1]}|{host.NODE}"])):
                    host.scheduler_guard(target, old, role, (p, l))
                    reject(lambda: host.scheduler_guard(target, old, providers=(p, l)), "OTHER_NUMBERED_CONTROLLER_STEP")
                with patch.object(old, "query", queries(extra=[gpu.replace("dcu=1", "dcu=8")])):
                    reject(lambda: host.scheduler_guard(target, old, role, (p, l)), "EXACT_NEW_ONE_DCU_STEP")
                with patch.object(old, "query", queries(queue=[f"{l.OLD}|{l.OLD_LABEL}|{host.NODE}"])):
                    reject(lambda: host.scheduler_guard(target, old, providers=(p, l)), "OTHER_NUMBERED_QUEUE_STEP")
                with patch.object(old, "query", queries(stats=["1194900.189|1|12K"])):
                    reject(lambda: host.scheduler_guard(target, old, providers=(p, l)), "PROTECTED_189_HAS_LIVE_STAT")
            gpu_argv = host.step_spec(target, work, request, "gpu", "b" * 64)
            cpu_argv = host.step_spec(target, work, request, "post", "b" * 64)
            assert "--gres=dcu:1" in gpu_argv and "--gres=none" in cpu_argv
            assert "--overlap" not in gpu_argv and "--nodelist=f13r4n03" in gpu_argv
            args = host.guest(target, work, root, request, "b" * 64)
            assert "/owner-code/device.py" in args and "/scope/pipeline.json" in args and "--rocm" in args
    print(json.dumps(dict(stage_and_manifest=True, exact_original_helpers=True, protected_189=True,
                          own_one_card_only=True, bound_accounting_parent_and_override=True, local_scheduler_fixtures=True,
                          pipeline_test_scope="stage-receipt-bind-not-target-bytes", target_calls=0)))


if __name__ == "__main__":
    main()
