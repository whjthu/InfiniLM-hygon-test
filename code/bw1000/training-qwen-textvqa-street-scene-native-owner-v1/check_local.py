"""Exercise the real private archive and owner boundaries without a cluster."""

import argparse
import ast
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import BadZipFile

import owner


ROOT = Path(__file__).resolve().parents[3]


def reject(call, wanted):
    try:
        call()
    except (ValueError, OSError, BadZipFile) as error:
        if not isinstance(error, FileExistsError):
            assert str(error) == wanted, (str(error), wanted)
        return wanted
    raise AssertionError(wanted)


def check(archive, receipt):
    request, publication, parent, node = "a" * 32, "b" * 40, "1", "f0r0n0"
    epoch = 1795586400
    claim = dict(request=request, publication=publication, parent=parent, node=node, CPU_GO=True,
                 GPU_GO=False, exclusive_node=True, replay_allowed=False, zip_SHA256=owner.ZIP_SHA,
                 dataset_SHA256=owner.DATA_SHA, cpus=8, memory_MiB=12288, allocated_DCU=0,
                 max_deadline_s=480, planned=41, owner_SHA256=owner.digest(Path(owner.__file__).read_bytes()))
    authority = dict(request=request, parent=parent, node=node, publication=publication, GPU=0,
                     exclusive_cpu_admit=True, coordinator_proof_SHA256="c" * 64, observed_epoch=epoch-3)
    nodes = [node, *(f"f0r1n{i}" for i in range(39))]
    selected = "1.18"
    denials = []
    captured = []
    stamp = lambda e: datetime.fromtimestamp(e, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    for path in (Path(owner.__file__), Path(__file__)):
        ast.parse(path.read_bytes(), filename=str(path))
    ast.parse(owner.WORKER)
    with TemporaryDirectory(dir=archive.parent.parent, prefix="textvqa-owner-fixture-") as scratch:
        root = Path(scratch).resolve()
        results = root / "results"
        results.mkdir()
        members = owner.manifest(receipt)
        assert len(members) == 22
        assert owner.stage(results, request, archive, receipt)["members"] == 22
        denials.append(reject(lambda: owner.stage(results, request, archive, receipt), "DUPLICATE_STAGE"))
        run = owner.namespace(results, request)
        data = run / "inputs"
        assert owner.installed(run, request) == members
        for name in owner.SOURCES:
            source = ROOT / owner.PILOT / Path(name).name
            raw = source.read_bytes()
            assert owner.digest(raw) == members[name]["SHA256"] == owner.SOURCES[name]
            ast.parse(raw, filename=name)
        sys.path.insert(0, str(data / "src"))
        try:
            native = runpy.run_path(str(data / "src/check_textvqa_street_scene_native.py"))
            rows = native["frozen_inputs"](data / "TRAIN_SWIFT.jsonl", data / "MEDIA_PINS.json", owner.MEDIA_SHA)
        finally:
            sys.path.pop(0)
        assert [sum(kind == name for kind, _ in rows) for name in ("VQA", "OCR")] == [26, 15]
        worker_guard = compile(owner.WORKER, "<actual worker gate>", "exec")
        worker_output = io.StringIO()
        worker_env = dict(SLURM_JOB_ID=parent, SLURM_STEP_ID="18", SLURM_JOB_GPUS="0,1",
                          SLURM_GPUS_ON_NODE="0", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
                          ROCR_VISIBLE_DEVICES="")
        checker = str(data / "src/check_textvqa_street_scene_native.py")
        with patch.dict(os.environ, worker_env, clear=True), patch.object(socket, "gethostname", return_value=node), \
             patch.object(sys, "path", sys.path.copy()), patch.object(sys, "argv", ["-c", parent, node, checker]), \
             patch.object(runpy, "run_path") as native_start, redirect_stdout(worker_output):
            exec(worker_guard, {})
            native_start.assert_called_once_with(checker, run_name="__main__")
        assert worker_output.getvalue() == "EXACT_STEP=1.18\n"
        for name in ("SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE"):
            with patch.dict(os.environ, dict(worker_env, **{name:"1"}), clear=True), \
                 patch.object(socket, "gethostname", return_value=node), \
                 patch.object(sys, "argv", ["-c", parent, node, checker]), patch.object(runpy, "run_path") as native_start:
                try:
                    exec(worker_guard, {})
                except SystemExit as error:
                    assert str(error) == "NUMERIC_ZERO_GRES_STEP_ENV_REQUIRED"
                else:
                    raise AssertionError("worker accepted a GPU-assigned step")
                native_start.assert_not_called()
            denials.append("NUMERIC_ZERO_GRES_STEP_ENV_REQUIRED_" + name)
        bad = root / "wrong"
        bad.write_bytes(archive.read_bytes()[:-1]+b"!")
        denials.append(reject(lambda: owner.stage(results, "d"*32, bad, receipt), "EXACT_INPUT_BYTES"))
        assert not owner.namespace(results, "d"*32).exists()
        member = data / "TRAIN_SWIFT.jsonl"
        raw = member.read_bytes()
        member.write_bytes(raw[:-1] + b"!")
        denials.append(reject(lambda: owner.installed(run, request), "EXACT_INPUT_BYTES"))
        member.write_bytes(raw)
        claim_file, admission = root / "claim.json", root / "authority.json"
        claim_raw, auth_raw = owner.encode(claim), owner.encode(authority)
        claim_file.write_bytes(claim_raw)
        admission.write_bytes(auth_raw)

        def scheduler(*argv):
            if argv[:3] == ("scontrol", "show", "job"):
                return [f"JobId={parent} JobState=RUNNING NumNodes=40 AllocTRES=gres/dcu=320,mem=480G "
                        f"UserId=fixture(100) NodeList=whole EndTime={stamp(epoch+800)}"]
            if argv[:3] == ("scontrol", "show", "hostnames"):
                return nodes if argv[3] == "whole" else [argv[3]]
            if argv[:3] == ("scontrol", "show", "node"):
                return [f"NodeName={node} State=MIXED CPUTot=64 RealMemory=200000"]
            if argv[:3] == ("scontrol", "show", "step"):
                return [f"StepId=1.77 NodeList={nodes[-1]} State=RUNNING"]
            if argv[:3] == ("squeue", "--steps", "-h"):
                return [f"1.77|{nodes[-1]}"]
            if argv[:3] == ("squeue", "-h", "-j"):
                return ["1|RUNNING|whole"]
            if argv[:4] == ("scontrol", "-d", "show", "job"):
                return ["Nodes=whole CPU_IDs=0-7 Mem=12G"]
            if argv[0] == "sacct":
                return []
            raise AssertionError(argv)

        def provider(root_arg, data_arg):
            assert root_arg == root and data_arg == data
            return root / "model", root / "framework", root / "sandbox", [(data, data)]

        simulated_report = dict(planned=41, processed=41, rows=[{}]*41, dataset_SHA256=owner.DATA_SHA,
                                media_inventory_SHA256=owner.MEDIA_SHA, template_SHA256=owner.TEMPLATE_SHA,
                                provider_versions=owner.VERSIONS, GPU=0, optimizer_updates=0,
                                model_materialized=False, collator_passed=True, failures=[])

        def execute(argv, **kwargs):
            assert (run / "intent.json").is_file()
            assert {"--gres=none", "--nodes=1", "--ntasks=1", "--nodelist="+node} <= set(argv)
            assert str(data / "MEDIA_PINS.json") in argv[-1] and "ROCR_VISIBLE_DEVICES=" in argv[-1]
            assert kwargs["timeout"] == 472
            captured.append(argv)
            kwargs["stdout"].write(f"EXACT_STEP={selected}\n".encode() + owner.encode(simulated_report))
            return subprocess.CompletedProcess(argv, 0)

        def start(scheduler_call=scheduler, now=epoch):
            return owner.start(root, request, publication, claim_file, owner.digest(claim_raw),
                               admission, owner.digest(auth_raw), query_job=scheduler_call,
                               get_provider=provider, execute=execute, clock=lambda:now, uid=100)

        def occupied(*argv):
            if argv[:3] == ("squeue", "--steps", "-h"):
                return [f"1.28|{node}"]
            return scheduler(*argv)

        denials.append(reject(lambda: start(now=epoch+100), "FRESH_EXCLUSIVE_COORDINATOR_PROOF"))
        denials.append(reject(lambda: start(occupied), "PLACEMENT_ALREADY_OCCUPIED_BY_NUMERIC_STEP"))
        assert not captured and not (run / "intent.json").exists()
        assert start()["launch_returncode"] == 0 and len(captured) == 1
        denials.append(reject(start, "OWNER_ATTEMPT_ALREADY_USED"))
        assert len(captured) == 1
        def terminal(gres="", duration=20):
            line = f"{selected}|{run.name}|COMPLETED|0:0|{duration}|{node}|cpu=8,mem=12288M,node=1{gres}|"
            return owner.terminal(results, request, lambda *args: [line+stamp(epoch+1)+"|"+stamp(epoch+duration)])
        assert terminal()["native_pass"]
        assert owner.terminal(results, request, lambda *args: [])["processed"] == 0
        denials.append(reject(lambda: terminal(",gres/dcu=1"), "EXACT_ZERO_GRES_NUMERIC_TERMINAL"))
        denials.append(reject(lambda: terminal(",gres/gpu=1"), "EXACT_ZERO_GRES_NUMERIC_TERMINAL"))
        denials.append(reject(lambda: terminal(duration=500), "EXACT_TERMINAL_BOUNDED_EPOCH"))
    return dict(state="ARCHIVE_AND_OWNER_FIXTURES_CHECKED", members=len(members), original_frozen_rows=41,
                VQA=26, OCR=15, captured_CPU_calls=len(captured), rejected=denials,
                actual_worker_guard_fixture=True,
                actual_native_cpu_rows=0, scheduler_calls=0, remote_calls=0, GPU=0, Git=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.zip.resolve(strict=True), args.receipt.resolve(strict=True)), sort_keys=True))
