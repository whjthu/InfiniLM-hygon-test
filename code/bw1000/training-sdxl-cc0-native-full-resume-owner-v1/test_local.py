"""Verify the exact frozen v3 owner adapter, evidence and CP20 argv without Slurm."""

from contextlib import nullcontext, redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import host


REQUEST = "9bbc653559a04c62b863baae9567218f"


def reject(fn, reason):
    try:
        fn()
    except ValueError as error:
        assert str(error) == reason, str(error)
    else:
        raise AssertionError(reason)


def main():
    repo = Path(__file__).resolve().parents[3]
    code = repo / "code/bw1000"
    original = code / "training-sdxl-cc0-native-first-update-owner-v3/host.py"
    proof_dir = repo / "local-only"
    with TemporaryDirectory() as directory:
        temp = Path(directory).resolve()
        (temp / "incoming").mkdir()
        (temp / "results").mkdir()
        image = temp / "containers/dtk26-sandbox-r1"
        image.mkdir(parents=True)
        frame = temp / "framework"
        (frame / "genmlu").mkdir(parents=True)
        with tarfile.open(code / "training-framework-bd24ec3-v1/FRAMEWORK.tar.gz") as archive:
            name = "framework/generation/src/genmlu/sdxl_training_resume_checkpoint.py"
            (frame / "genmlu/sdxl_training_resume_checkpoint.py").write_bytes(archive.extractfile(name).read())
        with patch.multiple(host, ROOT=temp, RESULTS=temp / "results", IMAGE=image):
            base = host.bind(original)
            base.FRAME = frame
            base.assets = lambda *_: None
            base.live_runtime = lambda *_: None
            image_id = "sha256:" + "a" * 64
            step = "1194900.9911"
            value = dict(
                rootfs=dict(host_path=image.as_posix(), image_id=image_id, method="fixture-content-root",
                            source_receipt_SHA256="c" * 64, stat=host.stamp()),
                provider=dict(image_id=image_id, receipt_SHA256="d" * 64, step=step, parent=host.PARENT,
                              node=host.NODE, accounting=[step, "fixture-only", "COMPLETED", "0:0", "1", host.NODE,
                              "cpu=2,mem=8G,node=1", "2026-11-28T01:00:00", "2026-11-28T01:00:01"],
                              zero_gres=True, hip_full_sources_SHA256=host.DERIVED, native_cli_worker_count=0,
                              native_full_checkpoint=True))
            assert host.proof(value, host.stamp()) == image_id
            reject(lambda: host.proof(value, [0, 0, 0, 0]), "EXACT_INDEPENDENT_DTK_ROOTFS_IDENTITY")
            for field, bad in (("image_id", "sha256:" + "b" * 64), ("native_cli_worker_count", 2),
                               ("hip_full_sources_SHA256", {})):
                mutated = dict(value, provider=dict(value["provider"], **{field: bad}))
                reject(lambda: host.proof(mutated, host.stamp()), "EXACT_INDEPENDENT_ZERO_GRES_PROVIDER")
            numbers = value["provider"]["accounting"]
            for index, replacement, error in (
                    (2, "RUNNING", "EXACT_INDEPENDENT_NUMBERED_RUNTIME_ACCOUNTING"),
                    (6, "cpu=2,mem=8G,node=1,gres/dcu=1", "EXACT_RUNTIME_PROBE_ZERO_GRES_TRES"),
                    (8, "Unknown", "EXACT_INDEPENDENT_NUMBERED_RUNTIME_ACCOUNTING")):
                changed = numbers.copy()
                changed[index] = replacement
                reject(lambda: host.proof(dict(value, provider=dict(value["provider"], accounting=changed)),
                                           host.stamp()), error)
            cert = temp / "runtime-proof.json"
            cert.write_text(json.dumps(value), encoding="utf-8")
            args = SimpleNamespace(
                request=REQUEST, base_host=original, common=code / "training-llava-clevr-original5000-val16-owner-v1/host.py",
                support=code / "training-llava-clevr-first2-owner-v1",
                pilot=code / "fleet-dcu-standby-control-node-pilot-v2/pilot.py",
                legacy=code / "fleet-dcu-standby-f13r4n03-v1/shared_standby.py",
                full_source=code / "training-sdxl-cc0-native-full-resume-hip-v1", runtime_proof=cert,
                runtime_proof_sha=sha256(cert.read_bytes()).hexdigest(),
                manifest_sha=sha256((host.HERE / "SOURCE_FILES.json").read_bytes()).hexdigest(),
                runtime_import_proof=proof_dir / "sdxl-actual-native-complete-runtime-import-probe-e7081657a38946ff939bedfe2b5326b7.json",
                tracker_proof=proof_dir / "sdxl-tracker-exact-native-zero-gres-67aab4de054c43bd9339800cfe39d9cc.json",
                runtime_terminal_proof=proof_dir / "domain-12-native-sdxl-gemma-llava-exact-accounting-c29c531b64634a0b92b708f54e0f1978.json",
            )
            receipt = host.stage(base, args)
            target, work = host.paths(REQUEST)
            common, old, *_ = base.common(target)
            base.staged(common, target, REQUEST, receipt["stage_SHA256"])
            argv = base.guest(target, work, temp, REQUEST, "e" * 64)
            assert argv[-6:] == ["--firstupdate-source", "/first-update-source", "--image-id", image_id, "--resume-step", "20"]
            assert "/first-update-source/pipeline.json" in argv and "--pipeline-manifest-sha" not in argv
            assert f"{(target / 'full').as_posix()}:/scope:ro" in argv and "--rocm" in argv
            gpu = base.step_spec(target, work, REQUEST, "gpu", "f" * 64)
            assert "--gres=dcu:1" in gpu and "--time=30" in gpu and "--mem=64000M" in gpu and "--overlap" not in gpu
            auth = dict(request=REQUEST, parent=host.PARENT, node=host.NODE, boot="00000000-0000-0000-0000-000000000000",
                        epoch=time.time(), deadline=time.time()+1000, port=48373,
                        pci=[f"0000:{x:02x}:00.0" for x in range(8)], root_GO=True, ordinary_excluded=True,
                        fresh_physical_SHA256="e" * 64, stage_SHA256=receipt["stage_SHA256"])
            grant = temp / "auth.json"
            grant.write_bytes(common.wire(auth))
            run = SimpleNamespace(request=REQUEST, auth=grant, auth_sha=sha256(grant.read_bytes()).hexdigest())
            output = io.StringIO()
            with patch.object(host.os, "geteuid", lambda: base.UID, create=True), \
                 patch.object(old, "held_lock", nullcontext), \
                 patch.object(base, "scheduler_guard", side_effect=ValueError("fixture busy")), redirect_stdout(output):
                assert base.start(run, target, work, common, old, None) == 2
            failed = json.loads(output.getvalue())
            assert failed["planned_updates"] == 30 and failed["gpu_launch_attempted"] is False and not work.exists()
    print("exact v3 stage/owner qualification, independent-runtime negative branches, CP20-only argv and failure denominator PASS")


if __name__ == "__main__":
    main()
