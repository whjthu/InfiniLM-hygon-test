"""Exercise the exact target-stage and scheduler guard code without running Slurm."""

from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import host


REQUEST = "debc02e9491145fc865abef8b4059134"


def rejects(fn, code):
    try:
        fn()
    except ValueError as error:
        assert str(error) == code, str(error)
    else:
        raise AssertionError(code)


def main():
    repo = Path(__file__).resolve().parents[3]
    native = repo / "code/bw1000/training-llava-clevr-original5000-val16-native-v1"
    support = repo / "code/bw1000/training-llava-clevr-first2-owner-v1"
    reload = repo / "tools/llava_clevr_cp25_reload.py"
    with TemporaryDirectory() as directory:
        temp = Path(directory).resolve()
        (temp / "incoming/media/images").mkdir(parents=True)
        (temp / "results").mkdir()
        media = temp / "incoming/media"
        pins = []
        for i in range(16):
            name = f"{i:02d}.png"
            raw = f"test fixture media {i}".encode()
            (media / "images" / name).write_bytes(raw)
            pins.append(dict(name=name, slot=i, bytes=len(raw), SHA256=sha256(raw).hexdigest()))
        requests_sha = host.put(media / "requests.jsonl", b"fixture does not invoke the image decoder\n")
        images_sha = host.put(media / "IMAGES.json", host.wire(pins))
        (media / "input-readback.json").write_bytes(b"fixture extra readback receipt")
        with patch.multiple(host, ROOT=temp, RESULTS=temp / "results", MEDIA=media,
                            REQUESTS=requests_sha, IMAGES=images_sha):
            args = SimpleNamespace(request=REQUEST, native_source=native, support=support, reload_tool=reload,
                                   host_sha=sha256(Path(host.__file__).read_bytes()).hexdigest())
            output = io.StringIO()
            with redirect_stdout(output):
                host.stage(args)
            stage_receipt = json.loads(output.getvalue())
            target, _, stage_sha = host.staged(REQUEST, stage_receipt["stage_SHA256"])
            assert stage_sha == stage_receipt["stage_SHA256"]
            assert {item.name for item in (target / "inputs").iterdir()} == {"requests.jsonl", "IMAGES.json", "images"}
            assert os.path.samefile(media / "images/00.png", target / "inputs/images/00.png")
            assert not (target / "native/test_local.py").exists()
            old, _, _, _ = host.binder(target)
            try:
                host.stage(args)
            except FileExistsError:
                pass
            else:
                raise AssertionError("duplicate stage accepted")
            (media / "images/15.png").write_bytes(b"wrong")
            rejects(lambda: host.projection(media, temp / "never", requests_sha, images_sha), "EXACT_MEMBER_SHA")
            assert not (temp / "never").exists()

            def query(argv):
                if argv[0:3] == ["scontrol", "show", "job"]:
                    return [f"JobId={host.PARENT} UserId=user({host.UID}) JobState=RUNNING NodeList={host.NODE}"]
                if argv[0:3] == ["scontrol", "show", "step"]:
                    return [f"StepId={host.PARENT}.batch NodeList={host.NODE}"]
                if argv[1] == "--steps":
                    return [f"{host.PARENT}.extern|{host.NODE}"]
                return [f"{host.PARENT}|RUNNING|{host.NODE}"]

            with patch.object(old, "query", query):
                host.no_occupancy(old)
            def busy(argv):
                if argv[1] == "--steps":
                    return [f"{host.PARENT}.7777|{host.NODE}"]
                return query(argv)
            with patch.object(old, "query", busy):
                rejects(lambda: host.no_occupancy(old), "NUMBERED_NODE_OCCUPANCY_OR_UNKNOWN_STEP")
            assert old.tres("cpu=8,gres/dcu=1,mem=62.50G,node=1", 1, 8, 64000)
            assert not old.tres("cpu=8,gres/dcu=2,mem=62.50G,node=1", 1, 8, 64000)
            assert not old.tres("cpu=2,gres/dcu=1,mem=2G,node=1", 0, 2, 2048)
            prior = [host.PARENT + ".555", "training80-exact-prior", "COMPLETED", "0:0", "3", host.NODE,
                     "cpu=8,gres/dcu=8,mem=16G,node=1", "2026-11-28T01:00:00", "2026-11-28T01:00:03"]
            auth = dict(request=REQUEST, prior_holder=prior)
            with patch.object(old, "account", lambda _: [prior]):
                host.prior_holder(auth, old)
            with patch.object(old, "account", lambda _: [prior[:2] + ["RUNNING"] + prior[3:]]):
                rejects(lambda: host.prior_holder(auth, old), "EXACT_PRIOR_IDLE_RELEASE_ACCOUNTING")
            release = dict(request=REQUEST, parent=host.PARENT, node=host.NODE, holder=prior,
                           independent_physical_clean=True, worker_returned=True, source_SHA256="b" * 64)
            host.release_document(auth, release)
            rejects(lambda: host.release_document(auth, dict(release, worker_returned=False)),
                    "EXACT_INDEPENDENT_ROOT_RELEASE_PROOF")
            pre = host.step_spec(target, temp, REQUEST, "pre", "a" * 64)
            gpu = host.step_spec(target, temp, REQUEST, "gpu", "a" * 64)
            assert "--gres=none" in pre and "--mem=2048M" in pre
            assert "--gres=dcu:1" in gpu and "--cpus-per-task=8" in gpu and "--time=30" in gpu
            command = host.guest(target, temp, temp, auth, "a" * 64)
            assert f"{(target / 'inputs').as_posix()}:/llava-blind:ro" in command
            assert "/scope/native.py" in command and "--rocm" in command
    print("host checks PASS: exact source stage, 18 hardlinks, duplicate and altered input fail, dual scheduler and TRES")


if __name__ == "__main__":
    main()
