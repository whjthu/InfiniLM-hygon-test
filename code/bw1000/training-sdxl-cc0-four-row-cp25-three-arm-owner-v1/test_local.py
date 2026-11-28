"""Offline integration checks for frozen owner derivation and the new zero-GRES gate."""

from contextlib import nullcontext, redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import host

REQUEST = "abf3ea43c0a04bbab7457487de569712"
STAGE = "4" * 64
STEP = "1194900.9900"


def reject(func, reason):
    try:
        func()
    except ValueError as error:
        assert str(error) == reason, str(error)
    else:
        raise AssertionError(reason)


def write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
    return sha256(raw).hexdigest()


def read(path, pin=None, limit=1 << 20):
    raw = path.read_bytes()
    assert len(raw) <= limit and (pin is None or sha256(raw).hexdigest() == pin)
    return raw


def wire(value):
    return (json.dumps(value, separators=(",", ":")) + "\n").encode()


def fixture(base, old, root):
    target = root / "stage"
    target.mkdir()
    intent = dict(request=REQUEST, stage_SHA256=STAGE, role="verify", GPU=False)
    actual = dict(plan_SHA256=host.PLAN, planned_slots=6, planned_pairs=6, actual_generations=0)
    common = SimpleNamespace(read=read, put=write, wire=wire)
    args = SimpleNamespace(request=REQUEST, stage_sha=STAGE)
    live = f"StepId={STEP} Name={host.verify_name(REQUEST)} State=RUNNING NodeList={host.NODE} TRES=cpu=2,mem=2G,node=1"
    terminal = [STEP, host.verify_name(REQUEST), "COMPLETED", "0:0", "2", host.NODE, "cpu=2,mem=2G,node=1",
                "2026-11-28T10:10:10", "2026-11-28T10:10:12"]
    assert host.verify_name(REQUEST).endswith("-verify")
    with patch.object(old, "query", return_value=["StepId=1194900.9901", live]):
        assert host.exact_verify_live(base, old, host.PARENT, "9900", REQUEST) == STEP
    with patch.object(old, "query", return_value=[live + ",gres/dcu=1"]):
        reject(lambda: host.exact_verify_live(base, old, host.PARENT, "9900", REQUEST), "EXACT_LIVE_VERIFY_NO_GRES_TRES")
    with patch.object(old, "query", return_value=[live, live]):
        reject(lambda: host.exact_verify_live(base, old, host.PARENT, "9900", REQUEST), "EXACT_ONE_CURRENT_VERIFY_CONTROLLER_RECORD")

    launches = []
    def fake_srun(argv, **_):
        launches.append(argv)
        assert argv[0] == "srun" and "--gres=none" in argv and "--cpus-per-task=2" in argv
        assert "--overlap" not in argv and "compute-verify" in argv
        assert json.loads(read(target / "verify-intent.json")) == intent
    with (patch.object(old, "account", return_value=[]), patch.object(old, "held_lock", nullcontext),
          patch.object(base, "scheduler_guard"), patch.object(host.os, "geteuid", lambda: base.UID, create=True),
          patch.object(host.subprocess, "run", side_effect=fake_srun), redirect_stdout(StringIO())):
        assert host.verify_plan(base, common, old, target, args) == 2
        try:
            host.verify_plan(base, common, old, target, args)
        except FileExistsError:
            pass
        else:
            raise AssertionError("duplicate source verification")
    assert len(launches) == 1

    with (patch.object(host.os, "geteuid", lambda: base.UID, create=True),
          patch.dict(host.os.environ, {"SLURM_JOB_ID": host.PARENT, "SLURM_STEP_ID": "9900", "SLURM_STEP_GPUS": ""}),
          patch.object(host.socket, "gethostname", return_value=host.NODE),
          patch.object(Path, "read_text", return_value="0::/slurm/job_1194900/step_9900"),
          patch.object(host, "exact_verify_live", return_value=STEP),
          patch.object(host.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr=b"", stdout=wire(actual))) as guest):
        assert host.compute_verify(base, common, old, target, args) == 0
    argv = guest.call_args.args[0]
    assert "--rocm" not in argv and all(value.endswith(":ro") for value in argv if value.endswith((":ro", ":rw")))
    assert "HIP_VISIBLE_DEVICES=-1" in argv[argv.index("-c") + 1]
    assert "/owner-code/device.py" in argv and "verify-plan" in argv and host.CP25_MANIFEST in argv
    with patch.object(old, "account", return_value=[terminal]):
        assert host.verify_terminal(common, old, target, REQUEST, STAGE)["passed"]
        reject(lambda: host.verify_terminal(common, old, target, REQUEST, "5" * 64), "EXACT_STAGE_BOUND_VERIFY_RECEIPT")
    bad = terminal.copy()
    bad[6] += ",gres/dcu=1"
    with patch.object(old, "account", return_value=[bad]):
        reject(lambda: host.verify_terminal(common, old, target, REQUEST, STAGE), "INDEPENDENT_VERIFY_ZERO_GRES_TERMINAL_ATTESTATION")
    with patch.object(old, "account", return_value=[]):
        assert not host.verify_terminal(common, old, target, REQUEST, STAGE)["passed"]


def main():
    repo = Path(__file__).resolve().parents[3]
    code = repo / "code/bw1000"
    worker = code / "training-sdxl-cc0-four-row-cp25-three-arm-v1"
    manifest = json.loads(read(worker / "SOURCE_FILES.json", host.RUNNER))
    for name in host.FILES.values():
        assert len(read(worker / name, manifest[name]["SHA256"])) == manifest[name]["bytes"]
    for filename, pin in (("specialized-sdxl-2675-native-full25-cp20-exact-recovery-and-checkpoints.json", host.R33_PROOF),
                          ("specialized-sdxl-2769-four-native-rows-25-plus-5-exact-CP20-and-clean.json", host.TRAINING_PROOF)):
        read(repo / "results/bw1000/2026-11-28" / filename, pin)
    v3 = code / "training-sdxl-cc0-native-first-update-owner-v3/host.py"
    r33 = code / "training-sdxl-cc0-native-full-resume-owner-v1/host.py"
    support = code / "training-llava-clevr-first2-owner-v1/owner.py"
    support_sha = sha256(support.read_bytes()).hexdigest()
    base = host.module(v3, host.BASE)
    old = host.module(support, support_sha)
    original_module = host.module
    with patch.object(host, "module", side_effect=lambda path, pin: original_module(r33 if pin == host.R33_HOST else v3, pin)):
        bound = host.bind()
    assert bound.observe is host.observe
    with TemporaryDirectory() as path:
        temp = Path(path).resolve()
        fixture(base, old, temp)
        for role, budget in (("pre", "--gres=none"), ("gpu", "--gres=dcu:1"), ("post", "--gres=none")):
            argv = bound.step_spec(temp, temp, REQUEST, role, "2" * 64)
            assert budget in argv and "--jobid=1194900" in argv and "--nodelist=f13r4n03" in argv
    reject(lambda: host.label("94f0acaeec614c5fa9715033ae1a755d"), "EXACT_NEW_REQUEST")
    print("frozen science/predecessor, shared-lock single verify, exact zero-GRES live/terminal, wrong stage/GRES and no replay PASS")


if __name__ == "__main__":
    main()
