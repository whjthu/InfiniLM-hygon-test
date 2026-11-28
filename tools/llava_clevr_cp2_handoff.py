"""Hand the existing owned eight-card standby to the qualified T04 CP2 continuation."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import stat
import time
from types import ModuleType


REQUEST = "0e4a27af103c45e7b3bddcaeeb7d7c67"
BASE_PIN = "0cac314f5f38d159d37c4e51b46628afccf975c7c5c75c6e0b18dd62bc8fd02b"
HOLDER = "1103606.4988"
STANDBY_REQUEST = "b28d634870bb4446bbd88742f691d28c"
CPU_STEP = "1103606.4993"
PLAN = "13577b6c5d372f3cf1ec6b0f8b80d2bb6b1a173abef60ae30ddb4c7411769fb5"
ASSEMBLY_STEP = "1103606.4996"
ASSEMBLY_PIN = "8709e75ec1572b0f2abcf11fb61b93e94c5baadd34cf0c47ee55a38b94903c27"
SCIENCE = "0b16fb53ca923e027462c13ac8a5bb6e548e60d4c640fe1444fd6a793c8cc561"
OWNER_PIN = "9f215f48b662638fa157950674600ec0f4ec5b497ef4440134ef5ac252f55288"


def provider():
    path = Path(__file__).resolve().with_name("llava_clevr_orderly_handoff.py")
    source = path.read_bytes()
    if sha256(source).hexdigest() != BASE_PIN:
        raise ValueError("Exact original handoff provider required")
    base = ModuleType("previous_llava_handoff")
    exec(compile(source, str(path), "exec"), vars(base))
    base.HOLDER, base.STANDBY_REQUEST = HOLDER, STANDBY_REQUEST
    base.STANDBY_LABEL = f"training80-dcu-standby-{STANDBY_REQUEST[:12]}-{base.NODE}"
    base.STANDBY = base.RESULTS / f"training80-dcu-standby-{STANDBY_REQUEST}"
    return base


def wait_for_return(base, work, terminal, deadline):
    result = None
    previous = None
    with (work / "release-observations.jsonl").open("xb") as stream:
        while time.monotonic() < deadline:
            observation = {}
            try:
                row = base.exact_row(HOLDER)
                observation["accounting"] = row[2:4]
            except base.SlurmQueryUnavailable as error:
                observation["accounting_unavailable"] = str(error)
            try:
                controller, queue = base.node_occupancy()
                observation.update(controller=sorted(controller), queue=sorted(queue))
                base.require(set(controller) <= {HOLDER} and queue <= {HOLDER}, "New work during CP2 handoff")
            except base.SlurmQueryUnavailable as error:
                observation["occupancy_unavailable"] = str(error)
            if terminal.is_file():
                returned = json.loads(base.read(terminal))
                base.require((returned["node"], returned["step"], returned["state"]) ==
                             (base.NODE, HOLDER, "WORKER_RETURNED"), "Unexpected standby worker terminal")
                observation["worker"] = "WORKER_RETURNED"
            else:
                observation["worker"] = None
            if previous != observation:
                stream.write(base.encode(dict(epoch=time.time(), **observation)))
                stream.flush()
                os.fsync(stream.fileno())
                previous = observation
            if (observation.get("accounting") == ["COMPLETED", "0:0"]
                    and observation.get("controller") == observation.get("queue") == []
                    and observation["worker"] == "WORKER_RETURNED"):
                result = row
                break
            time.sleep(min(1.5, max(0, deadline - time.monotonic())))
        stream.write(base.encode(dict(epoch=time.time(), simultaneous_release_proved=result is not None)))
        stream.flush()
        os.fsync(stream.fileno())
    base.require(result is not None, "Exact CP2 standby release remains unproved")
    return result


def verify_inputs(base, owner_dir):
    manifest = json.loads(base.read(owner_dir / "SOURCE_FILES.json", expected=OWNER_PIN))["files"]
    base.require(set(manifest) == {item.name for item in owner_dir.iterdir() if item.name != "SOURCE_FILES.json"},
                 "Exact continuation owner members required")
    for name, pin in manifest.items():
        base.require(Path(name).name == name and len(base.read(owner_dir / name, cap=pin["bytes"],
                     expected=pin["SHA256"])) == pin["bytes"], "Exact continuation source identity")
    assembly = base.RESULTS / f"llava-cp2-owner-assembly-{REQUEST}/assembly.json"
    proof = json.loads(base.read(assembly, expected=ASSEMBLY_PIN))
    base.require((proof["request"], proof["step"], proof["plan_SHA256"], proof["science_SHA256"]) ==
                 (REQUEST, ASSEMBLY_STEP, PLAN, SCIENCE) and proof["passed"] is True and
                 proof["actual_GPU"] == 0 and not proof["model_loaded"], "Exact same-request assembly required")
    row = base.exact_row(ASSEMBLY_STEP)
    base.require(row[1:4] == [f"training-llava-clevr-resume25-{REQUEST[:12]}-assembly", "COMPLETED", "0:0"]
                 and row[5:7] == [base.NODE, "cpu=2,mem=2G,node=1"], "Actual assembly accounting differs")
    base.require(base.exact_row(CPU_STEP)[2:4] == ["COMPLETED", "0:0"], "Original order proof must be terminal")
    return assembly


def main():
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    base = provider()
    work = args.work
    base.require(work.resolve(strict=True) == work and work.parent == base.RESULTS
                 and work.name == f"llava-cp2-resume-control-{REQUEST}" and work.stat().st_uid == os.getuid()
                 and socket.gethostname().split(".")[0] == base.CONTROL_NODE
                 and os.environ.get("SLURM_JOB_ID") == base.PARENT
                 and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "Owned numbered CPU control required")
    step = os.environ.get("SLURM_STEP_ID", "")
    base.require(step.isdigit() and re.search(rf"/job_{base.PARENT}(?:\.scope)?/step_{step}(?:\.scope)?(?:/|$)",
                 Path("/proc/self/cgroup").read_text(), re.MULTILINE) is not None, "Control cgroup differs")
    control_step = f"{base.PARENT}.{step}"
    label = f"training-llava-clevr-resume25-{REQUEST[:12]}"
    rows = [row for row in map(base.fields, base.query("scontrol", "show", "step", base.PARENT, "--oneliner"))
            if row.get("StepId") == control_step]
    base.require(len(rows) == 1 and (rows[0]["Name"], rows[0]["State"], rows[0]["NodeList"], rows[0]["TRES"])
                 == (label + "-control", "RUNNING", base.CONTROL_NODE, "cpu=2,mem=2G,node=1"),
                 "Actual control placement differs")
    for key in tuple(os.environ):
        if key.startswith(("SLURM_", "SRUN_", "PMI_", "PMIX_")):
            os.environ.pop(key)
    os.environ.update(LC_ALL="C", TZ="UTC", SLURM_TIME_FORMAT="standard")
    run = base.RESULTS / label
    owner = base.ROOT / "incoming/training-llava-clevr-cp2-owner-v1-9f215f48b662"
    assembly = verify_inputs(base, owner)
    lock = os.open(base.RESULTS / ".training80-dcu-standby.lock", os.O_RDWR | os.O_NOFOLLOW)
    try:
        base.require(stat.S_ISREG(os.fstat(lock).st_mode), "Exact fleet lock must be regular")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.require(not run.exists(), "Original successor must never be replayed")
        base.save(work / "release-intent.json", dict(request=REQUEST, control_step=control_step, frozen_holder=HOLDER))
        base.wait_for_holder_return = lambda terminal, deadline: wait_for_return(base, work, terminal, deadline)
        row, pci, plan_sha, worker_sha, handoff = base.release_holder(work, REQUEST)
        independent = base.independent_release(work, owner, REQUEST, pci)
        release_path = work / "release.json"
        release_pin = base.save(release_path, dict(
            request=REQUEST, mode="ORDERLY_ROOT_HANDOFF" if handoff else "NATURAL", parent=base.PARENT, node=base.NODE,
            holder_step=HOLDER, holder_label=base.STANDBY_LABEL, accounting=row, worker_SHA256=worker_sha,
            plan_SHA256=plan_sha, independent_post_SHA256=independent, worker_returned=True,
            independent_physical_clean=True, exact_root_handoff_confirmed=handoff))
        authority = work / "authority.json"
        epoch = time.time()
        auth_pin = base.save(authority, dict(
            request=REQUEST, parent=base.PARENT, node=base.NODE, label=label, port=base.PORT,
            epoch=epoch, deadline=epoch + 1500, GPU_GO=True, node_exclusive_to_request=True, release_SHA256=release_pin,
            owner_manifest_SHA256=OWNER_PIN, science_manifest_SHA256=SCIENCE, TENSORS_SHA256=base.TENSORS,
            vision_root=str(base.VISION), CPU_step=CPU_STEP, plan_SHA256=PLAN, assembly_step=ASSEMBLY_STEP,
            assembly_path=str(assembly), assembly_SHA256=ASSEMBLY_PIN))
        os.set_inheritable(lock, True)
        os.execv(base.PYTHON, [base.PYTHON, "-I", "-S", "-B", str(owner / "owner.py"), "launch",
                 "--run", str(run), "--auth", str(authority), "--auth-sha", auth_pin, "--release", str(release_path),
                 "--inherited-lock-fd", str(lock)])
    finally:
        os.close(lock)


if __name__ == "__main__":
    main()
