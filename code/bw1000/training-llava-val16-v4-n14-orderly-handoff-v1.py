"""Release one numbered BF16 standby before starting the frozen Val16 owner."""

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
OWNER = "7344da28bdbd49c59060db6595a3d692"
STAGE_SHA256 = "ae38380dfc662bc0663c49f0202a395b1d699a3139fa4b5e284fa82894b9f812"
HOST_SHA256 = "59071b1f898087a73e084dcd3e9ea477c8391eb709a65d11e913edebfb3ca79b"
HANDOFF_SHA256 = "0cac314f5f38d159d37c4e51b46628afccf975c7c5c75c6e0b18dd62bc8fd02b"
STANDBY_REQUEST = "879c5fe1eb35448f926b5596849ce16e"
HOLDER = "1103606.5220"
PLAN_SHA256 = "9ff556190123f7cd072a835f4fc78802628eb305de382177cc24c8b979d8eb04"
WORKER_SHA256 = "74333b01242f8555b3c71a79b5acea985aad4adce1df36521ea091c5b46f5b77"
PYTHON = "/usr/local/python3.12/bin/python3"


def load(path, name, pin):
    from hashlib import sha256

    if path.resolve(strict=True) != path or not path.is_file() or not 0 < path.stat().st_size <= 128 << 10:
        raise ValueError("Source path differs")
    if sha256(path.read_bytes()).hexdigest() != pin:
        raise ValueError("Source pin differs")
    spec = importlib.util.spec_from_file_location(name, path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    return source


def run(publication, handoff_source):
    if not re.fullmatch(r"[0-9a-f]{40}", publication):
        raise ValueError("A published claim must bind this handoff")
    guard = load(handoff_source, "qualified_standby_release", HANDOFF_SHA256)
    guard.HOLDER = HOLDER
    guard.STANDBY_REQUEST = STANDBY_REQUEST
    guard.STANDBY_LABEL = f"training80-dcu-standby-{STANDBY_REQUEST[:12]}-{guard.NODE}"
    guard.STANDBY = ROOT / "results" / f"training80-dcu-standby-{STANDBY_REQUEST}"
    stage = ROOT / "incoming" / f"llava-clevr-5000-val16-v4-{OWNER[:12]}-stage"
    owner = load(stage / "host.py", "frozen_llava_val16_v4", HOST_SHA256)
    guard.require(os.geteuid() == owner.UID, "Exact project scope UID required")
    target, _, _ = owner.staged(OWNER, STAGE_SHA256)
    old, _, _, _ = owner.binder(target)
    evidence = ROOT / "results" / ("llava-v4-n14-root-release-" + OWNER)
    phase = "before_shared_lock"
    invoked = False
    try:
        with old.held_lock():
            phase = "pre_signal"
            guard.require(not owner.run_dir(OWNER).exists()
                          and all(owner.role_row(old, OWNER, role) is None for role in ("pre", "gpu", "post")),
                          "The frozen owner labels must never have run")
            plan, worker, _ = guard.standby_receipts()
            guard.require((plan, worker) == (PLAN_SHA256, WORKER_SHA256),
                          "Different incumbent worker")
            evidence.mkdir(mode=0o700)
            guard.save(evidence / "intent.json", dict(owner=OWNER, holder=HOLDER, publication=publication,
                                                       parent=guard.PARENT, node=guard.NODE, replay=False))
            phase = "orderly_standby_release"
            holder, pci, plan, worker, signalled = guard.release_holder(evidence, OWNER)
            phase = "independent_physical_release"
            physical_sha = guard.independent_release(evidence, stage / "support", OWNER, pci)
            physical = json.loads(guard.read(evidence / "standby-independent-post.json", expected=physical_sha))
            release_record = dict(request=OWNER, parent=guard.PARENT, node=guard.NODE, holder=holder,
                                  independent_physical_clean=True, worker_returned=True, source_SHA256=physical_sha)
            release = evidence / "prior-release.json"
            release_sha = guard.save(release, release_record)
            epoch = time.time()
            grant = dict(request=OWNER, parent=guard.PARENT, node=guard.NODE, stage_SHA256=STAGE_SHA256,
                         prior_holder=holder, prior_release_SHA256=release_sha, epoch=epoch, deadline=epoch + 2100,
                         root_GO=True, ordinary_excluded=True, idle_release_clean=True, port=guard.PORT,
                         boot=physical["physical"]["boot_id"], pci=sorted(pci))
            owner.scope(grant, OWNER)
            grant_path = evidence / "root-grant.json"
            grant_sha = guard.save(grant_path, grant)
        phase = "single_owner_dispatch"
        with (evidence / "controller.private.log").open("xb") as stream:
            child = subprocess.Popen([PYTHON, "-I", "-S", "-B", str(stage / "host.py"), "start", "--request", OWNER,
                                      "--auth", str(grant_path), "--auth-sha", grant_sha, "--release", str(release)],
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=os.environ.copy())
            invoked = True
        out = dict(rc=0, owner=OWNER, incumbent=HOLDER, incumbent_term_signalled=signalled, known_terminal=holder,
                   independent_release_step=physical["accounting"][0], independent_release_SHA256=physical_sha,
                   controller_pid=child.pid, deadline=grant["deadline"], gpu_step_confirmed=False)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        out = dict(rc=2, owner=OWNER, incumbent=HOLDER, phase=phase, exception=type(error).__name__,
                   failure_SHA256=guard.digest(str(error).encode()), incumbent_signal_intent=(evidence / "standby-signal-intent.json").is_file(),
                   controller_invoked=invoked, automatic_replay=False)
    if (evidence / "intent.json").is_file():
        guard.save(evidence / "dispatch-outcome.json", out)
    return out


if __name__ == "__main__":
    stamp, dependency = sys.argv[1:]
    result = run(stamp, Path(dependency))
    print(json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(result["rc"])
