"""Login-side once-only controller for the admitted HunyuanVideo ABO clip."""

from base64 import b64decode
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = PROJECT / "results"
INSPECT = RESULTS / "training-hunyuanvideo-abo-vae-inspect-v1/a6b6685a35a52543d3f6eb19ff141d31"
PREVIOUS = RESULTS / "training-hunyuanvideo-t09-sft-repeat3-n14-untraced-v1/src"
RUNTIME = RESULTS / "training-t06-sdxl-runtime-v1"
NATIVE = RESULTS / "training-hunyuanvideo-original-source-v1/source"
VAE = RESULTS / "training-hunyuanvideo-public-vae-v1/vae"
IMAGE = PROJECT / "containers/dtk26-sandbox-r1"
PYTHON = "/usr/local/python3.12/bin/python3"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
PARENT, NODE = "1103606", "f13r3n14"
LABEL = "training-hunyuanvideo-abo-vae-native-encode-v1"
PREDECESSOR = ["1103606.4128", "training80-dcu-standby-0a0b4accabc5-f13r3n14"]
TIME = "%Y-%m-%dT%H:%M:%S"
COLUMNS = "JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
WORKER_READBACK_SHA = "af211a7439d65a499654cdfbe996f474ab8a030e3fcec4ccf1f9b2a6e98b2cfc"
WORKER_SHA = "ab2c0fc162fb41c558aea87e61e53221b67686ab6126b8047c84758c68e1f1e2"
CONTRACT_SHA = "f8213f27675aadbd6a53c1acebe5f74e3dc5befbde0d65b57cfd98840577df3f"
READER_SHA = "8d62727e0a65bf5addbb3e09d12f62b244b4fe672284bb8fb877f222dd878b2e"


def need(test, code):
    if not test:
        raise ValueError(code)


def pin(raw):
    return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())


def read(path, maximum, digest=None):
    with path.open("rb") as f:
        raw = f.read(maximum + 1)
    need(not path.is_symlink() and 0 < len(raw) <= maximum and (digest is None or digest == pin(raw)["SHA256"]),
         "EXACT_BOUNDED_FILE")
    return raw


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def write(path, raw):
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return pin(raw)


def query(argv):
    child = subprocess.run(["env", "TZ=UTC", "LC_ALL=C", "SLURM_TIME_FORMAT=" + TIME, *argv],
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=9)
    need(child.returncode == 0 and len(child.stdout) <= 2 << 20, "BOUNDED_SLURM_ACCOUNTING")
    return child.stdout.decode("ascii").splitlines()


def accounting(id, start):
    return [row.split("|") for row in query(["sacct", "-S", start, "-nP", "-j", id, "-o", COLUMNS])
            if row.count("|") == 8]


def verified_predecessor():
    rows = [r for r in accounting(PREDECESSOR[0], "2026-11-26T20:30:00") if r[:2] == PREDECESSOR]
    need(len(rows) == 1 and rows[0][5] == NODE and "gres/dcu=8" in rows[0][6], "EXACT_OLD_STANDBY_STEP")
    old = rows[0]
    need(old[2:4] == ["COMPLETED", "0:0"] and old[8] != "Unknown", "WAIT_FOR_NATURAL_STANDBY_TERMINAL")
    return old


def exact_stage_claim(payload):
    request = payload["request"]
    need(re.fullmatch(r"[0-9a-f]{32}", request) and re.fullmatch(r"[0-9a-f]{40}", payload["publication"]), "EXACT_NEW_SCOPE")
    pins = payload["inspected_members"]
    names = {"binding.json", "vae-metadata.json", "src/worker.py", "src/contract.py", "src/readback.py", "media/render.json", "media/frames.jsonl"}
    names.update(f"media/frame-{i:02d}.private.png" for i in range(9))
    need(set(pins) == names and sum(x["bytes"] for x in pins.values()) == 3021024, "EXACT_INSPECTED_MEMBER_SET")
    table = dict(actual_frames=9, shape_THWC=[9, 480, 848, 3], member_count=16, bytes=3021024, files=pins)
    need(pin(encoded(table))["SHA256"] == "063b973a0e9a9c6703c8e1d3bcd40048d20ed45de469bc719b603bb49bfbe2ed", "SEALED_INSPECTED_MEMBERS")
    need(pins["src/worker.py"]["SHA256"] == WORKER_SHA and pins["src/contract.py"]["SHA256"] == CONTRACT_SHA
         and pins["src/readback.py"]["SHA256"] == READER_SHA, "FROZEN_WORKER_CONTRACT_AND_READER")
    for name, v in pins.items():
        need(pin(read(INSPECT / name, v["bytes"], v["SHA256"])) == v, "EXACT_INSPECTED_FILE")
    witness = json.loads(read(INSPECT / "independent-readback.json", 4096, WORKER_READBACK_SHA))
    need(witness["validated_frames"] == 9 and witness["mode"] == "inspect" and witness["accounting"][0] == "1103606.4190",
         "ORIGINAL_NINE_FRAME_TARGET_INSPECTION")
    return pins


def run_command(argv, root, name, timeout):
    with (root / (name + ".stdout")).open("xb") as out, (root / (name + ".stderr")).open("xb") as err:
        try:
            return subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return None


def controlled(payload):
    pins = exact_stage_claim(payload)
    previous = verified_predecessor()
    need(os.getuid() == 154459 and all(x.is_dir() for x in (PREVIOUS, RUNTIME, NATIVE, VAE, IMAGE))
         and os.access(APPTAINER, os.X_OK), "EXACT_LOCAL_RUNTIME_AND_OWNER")
    staging = b64decode(payload["placement_source"], validate=True)
    need(len(staging) < 16_384 and pin(staging) == payload["placement_pin"], "EXACT_GPU_ENTRY_SOURCE")
    ns = RESULTS / LABEL
    root = ns / payload["request"]
    job = LABEL + "-" + root.name[:12]
    start = datetime.fromtimestamp(time.time() - 120, timezone.utc).strftime(TIME)
    sys.path.insert(0, str(PREVIOUS))
    import placement

    with placement.shared_lock(PROJECT):
        placement.live_views(query, os.getuid(), time.time())
        need(not os.path.lexists(root) and not any(r[1].startswith(job + "-") for r in accounting(PARENT, start)),
             "SINGLE_NAMESPACE_AND_NO_PRIOR_STEP")
        os.umask(0o077)
        ns.mkdir(mode=0o700, exist_ok=True)
        need(not ns.is_symlink(), "CANONICAL_ENCODING_NAMESPACE")
        root.mkdir(mode=0o700)
        for name in ("home", "cache", "config", "tmp", "var-tmp", "src"):
            (root / name).mkdir(mode=0o700)
        write(root / "src/encode_placement.py", staging)
        epoch = time.time()
        intent = dict(request=root.name, publication=payload["publication"], claim_SHA256=payload["claim_SHA256"],
                      epoch=epoch, deadline=epoch + 600, parent=PARENT, node=NODE, step_prefix=job,
                      original_standby=previous, original_inspect_SHA256=WORKER_READBACK_SHA, replay_allowed=False)
        write(root / "intent.json", encoded(intent))
        authority = dict(request=root.name, label="training-hunyuanvideo-abo-vae-encode-v1", mode="encode",
                         recipe="ABO_RGB9_NCTHW_BF16_NATIVE_POSTERIOR_MODE_FP32_SCALED_V1", parent=PARENT, node=NODE,
                         root_authorized=True, network_GO=False, training_GO=False, text_GO=False, zero_GRES_GO=False,
                         GPU_GO=True, weight_payload_GO=True, allocatable_DCU=1, epoch=epoch, deadline=epoch + 600,
                         media_receipt_SHA256=pins["media/render.json"]["SHA256"], upstream_binding_SHA256=pins["binding.json"]["SHA256"],
                         vae_metadata_SHA256=pins["vae-metadata.json"]["SHA256"],
                         original_full_weight_SHA256="2609ed7033c052fdf164afcac71e8a7c82afccdbb001d673444a72f194fbb918")
        auth = write(root / "authority.json", encoded(authority))

        def slurm(phase):
            gpu = phase == "gpu"
            return ["srun", "--jobid=" + PARENT, "--job-name=" + job + "-" + phase, "--exact", "--immediate=6",
                    "--nodes=1", "--ntasks=1", "--nodelist=" + NODE, "--cpus-per-task=" + ("8" if gpu else "1"),
                    "--mem=" + ("128000M" if gpu else "1G"), "--gres=" + ("dcu:1" if gpu else "none"),
                    "--time=" + ("00:10:00" if gpu else "00:01:00"), "--kill-on-bad-exit=1", PYTHON, "-I", "-S", "-B",
                    str(root / "src/encode_placement.py"), "gate" if gpu else phase, str(root)]

        pre = run_command(slurm("pre"), root, "pre", 28)
        if pre != 0:
            result = dict(request=root.name, pre_rc=pre, actual_GPU_dispatch_attempted=False, old_standby=previous,
                          accounting=[r for r in accounting(PARENT, start) if r[1].startswith(job + "-")])
            write(root / "launcher-result.json", encoded(result))
            return result

        env = '''set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
unset PYTHONPYCACHEPREFIX MIOPEN_SYSTEM_DB_PATH MIOPEN_LOCKFILE_PATH MIOPEN_LOCKFILE_DIR
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/training-runtime/site
test -z "${LD_PRELOAD:-}${LD_AUDIT:-}${LD_PROFILE:-}"
export HOME="$1/miopen-private/private-home" XDG_CONFIG_HOME="$1/config" XDG_CACHE_HOME="$1/cache"
export MIOPEN_USER_DB_PATH="$HOME/.config/miopen/user-db" MIOPEN_CUSTOM_CACHE_DIR="$HOME/.cache/miopen"
export TMPDIR=/tmp TEMP=/tmp TMP=/tmp
shift
exec python3 -B -u "$@"'''
        apptainer = [APPTAINER, "exec", "--rocm", "--no-mount", "home,cwd", "--pwd", "/"]
        for source, target, flag in ((root, root, "rw"), (root / "src", root / "src", "ro"),
            (INSPECT, INSPECT, "ro"), (PREVIOUS, PREVIOUS, "ro"), (NATIVE, NATIVE, "ro"), (VAE, VAE, "ro"),
            (RUNTIME, "/training-runtime", "ro"), ("/opt/hyhal", "/opt/hyhal", "ro"),
            (root / "miopen-private/provider/libMIOpen.so.1.0", "/opt/dtk/lib/libMIOpen.so.1.0", "ro"),
            (root / "tmp", "/tmp", "rw"), (root / "var-tmp", "/var/tmp", "rw")):
            apptainer += ["--bind", f"{source}:{target}:{flag}"]
        apptainer += [str(IMAGE), "/bin/bash", "--noprofile", "--norc", "-c", env, LABEL, str(root),
                      str(root / "src/encode_placement.py"), "image", str(root), "encode", "--authority", str(root / "authority.json"),
                      "--authority-sha256", auth["SHA256"], "--request", root.name, "--media", str(INSPECT / "media"),
                      "--upstream-binding", str(INSPECT / "binding.json"), "--vae-metadata", str(INSPECT / "vae-metadata.json"),
                      "--out", str(root / "worker")]
        with (root / "gpu.stdout").open("xb") as out, (root / "gpu.stderr").open("xb") as err:
            child = subprocess.Popen([*slurm("gpu"), *apptainer], stdin=subprocess.DEVNULL, stdout=out, stderr=err)
            # Once the exact step is live, Slurm and the compute gate retain exclusive custody.
            while child.poll() is None and not (root / "gpu-gate.json").exists() and time.time() < epoch + 50:
                time.sleep(.2)
    try:
        rc = child.wait(timeout=max(1, epoch + 615 - time.time()))
    except subprocess.TimeoutExpired:
        rc = None
    post = None if rc is None else run_command(slurm("post"), root, "post", 28)
    rows = [r for r in accounting(PARENT, start) if r[1].startswith(job + "-")]
    event = root / "worker/worker.json"
    worker = None
    if event.is_file():
        raw = read(event, 16384)
        doc = json.loads(raw)
        need(doc["request"] == root.name and doc["optimizer_updates"] == 0, "EXACT_WORKER_RESULT")
        worker = dict(pin=pin(raw), **{k: doc[k] for k in ("passed", "actual_step", "phase", "actual_GPU",
            "validated_frames", "actual_native_encode_calls", "actual_candidate_latents")}, failure=doc.get("failure"))
    result = dict(request=root.name, pre_rc=pre, actual_GPU_dispatch_attempted=True, gpu_rc=rc, post_rc=post,
                  accounting=rows, worker=worker, old_standby=previous)
    for name in ("pre.json", "gpu-gate.json", "post.json"):
        path = root / name
        result[name] = pin(read(path, 16384)) if path.is_file() else None
    write(root / "launcher-result.json", encoded(result))
    return result


if __name__ == "__main__":
    wire = sys.stdin.buffer.read(96_001)
    need(0 < len(wire) <= 96_000, "EXACT_CONTROLLER_WIRE_SIZE")
    print(json.dumps(controlled(json.loads(wire)), sort_keys=True))
