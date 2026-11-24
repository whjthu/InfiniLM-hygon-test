"""Send the fixed T07 GPU request once after new device-only admission."""
import argparse
from base64 import b64encode
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIVATE = REPO / "local-only"
REQUEST = "8dcecfc1023c4bb5bc05f4bccb850b1f"
NEW_ADMISSION = "783da17f2c93470587254aad37dd1a24"
OLD_ADMISSION = "42f59a4f4f004d6db5ac2c4560afc034"
NODE, PARENT = "f13r2n07", "1103606"
PYTHON = "/usr/local/python3.12/bin/python3"
SOURCE_PINS = {
    "code/bw1000/training80-t07-2429-successor-refill-v1/launch.py": "349f03f0ce2b3820f32d42f1e973e5d67a6d76a41ab7c57a31aa8ae3b108a223",
    "code/bw1000/training80-t07-2429-successor-refill-v1/guard.py": "bd47166ab8661342216dcde0d647c21ab6cdbae1f26a59ea92c09effab354c03",
    "local-only/run-minicpm-sdpa-target-asset-preflight-original.py": "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd",
    "local-only/run-training80-T05-2417-audited-refill-f67445c1a3204a708cd68006b03cbd9d.py": "7d9ce780222e6b87ab54b3b08c3cdacd95b99fe44d384fc2cda6dbce7eaf4766",
    "code/bw1000/fleet-dcu-standby-v2/controller.py": "be504e7670b40d77e503c60c93d76bf2a75003768d764d54884c5f8272976bba",
    "code/bw1000/fleet-dcu-standby-v2/worker.py": "dc8820e17b8d4d2886325c6093ad8884e6a5a1085e0eedd40eb53cfb12235059",
    "code/bw1000/fleet-dcu-standby-v2/CLAIM.json": "1f441531cc4c9fdbf3226fb887242d693475f4e4eca603cf0ce69ccc7f7299cd",
    "code/bw1000/fleet-dcu-standby-compute-bridge-v1/bridge.py": "2e796e5a136bb7fcac5892b49a34f0fd81fabd1f6f8774945725a0b199e58360",
}
OLD_RECEIPTS = {
    f"training80-T07-2429-successor-{REQUEST}-single-call.json": "008ea8b511d0e580ac9483398149386128de3e09c93aff88ea578b63c898b7eb",
    f"training80-T07-2429-{REQUEST}-launch-local-stdout.txt": "50e944354bafc1e04d303bf5386c5542e9b1f1b556f4e82cb0634d31e96907d8",
    "training80-T07-2429-admission42f59a-refill8dce-reconcile-fc0905f9115644359bdf6c1b57116b1b.json": "31ea8ea11485136816cf8547c0b8e8c6623967c8fc7b132c6eaae873913e9829",
}


def digest(raw):
    return sha256(raw).hexdigest()


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def save(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(data)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def locally_unsent():
    first, output, reconcile = [(PRIVATE / name).read_bytes() for name in OLD_RECEIPTS]
    need([digest(raw) for raw in (first, output, reconcile)] == list(OLD_RECEIPTS.values()),
         "original terminal-before-GPU-send evidence differs")
    intent, scope = json.loads(first), json.loads(reconcile)["observed"]
    need(intent["request"] == scope["request"] == REQUEST and intent["admission"] == scope["admission"] == OLD_ADMISSION
         and output.decode("utf-8-sig").strip().startswith("Bounded admission did not complete 255 ")
         and not scope["controller"] and not scope["exact_accounting"]
         and not any(row["exists"] or row["symlink"] for row in scope["namespaces"].values())
         and not (PRIVATE / f"training80-T05-refill-transfer-{REQUEST}").exists(),
         "original launcher is not provably before GPU send")


def protected_jobs():
    sys.path.insert(0, str(REPO / "tools"))
    from report_training80_backlog import read_original_holds, read_queue_holds, TRAINING80_LEDGER
    from training_dispatch_allocation import allocation, REPLACEMENT_LEDGER
    holds = read_queue_holds(PRIVATE / "tuning/queue.sqlite3")
    jobs = json.loads((REPO / "coordination/training/training80-backlog-v1.json").read_bytes())["items"]
    known = set(allocation(REPLACEMENT_LEDGER.read_bytes())[2]) | set(allocation(TRAINING80_LEDGER.read_bytes())[2])
    direct, _, _ = read_original_holds(REPO, jobs, known, holds)
    secured = set().union(*(set(row["nodes"]) for row in (*holds, *direct.values())))
    need(NODE not in secured and not any(
        j["kind"] == "training" and j["state"] == "ready" and NODE in j["placement"] for j in jobs),
        "project reservation or ready training takes priority")


def render(source, old_launch):
    guard_path = REPO / "code/bw1000/training80-t07-2429-successor-refill-v1/guard.py"
    guard = guard_path.read_text(encoding="utf8")
    guard = guard.replace(OLD_ADMISSION, NEW_ADMISSION)
    marker = '    certificate = envelope["t07_successor_device"]'
    insert = marker + f'''
    receipt_dir = results / "training80-t07-device-only-{NEW_ADMISSION}"
    observed = bounded(receipt_dir / "receipt.json")
    raw_pulse = bounded(receipt_dir / "worker.stdout.private")
    record = json.loads(observed)
    require(record["observer_SHA256"] == {source!r}
            and record["status"] == "ZERO_GRES_DEVICE_CLEAN" and record["gpu_request"] == REQUEST
            and record["request"] == ADMISSION and record["certified_observation"] == certificate
            and certificate["observer_uid"] == os.getuid()
            and digest(raw_pulse) == certificate["payload_SHA256"],
            "independently observed PCI or proof identity differs")
'''
    need(guard.count(marker) == 1, "frozen dispatch instrumentation anchor differs")
    guard = guard.replace(marker, insert)
    old_bridge = REPO / "code/bw1000/fleet-dcu-standby-compute-bridge-v1/bridge.py"
    return old_launch["bridge"](old_bridge.read_text(encoding="utf8"), guard.encode())


def package():
    for rel, pin in SOURCE_PINS.items():
        need(digest((REPO / rel).read_bytes()) == pin, "pinned existing entry point differs")
    new_claim = json.loads((HERE / "CLAIM.json").read_bytes())
    need(new_claim["GPU_request"] == REQUEST and new_claim["device_request"] == NEW_ADMISSION
         and new_claim["protected_original_zero_GRES_request"] == OLD_ADMISSION
         and all(digest((HERE / name).read_bytes()) == pin for name, pin in new_claim["files"].items()),
         "new scoped claim bytes differ")
    probe = (HERE / "observe.py").read_bytes()
    compile(probe, "<direct narrow T07 device observation>", "exec")
    old_entry = REPO / "code/bw1000/training80-t07-2429-successor-refill-v1/launch.py"
    old = runpy.run_path(str(old_entry))
    bridge = render(digest(probe), old)
    return old, probe, bridge


def probe_once(argv, source, leaf, mode="observe"):
    command = shlex.join([PYTHON, "-I", "-S", "-B", "-u", "-c", source.decode(), mode, digest(source)])
    start = time.monotonic()
    try:
        result = subprocess.run([argv[0], "-v", *argv[1:], command], stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=52)
        out, err, rc = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        out, err, rc = error.stdout or b"", error.stderr or b"", None
    duration = time.monotonic() - start
    save(leaf / "observer-ssh-stderr.private", err)
    save(leaf / "observer-stdout.json", out)
    audit = dict(request=NEW_ADMISSION, mode=mode, rc=rc, duration=duration, ssh_authenticated=b"Authenticated to" in err,
                 ssh_command_sent=b"Sending command:" in err,
                 preauth_connection_error=any(x in err.lower() for x in (
                     b"kex_exchange_identification:", b"connection reset by peer", b"connection timed out")),
                 stderr_SHA256=digest(err), stdout_SHA256=digest(out), stdout_bytes=len(out))
    save(leaf / "observer-transport.json", json_bytes(audit))
    print(json.dumps({"observer_transport": audit}, sort_keys=True), flush=True)
    need(rc == 0 and 0 < len(out) <= 8192 and duration <= 50, "bounded narrow observation did not complete")
    return json.loads(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--publication")
    parser.add_argument("--terminal", type=Path)
    parser.add_argument("--sha")
    args = parser.parse_args()
    old, probe, bridge = package()
    locally_unsent()
    protected_jobs()
    if args.check:
        print(json.dumps(dict(offline_check=True, GPU_request=REQUEST, device_request=NEW_ADMISSION,
                              bridge_SHA256=digest(bridge), remote_calls=0), sort_keys=True))
        return
    need(args.publication and re.fullmatch("[a-f0-9]{40}", args.publication)
         and args.publication != "0" * 40 and args.terminal and args.sha and re.fullmatch("[a-f0-9]{64}", args.sha),
         "verified publication and fresh actual old terminal evidence required")
    old["predecessor"](args.terminal, args.sha, ready=True)
    path = PRIVATE / f"training80-T07-2429-narrow-device-{NEW_ADMISSION}"
    path.mkdir(mode=0o700)
    save(path / "intent.json", json_bytes(dict(request=NEW_ADMISSION, GPU_request=REQUEST, old_device_request=OLD_ADMISSION,
         publication=args.publication, bridge_SHA256=digest(bridge), observer_SHA256=digest(probe),
         terminal_SHA256=args.sha, replay_allowed=False)))
    transport = runpy.run_path(str(PRIVATE / "run-minicpm-sdpa-target-asset-preflight-original.py"))
    connection = transport["ssh"](REPO.parent)
    result = probe_once(connection, probe, path)
    certificate = result["certified_observation"]
    need(result["request"] == NEW_ADMISSION and result["gpu_request"] == REQUEST
         and result["status"] == "ZERO_GRES_DEVICE_CLEAN" and result["observer_SHA256"] == digest(probe)
         and certificate["request_id"] == NEW_ADMISSION and certificate["parent"] == PARENT and certificate["node"] == NODE
         and certificate["active_steps"] == [] and certificate["kfd_process_count"] == certificate["own_device_fd_process_count"] == 0
         and result["original_predecessor"][0] == "1103606.2429" and result["old_bridge"][0] == "1103606.2428"
         and result["device_accounting"][2:4] == ["COMPLETED", "0:0"], "independent device receipt differs")
    locally_unsent()
    protected_jobs()
    cutoff = time.monotonic() + 25
    stock = REPO / "code/bw1000/fleet-dcu-standby-v2"
    worker = (stock / "worker.py").read_bytes()
    controller = (stock / "controller.py").read_bytes()
    claim = json.loads((stock / "CLAIM.json").read_bytes())
    binding = dict(request=REQUEST, publication=args.publication, seconds=10800,
                   nodes={NODE: {"parent": PARENT, "pci": certificate["PCIs"]}}, worker=worker, claim=claim,
                   admission_SHA256=certificate["payload_SHA256"])
    entry = ("INPUT=" + repr(binding) + "\n").encode() + controller
    payload = json.dumps(dict(request=REQUEST, entry=b64encode(entry).decode(), entry_SHA256=digest(entry),
          bridge=b64encode(bridge).decode(), bridge_SHA256=digest(bridge), t07_successor_device=certificate),
          sort_keys=True).encode()
    command = shlex.join([PYTHON, "-I", "-S", "-B", "-c", bridge.decode(), "start", "--launch-only"])
    previous_audit = runpy.run_path(str(PRIVATE / "run-training80-T05-2417-audited-refill-f67445c1a3204a708cd68006b03cbd9d.py"))
    audited = previous_audit["audited_bridge"]
    audited.__globals__.update(REFILL=REQUEST, STEP="1103606.2429", GATE=old["GATE"])
    output = audited(connection, command, payload, cutoff)
    save(path / "gpu-stdout.json", output.stdout)
    need(output.returncode == 0 and output.stdout, "GPU transport did not acknowledge the one fixed payload")
    print(output.stdout.decode().strip())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(request=REQUEST, failure=type(error).__name__,
                              failure_SHA256=digest(str(error).encode("utf8", "replace")))))
        raise SystemExit(1)
