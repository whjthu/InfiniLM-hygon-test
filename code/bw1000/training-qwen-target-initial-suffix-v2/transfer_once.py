"""Drive the one-shot A100 Qwen initial trainables transfer from a published task."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
import uuid

from stream import CHUNK, COMMIT_FRAME, frame, require

TASK = "coordination/tasks/training-qwen-target-initial-suffix-v2.json"
RATE = 1024 * 1024


def save(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def verify_receipt(path, expected):
    raw = path.read_bytes()
    require(len(raw) <= 32768, "Unexpected endpoint output size")
    records = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("request_id") == expected["request_id"]:
            records.append(row)
    require(records == [expected], "Endpoint did not confirm the exact plan")
    return hashlib.sha256(raw).hexdigest()


def relay(sender, receiver, expected_bytes, remaining, pace, progress):
    copied = last_report = 0
    while True:
        require(remaining() > 15, "Transfer closing reserve exhausted")
        data = sender.stdout.read(CHUNK)
        if not data:
            break
        copied += len(data)
        require(copied <= expected_bytes, "Sender exceeded the registered byte count")
        require(receiver.stdin.write(data) == len(data), "Short pipe write")
        pace(copied)
        if copied - last_report >= 64 * CHUNK:
            progress(copied)
            last_report = copied
    require(copied == expected_bytes, "Sender stream is incomplete")
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("receipt_directory", type=Path)
    parser.add_argument("hygon_connection", type=Path)
    parser.add_argument("a100_ssh_config", type=Path)
    args = parser.parse_args()
    repo = Path.cwd()
    require(len(args.commit) == 40 and all(c in "0123456789abcdef" for c in args.commit), "Invalid claim commit")
    git = ["git", "-c", "safe.directory=" + repo.as_posix()]

    def show(path):
        return subprocess.check_output(git + ["show", args.commit + ":" + path], timeout=15)

    task = json.loads(show(TASK))
    require(task["transfer_GO"] is True and task["GPU_GO"] is False
            and task["max_invocations"] == 1, "Invalid continuation claim")
    blobs = {}
    for name, digest in task["source_SHA256"].items():
        raw = show(task["source"] + "/" + name)
        require(hashlib.sha256(raw).hexdigest() == digest, "Published package changed")
        blobs[name] = raw
    for name in ("transfer_once.py", "stream.py"):
        require(Path(__file__).with_name(name).read_bytes() == blobs[name], "Local source differs")
    plan = json.loads(blobs["PLAN.json"])
    require(plan["relay_bytes"] == task["suffix_bytes"] == plan["file"]["bytes"] - plan["file"]["offset"],
            "Wrong suffix extent")
    evidence = task["prior_evidence"]
    require(hashlib.sha256(show(evidence["path"])).hexdigest() == evidence["SHA256"],
            "Original terminal evidence changed")
    helper_sha = hashlib.sha256(args.hygon_connection.read_bytes()).hexdigest()
    require(helper_sha == task["hygon_connection_sha256"], "Existing SSH connection helper changed")
    spec = importlib.util.spec_from_file_location("cluster_connection", args.hygon_connection)
    connection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connection)
    target = connection.ssh
    source = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
              "-F", str(args.a100_ssh_config), "a100-26"]
    out = args.receipt_directory
    out.mkdir(mode=0o700)
    request = uuid.uuid4().hex
    epoch = time.time()
    start = time.monotonic()
    budget = plan["timeout_seconds"]
    header_bytes = len(frame(plan["name"], plan["file"]))
    expected_bytes = plan["relay_bytes"] + header_bytes

    def remaining():
        return min(epoch + budget - time.time(), start + budget - time.monotonic())

    def pace(copied):
        delay = max(0, start + copied / RATE - time.monotonic())
        require(delay + 15 < remaining(), "Rate limit cannot retain closing reserve")
        time.sleep(delay)

    common = dict(claim=args.commit, request_id=request, epoch=epoch)
    endpoint = blobs["stream.py"].decode()
    save(out / "intent.json", dict(common, deadline=epoch + budget, wire_bytes=expected_bytes,
                                   suffix_bytes=plan["relay_bytes"], replay_allowed=False))

    def command(ssh, interpreter, role):
        code = endpoint + "\nraise SystemExit(run_endpoint(" + repr(plan) + ", " + repr(args.commit)
        code += ", " + repr(request) + ", " + repr(epoch) + ", " + repr(role)
        code += "))\n"
        return ssh + [interpreter + " -B -u -c " + shlex.quote(code)]

    def expected(role):
        return dict(common, role=role, passed=True, suffix_bytes=plan["relay_bytes"], files={plan["name"]: plan["file"]})

    def progress(copied):
        temp = out / "progress.tmp"
        save(temp, dict(common, wire_bytes=copied, observed_epoch=time.time()))
        temp.replace(out / "progress.json")

    receiver_cmd = command(target, "/usr/local/python3.12/bin/python3", "receiver")
    sender_cmd = command(source, "python3", "sender")
    require(max(len(subprocess.list2cmdline(cmd)) for cmd in (receiver_cmd, sender_cmd)) < 30000,
            "SSH command exceeds the Windows process limit")
    children = []
    result = dict(common, passed=False)
    stop_lock = threading.Lock()

    def stop():
        with stop_lock:
            for process in reversed(children):
                if process.poll() is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass

    timer = threading.Timer(remaining(), stop)
    timer.daemon = True
    sender_path = out / "sender.stderr.private"
    receiver_path = out / "receiver.stdout.private"
    try:
        with sender_path.open("xb") as se, receiver_path.open("xb") as ro, (out / "receiver.stderr.private").open("xb") as re:
            timer.start()
            with stop_lock:
                receiver = subprocess.Popen(receiver_cmd,
                                            stdin=subprocess.PIPE, stdout=ro, stderr=re)
                children.append(receiver)
                sender = subprocess.Popen(sender_cmd,
                                          stdout=subprocess.PIPE, stderr=se)
                children.append(sender)
            save(out / "transports.json", dict(common, sender_pid=sender.pid, receiver_pid=receiver.pid))
            result["wire_bytes"] = relay(sender, receiver, expected_bytes, remaining, pace, progress)
            require(sender.wait(timeout=max(.01, remaining())) == 0, "Sender failed")
            result["sender_receipt_sha256"] = verify_receipt(sender_path, expected("sender"))
            require(receiver.stdin.write(COMMIT_FRAME) == len(COMMIT_FRAME), "Short commit frame write")
            receiver.stdin.close()
            require(receiver.wait(timeout=max(.01, remaining())) == 0, "Receiver failed")
            result["receiver_receipt_sha256"] = verify_receipt(receiver_path, expected("receiver"))
            result["passed"] = True
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["error_type"] = type(error).__name__
    finally:
        timer.cancel()
        stop()
        for process in children:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                result["cleanup_pending"] = True
        result["transport_rcs"] = [process.returncode for process in children]
        save(out / "outcome.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
