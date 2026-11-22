"""Run a frozen source-to-target relay with no model data written on this computer."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time

from endpoint import CHUNK, CHECKPOINT_SIZE, COMMIT, frame, validate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOCAL = REPO / "local-only" / HERE.name

def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_scope(plan, official, proof, old):
    prior = plan["prior"]
    item = official["public_metadata"]["shards"][plan["file"]]
    validate((plan["file"], plan["bytes"], plan["prefix_bytes"], plan["network_suffix_bytes"],
              plan["rate_bytes_per_second"], plan["timeout_seconds"]) ==
             ("model-00001-of-00004.safetensors", 4995008464, 4442095616, 552912848, CHUNK, 1800)
             and plan["bytes"] == plan["prefix_bytes"] + plan["network_suffix_bytes"]
             and official["official_revision"] == plan["revision"]
             and item["bytes"] == plan["bytes"] == plan["source_stat"]["size"]
             and item["lfs_sha256"] == plan["SHA256"] and item["bytes_match"] is item["sha256_match"] is True,
             "OFFICIAL_SHARD_SCOPE")
    validate(old["request"] == prior["request"] and old["state"] == "LOCAL_RELAY_FAILED_TARGET_UNVERIFIED"
             and old["sent_commit"] is False and old["receiver_exit"] == 1 and old["sender_exit"] == 255
             and old["original_receiver_intent_prior_observed_SHA256"] == prior["intent_SHA256"],
             "FAILED_ORIGINAL_LOCAL_RELAY")
    validate(proof["original_request"] == prior["request"]
             and proof["state"] == "ORIGINAL_RECEIVER_FAILED_RETAINED_TARGET_PREFIX_INDEPENDENT_SHA_VERIFIED"
             and proof["independent_zero_gres_accounting"][0] == "1087656.949"
             and proof["independent_zero_gres_accounting"][2:4] == ["COMPLETED", "0:0"]
             and proof["independent_zero_gres_accounting"][5] == "a14r3n14"
             and "gres/" not in proof["independent_zero_gres_accounting"][6]
             and proof["original_receiver_terminal_SHA256"] == prior["terminal_SHA256"]
             and proof["original_prefix_receipt_SHA256"] == prior["prefix_receipt_SHA256"]
             and proof["actual_prefix_bytes"] == plan["prefix_bytes"]
             and proof["actual_prefix_sha"] == prior["prefix_SHA256"]
             and proof["still_missing_suffix_bytes"] == plan["network_suffix_bytes"]
             and proof["original_replayed"] is proof["full_file_qualified"] is False, "TARGET_PREFIX_PROOF")
    return dict(target_proof=dict(request=prior["request"], offset=plan["prefix_bytes"],
                                  prefix_SHA256=prior["prefix_SHA256"]),
                published_SHA256=prior["independent_evidence"]["SHA256"])


def open_published(publication):
    validate(re.fullmatch(r"[0-9a-f]{40}", publication), "EXACT_PUBLICATION")
    git = ["git", "-c", "safe.directory="+REPO.as_posix()]
    relative = HERE.relative_to(REPO).as_posix()

    def published(path):
        return subprocess.check_output([*git, "cat-file", "blob", publication+":"+path], cwd=REPO, timeout=12)

    manifest = (HERE / "SOURCE_FILES.json").read_bytes()
    validate(published(relative+"/SOURCE_FILES.json") == manifest, "PUBLISHED_MANIFEST_CHANGED")
    entries = json.loads(manifest)["files"]
    validate({"CLAIM.json","PLAN.json","endpoint.py","relay.py"} <= set(entries), "PUBLISHED_SCOPE_INCOMPLETE")
    for name, pin in entries.items():
        validate(Path(name).name == name and name not in (".",".."), "UNSAFE_SOURCE_MEMBER")
        raw = (HERE / name).read_bytes()
        validate((len(raw),sha(raw)) == (pin["bytes"],pin["SHA256"])
                 and raw == published(relative+"/"+name), "PUBLISHED_SOURCE_FILE_CHANGED")
    plan = json.loads((HERE / "PLAN.json").read_bytes())
    claim = json.loads((HERE / "CLAIM.json").read_bytes())
    validate((claim["request"],claim["target"],claim["network_suffix_bytes"]) ==
             (plan["request"],plan["target"],plan["network_suffix_bytes"])
             and claim["transfer_GO"] is True and claim["max_transfer_invocations"] == 1
             and claim["replay_allowed"] is False, "ROOT_PUBLISHED_SCOPE_REQUIRED")

    def evidence(pin):
        raw = published(pin["path"])
        validate(len(raw) <= 65536 and sha(raw) == pin["SHA256"], "PUBLISHED_EVIDENCE_CHANGED")
        return json.loads(raw)

    prior = plan["prior"]
    binding = validate_scope(plan, evidence(plan["official_evidence"]),
                             evidence(prior["independent_evidence"]), evidence(prior["local_terminal"]))
    return plan, binding


def command(plan, binding, role, budget, transport_argv):
    interpreter = ("/home/whj/nvidia-baseline/env-vlm/bin/python" if role == "sender"
                   else "/usr/local/python3.12/bin/python3")
    body = (HERE / "endpoint.py").read_text(encoding="utf-8")
    body += "\nraise SystemExit(endpoint("+repr(plan)+","+repr(binding)+","+repr(role)+","+str(budget)+"))\n"
    cmd = interpreter + " -I -S -B -u -c " + shlex.quote(body)
    argv = transport_argv("source" if role == "sender" else "target", cmd)
    validate(type(argv) is list and argv and all(isinstance(part,str) for part in argv)
             and len(subprocess.list2cmdline(argv)) < 30000, "BOUNDED_AUTHORIZED_TRANSPORT_ARGV")
    return argv


def expected_receipt(path, binding, role):
    with path.open("rb") as source:
        raw = source.read(32769)
    validate(len(raw) <= 32768, 'RECEIPT_SIZE')
    rows = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get('request') == binding['request'] and row.get('role') == role:
            rows.append(row)
    validate(len(rows) == 1 and all(rows[0][k] == v for k,v in binding.items())
             and rows[0]['role'] == role and rows[0]['passed'], 'ENDPOINT_RECEIPT')
    return rows[0], sha(raw)


def wait_receiver_ready(log, process, binding, remaining):
    limit = time.monotonic() + min(300, remaining() - 120)
    expected = dict(request=binding["request"], role="receiver_ready", durable_bytes=binding["resume"]["offset"],
                    prefix_SHA256=binding["resume"]["prefix_SHA256"], publication=binding["publication"])
    while time.monotonic() < limit:
        with log.open("rb") as source:
            raw = source.read(32769)
        validate(len(raw) <= 32768, "RECEIVER_READY_RECEIPT_BOUND")
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("role") == "receiver_ready":
                validate(row == expected and process.poll() is None, "LIVE_EXACT_RECEIVER_READY")
                return row
        validate(process.poll() is None, "RECEIVER_TERMINAL_BEFORE_SOURCE_STARTED")
        time.sleep(0.15)
    raise TimeoutError("RECEIVER_PRECOPY_DEADLINE")


def transfer(plan, publication, resume, transport_argv):
    LOCAL.mkdir(mode=0o700)
    epoch = time.time()
    started = time.monotonic()
    budget = plan['timeout_seconds']
    binding = {k:plan[k] for k in ('request','file','bytes','SHA256')}
    binding.update(publication=publication, epoch=epoch, deadline=epoch+budget,
                   resume=resume['target_proof'], published_binding_SHA256=resume['published_SHA256'])
    save(LOCAL/'intent.json', binding)
    children = []
    lock = threading.Lock()

    def remaining():
        return min(binding['deadline']-time.time(), started+budget-time.monotonic())

    def stop():
        with lock:
            for child in reversed(children):
                if child.poll() is None:
                    try:
                        child.kill()
                    except ProcessLookupError:
                        pass

    total = len(frame(binding)) + plan['bytes'] - binding['resume']['offset']
    timer = threading.Timer(max(0, remaining()), stop)
    timer.daemon = True
    sender_log = LOCAL/'sender.stderr.private'
    receiver_log = LOCAL/'receiver.stdout.private'
    copied = 0
    report = 0
    result = dict(binding, passed=False, local_weight_bytes_written=0,
                  network_suffix_planned=plan['bytes']-binding['resume']['offset'])
    try:
        receiver_cmd = command(plan, binding, 'receiver', budget, transport_argv)
        sender_cmd = command(plan, binding, 'sender', budget, transport_argv)
        with sender_log.open('xb') as se, receiver_log.open('xb') as ro, (LOCAL/'receiver.stderr.private').open('xb') as re:
            timer.start()
            with lock:
                receiver = subprocess.Popen(receiver_cmd, stdin=subprocess.PIPE, stdout=ro, stderr=re, cwd=REPO)
                children.append(receiver)
            ready = wait_receiver_ready(receiver_log, receiver, binding, remaining)
            save(LOCAL / "receiver-ready.json", ready)
            with lock:
                sender = subprocess.Popen(sender_cmd, stdout=subprocess.PIPE, stderr=se, cwd=REPO)
                children.append(sender)
            save(LOCAL/'transports.json', dict(sender_pid=sender.pid, receiver_pid=receiver.pid, request=plan['request']))
            pace_origin = None
            while True:
                validate(remaining() > 12, 'RELAY_CLOSING_RESERVE')
                chunk = sender.stdout.read(CHUNK)
                if not chunk:
                    break
                if pace_origin is None:
                    pace_origin = time.monotonic()
                copied += len(chunk)
                validate(copied <= total and receiver.stdin.write(chunk) == len(chunk), 'BOUND_PIPE_TRANSFER')
                delay = max(0, pace_origin+copied/plan['rate_bytes_per_second']-time.monotonic())
                validate(delay+12 < remaining(), 'RELAY_PACE_EXHAUSTS_RESERVE')
                time.sleep(delay)
                if copied - report >= CHECKPOINT_SIZE:
                    report = copied
                    stage = LOCAL/'progress.tmp'
                    save(stage, dict(request=plan['request'], wire_bytes=copied, observed_epoch=time.time()))
                    stage.replace(LOCAL/'progress.json')
            validate(copied == total and sender.wait(timeout=max(.1,remaining())) == 0, 'SOURCE_TERMINAL')
            _, result['sender_receipt_SHA256'] = expected_receipt(sender_log,binding,'sender')
            validate(receiver.stdin.write(COMMIT) == len(COMMIT), 'SHORT_COMMIT_FRAME')
            receiver.stdin.close()
            validate(receiver.wait(timeout=max(.1,remaining())) == 0, 'TARGET_TERMINAL')
            row, result['receiver_receipt_SHA256'] = expected_receipt(receiver_log,binding,'receiver')
            validate(row['whole_SHA256'] == plan['SHA256'] and row['durable_bytes'] == plan['bytes'], 'TARGET_PROOF')
            result['passed'] = True
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        result.update(error=type(exc).__name__,
                      failure_code=str(exc) if isinstance(exc, ValueError) else None)
    finally:
        timer.cancel()
        stop()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                result['local_cleanup_pending'] = True
        result.update(wire_bytes=copied, transport_rcs=[c.returncode for c in children],
                      suffix_bytes_in_pipe_upper_bound=max(0, copied-len(frame(binding))),
                      elapsed_seconds=time.monotonic()-started)
        save(LOCAL/'outcome.json', result)
    print(json.dumps(result,allow_nan=False),flush=True)
    return int(not result['passed'])


def run(publication, transport_argv, *, audit=False):
    """Run once using the root's already authenticated source and target argv builder."""
    plan, resume = open_published(publication)
    if not audit:
        return transfer(plan, publication, resume, transport_argv)
    binding = json.loads((LOCAL/"intent.json").read_bytes())
    outcome = json.loads((LOCAL/"outcome.json").read_bytes())
    validate(binding["publication"] == publication and binding["request"] == plan["request"]
             and outcome["request"] == binding["request"] and outcome["passed"] is True
             and not (LOCAL/"independent-intent.json").exists(), "ORIGINAL_NEW_SUCCESS_REQUIRED")
    save(LOCAL/"independent-intent.json", dict(request=plan["request"], publication=publication))
    r = subprocess.run(command(plan,binding,"audit",480,transport_argv), capture_output=True, timeout=490)
    validate(r.returncode == 0 and 1 <= len(r.stdout) <= 16384, "INDEPENDENT_AUDIT_RECEIPT")
    value = json.loads(r.stdout)
    validate(all(value[k] == v for k,v in binding.items()) and value["verified"] is True
             and value["bytes"] == plan["bytes"] and value["SHA256"] == plan["SHA256"],
             "INDEPENDENT_WHOLE_SHA")
    save(LOCAL/"independent.json",value)
    print(json.dumps(value,allow_nan=False),flush=True)
    return 0
