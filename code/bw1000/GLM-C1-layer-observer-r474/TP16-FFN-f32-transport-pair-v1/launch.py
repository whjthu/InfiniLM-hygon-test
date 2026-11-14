"""Send the frozen one-shot only after the verified cluster has acknowledged READY."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import shlex
import subprocess
import sys
import threading
import uuid

repo = Path.cwd()
run = Path(sys.argv[1])
helper = run.parent / "glm-paired-readonly-connection.py"
assert hashlib.sha256(helper.read_bytes()).hexdigest() == "3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1"
spec = importlib.util.spec_from_file_location("cluster_connection", helper)
connection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(connection)
commit = sys.argv[2]
assert re.fullmatch(r"[0-9a-f]{40}", commit)
git = ["git", "-c", f"safe.directory={repo.as_posix()}", "--no-optional-locks"]
label = "GLM474-TP16-FFN-f32-transport-pair-v1"


def git_show(path):
    return subprocess.check_output(git + ["show", f"{commit}:{path}"], timeout=10)


def save(name, value):
    with (run / name).open("x") as output:
        json.dump(value, output, indent=2)
        output.flush()
        os.fsync(output.fileno())


task = json.loads(git_show(f"coordination/tasks/{label}.json"))
assert task["GPU_GO"] and not task["Model_GO"] and task["max_experiment_invocations"] == 1
assert task["actual_experiment_invocations"] == 0 and not task["replay_allowed"]
for path, pin in task["pinned_references"].items():
    assert hashlib.sha256(git_show(path)).hexdigest() == pin
files = {}
for name, pin in task["source_SHA256"].items():
    payload = git_show(task["source"] + "/" + name)
    assert hashlib.sha256(payload).hexdigest() == pin
    if name == "receive_once.py":
        controller = payload
    elif name != "launch.py":
        files[name] = dict(b64=base64.b64encode(payload).decode(), bytes=len(payload), SHA256=pin)

request = uuid.uuid4().hex
nonce = uuid.uuid4().hex
payload = f"CLAIM={commit!r}\nREQUEST={request!r}\nFILES={files!r}\n".encode() + controller
wire = json.dumps(dict(nonce=nonce, code=base64.b64encode(payload).decode(),
                       sha256=hashlib.sha256(payload).hexdigest())).encode() + b"\n"
bootstrap = r"""
import base64, hashlib, json, sys
nonce=NONCE
print(json.dumps(dict(stage='READY', nonce=nonce)),flush=True)
line=sys.stdin.buffer.readline(262145)
assert len(line)<=262144 and line.endswith(b'\n')
wire=json.loads(line)
assert wire['nonce']==nonce
payload=base64.b64decode(wire['code'],validate=True)
assert hashlib.sha256(payload).hexdigest()==wire['sha256']
exec(compile(payload,'published-glm-tp16-ffn-f32','exec'), {'__name__':'__main__'})
""".replace("NONCE", repr(nonce))
intent = dict(commit=commit, request_id=request, wire_sha256=hashlib.sha256(wire).hexdigest(),
              replay_allowed=False)
assert not (run / "mutating-intent.json").exists()
command = "/usr/bin/timeout --kill-after=7s 820s /usr/local/python3.12/bin/python3 -B -u -c " + shlex.quote(bootstrap)
started = False
error_type = None
stdout = b""
attempt = uuid.uuid4().hex
with (run / (attempt + "-stderr.private")).open("xb") as err:
    with subprocess.Popen(connection.ssh + [command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=err) as proc:
        ready = queue.Queue()
        threading.Thread(target=lambda: ready.put(proc.stdout.readline()), daemon=True).start()
        try:
            assert json.loads(ready.get(timeout=18)) == dict(stage="READY", nonce=nonce)
            save("mutating-intent.json", intent)
            started = True
            stdout, _ = proc.communicate(wire, timeout=825)
        except Exception as error:
            error_type = type(error).__name__
            if proc.poll() is None:
                proc.kill()
            tail, _ = proc.communicate(timeout=8)
            stdout += tail
        rc = proc.returncode
with (run / (attempt + "-stdout.private")).open("xb") as output:
    output.write(stdout)

outcome = None
if started and not rc:
    try:
        outcome = json.loads(stdout)
        assert outcome["request_id"] == request and outcome["claim"] == commit
    except (ValueError, KeyError, AssertionError):
        outcome = None
summary = dict(**intent, payload_dispatch_attempted=started, returncode=rc, error=error_type,
               private_stdout_sha256=hashlib.sha256(stdout).hexdigest(),
               remote_outcome=("RETURNED" if outcome else "UNKNOWN" if started else "NO_PAYLOAD_SENT"))
if outcome:
    summary.update(component_passed=outcome['passed'],
                   phases={name: receipt['row'] for name,receipt in
                       dict(pre=outcome['pre'],**outcome['arms'],post=outcome['post']).items() if receipt},
                   ranks={name:r['ranks'] for name,r in outcome['rank_data'].items()},
                   controller_error=outcome['error'])
save(attempt + "-outcome.json", summary)
print(json.dumps(summary))
