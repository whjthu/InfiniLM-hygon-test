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
label = "GLM474-model-transport-gpu-v1"


def git_show(path):
    return subprocess.check_output(git + ["show", f"{commit}:{path}"], timeout=10)


def save(name, value):
    with (run / name).open("x") as output:
        json.dump(value, output, indent=2)
        output.flush()
        os.fsync(output.fileno())


task = json.loads(git_show(f"coordination/tasks/{label}.json"))
assert task["GPU_GO"] and not task["replay_allowed"] and task["max_invocations"] == 1
assert task["actual_invocations"] == 0
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == task["source_SHA256"]["launch.py"]
for path, pin in task["pinned_references"].items():
    assert hashlib.sha256(git_show(path)).hexdigest() == pin
package = task["package"]
resource = git_show(task["resource_claim"])
assert hashlib.sha256(resource).hexdigest() == task["resource_claim_SHA256"]
claim = json.loads(resource)
assert claim["GPU_GO"] is True and claim["run_id"] == task["run_id"]
assert claim["arm_order"] == ["socket", "shca"] and claim["native_arms"]["socket"] == claim["native_arms"]["shca"]
files = {}
for name, pin in task["package_files"].items():
    raw = git_show(package + "/" + name)
    assert hashlib.sha256(raw).hexdigest() == pin
    files[name] = (pin, base64.b64encode(raw).decode())
controller = base64.b64decode(files["dispatch_host.py"][1], validate=True)
files["GPU_RESOURCE_CLAIM.json"] = (task["resource_claim_SHA256"], base64.b64encode(resource).decode())

request = uuid.uuid4().hex
nonce = uuid.uuid4().hex
payload = (
    f"CLAIM={task['resource_claim_SHA256']!r}\nRUN={claim['run_id']!r}\n"
    f"REQUEST={request!r}\nCOMMIT={commit!r}\nFILES={files!r}\n"
).encode() + controller
wire = json.dumps(dict(nonce=nonce, code=base64.b64encode(payload).decode(),
                       sha256=hashlib.sha256(payload).hexdigest())).encode() + b"\n"
bootstrap = r"""
import base64, hashlib, json, sys
nonce=NONCE
print(json.dumps(dict(stage='READY', nonce=nonce)),flush=True)
line=sys.stdin.buffer.readline(1048577)
assert len(line)<=1048576 and line.endswith(b'\n')
wire=json.loads(line)
assert wire['nonce']==nonce
payload=base64.b64decode(wire['code'],validate=True)
assert hashlib.sha256(payload).hexdigest()==wire['sha256']
exec(compile(payload,'published-glm-model-transport-dispatch','exec'), {'__name__':'__main__'})
""".replace("NONCE", repr(nonce))
intent = dict(commit=commit, run_id=claim["run_id"], request_id=request, wire_sha256=hashlib.sha256(wire).hexdigest(),
              replay_allowed=False)
assert not (run / "mutating-intent.json").exists()
command = "/usr/bin/timeout --kill-after=7s 280s /usr/local/python3.12/bin/python3 -B -u -c " + shlex.quote(bootstrap)
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
            stdout, _ = proc.communicate(wire, timeout=285)
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
        assert outcome["request_id"] == request and outcome["claim_commit"] == commit and outcome["run_id"] == claim["run_id"]
    except (ValueError, KeyError, AssertionError):
        outcome = None
summary = dict(**intent, payload_dispatch_attempted=started, returncode=rc, error=error_type,
               private_stdout_sha256=hashlib.sha256(stdout).hexdigest(),
               remote_outcome=("RETURNED" if outcome else "UNKNOWN" if started else "NO_PAYLOAD_SENT"))
if outcome:
    assert outcome["stage"] == "LAUNCHER_STARTED"
    summary.update(launcher_stage=outcome["stage"], launcher_PID=outcome["PID"],
                   launcher_startticks=outcome["startticks"], epoch=outcome["epoch"])

save(attempt + "-outcome.json", summary)
print(json.dumps(summary))
