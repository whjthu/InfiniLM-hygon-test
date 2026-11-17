"""Run the published FFN package's zero-DCU target-container check once."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import uuid

repo = Path.cwd()
root = Path(sys.argv[2]).resolve()
helper = root.parent / "glm-paired-readonly-connection.py"
assert hashlib.sha256(helper.read_bytes()).hexdigest() == "3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1"
assert len(sys.argv) == 4 and sys.argv[1] in {"--verify", "--run"}
commit = sys.argv[3]
assert len(commit)==40 and all(c in "0123456789abcdef" for c in commit)
git = ["git", "-c", f"safe.directory={repo.as_posix()}", "--no-optional-locks"]

def show(name):
    return subprocess.check_output(git + ["show", f"{commit}:{name}"], timeout=10)

def pin(data):
    return hashlib.sha256(data).hexdigest()

task = json.loads(show("coordination/tasks/GLM474-attention-detail-selector-host-package-v1.json"))
package = task["package"]
run = task["run_id"]
label = f"GLM474-attention-detail-{run}-preflight"
out = root / "target-preflight"
out.mkdir(exist_ok=True)
assert not (out / "intent.json").exists()
inventory = show(package + "/PACKAGE_FILES.json")
assert pin(inventory) == task["package_manifest_SHA256"]
files = {}
for name, digest in json.loads(inventory).items():
    data = show(package + "/" + name)
    assert pin(data) == digest
    files[name] = (digest, base64.b64encode(data).decode())
original = "code/bw1000/GLM-C1-layer-observer-r474/full-model-transport-target-preflight-v2/"
preflight = show(original + "preflight.py")
controller = show(original + "receive_once.py")
assert pin(preflight) == "b5c289bc16e60cf2e10d02d8fda99f4c5094c0aef03073d011c10af29f81829f"
assert pin(controller) == "70ece9535d33ad3c6c6b52477dbbd0c05b099e8b640507742d8597a705c192c2"
preflight = preflight.replace(b"arms=2", b"arms=1")
files["preflight.py"] = pin(preflight), base64.b64encode(preflight).decode()
controller = controller.replace(b'LABEL="GLM474-model-transport-target-preflight-v2"', b"LABEL=" + repr(label).encode())
assert controller.count(label.encode()) == 1
request = uuid.uuid4().hex
envelope = f"REQUEST={request!r}\nCOMMIT={commit!r}\nFILES={files!r}\n".encode() + controller
nonce = uuid.uuid4().hex
wire = json.dumps(dict(nonce=nonce, code=base64.b64encode(envelope).decode(), sha256=pin(envelope))).encode() + b"\n"
assert len(wire) <= 1048576
identity = dict(label=label, run_id=run, request_id=request, commit=commit,
                wire_SHA256=pin(wire), controller_SHA256=pin(controller), preflight_SHA256=pin(preflight))
bootstrap = f"""import base64, hashlib, json, sys
print({nonce!r},flush=True)
line=sys.stdin.buffer.readline(1048577)
assert line.endswith(b'\\n') and len(line)<=1048576
wire=json.loads(line)
assert wire['nonce']=={nonce!r}
payload=base64.b64decode(wire['code'],validate=True)
assert hashlib.sha256(payload).hexdigest()==wire['sha256']
exec(compile(payload,'published-glm-indexer-preflight','exec'),{{'__name__':'__main__'}})
"""
if sys.argv[1] == "--verify":
    compile(preflight,"preflight.py","exec")
    compile(controller,"controller.py","exec")
    compile(bootstrap,"bootstrap.py","exec")
    print(json.dumps(dict(label=label, pinned_files=len(files), wire_bytes=len(wire),
                          controller_SHA256=pin(controller), preflight_SHA256=pin(preflight))))
    raise SystemExit(0)
assert sys.argv[1] == "--run"
spec = importlib.util.spec_from_file_location("cluster_connection", helper)
connection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(connection)
attempt = out / request
attempt.mkdir()
remote = "/usr/bin/timeout --kill-after=7s 280s /usr/local/python3.12/bin/python3 -B -u -c " + shlex.quote(bootstrap)
started = False
error = None
stdout = b""
with (attempt / "stderr.private").open("xb") as err:
    with subprocess.Popen(connection.ssh+[remote], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err) as proc:
        ready = queue.Queue()
        threading.Thread(target=lambda: ready.put(proc.stdout.readline()), daemon=True).start()
        try:
            assert ready.get(timeout=18).strip() == nonce.encode()
            with (out / "intent.json").open("x", encoding="utf8") as stream:
                json.dump(identity, stream)
                stream.flush()
                os.fsync(stream.fileno())
            started = True
            stdout, _ = proc.communicate(wire, timeout=285)
        except Exception as exc:
            error = type(exc).__name__
            if proc.poll() is None:
                proc.kill()
            stdout, _ = proc.communicate(timeout=8)
        rc = proc.returncode
(attempt / "stdout.private").write_bytes(stdout)
answer = None
if started and rc == 0:
    try:
        answer = json.loads(stdout)
        assert answer["request_id"] == request and answer["commit"] == commit
    except (AssertionError, ValueError, KeyError):
        answer = None
summary = dict(identity, attempted=started, exit_code=rc, error=error, raw_SHA256=pin(stdout),
               remote_outcome="RETURNED" if answer else "UNKNOWN" if started else "NO_PAYLOAD_SENT")
if answer:
    summary.update(passed=answer["passed"], terminal=answer["terminal"], ranks=[
        {k:r[k] for k in ("node","pci_count","arms","requests","GPU_cards","Native_loaded","weight_bytes_read",
                           "host_manifest_SHA256","worker_manifest_SHA256","ports_bindable")}
        for r in answer["ranks"]], failure_classes=answer["log_error_classes"])
(attempt / "summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
print(json.dumps(summary))
raise SystemExit(0 if answer and answer["passed"] else 2)
