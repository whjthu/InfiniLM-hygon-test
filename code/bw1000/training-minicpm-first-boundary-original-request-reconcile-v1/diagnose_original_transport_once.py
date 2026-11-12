"""Bounded readonly SSH phase diagnosis; never starts or replays a job."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

D = Path(__file__).resolve().parent
B = D.parent
ns = {"__file__": str(D / "reuse.py")}
source = (B / "i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf8")
exec(compile(source.split("REMOTE=", 1)[0], "existing_cluster_SSH", "exec"), ns)
ssh = list(ns["ssh"])
assert "-i" in ssh and Path(ssh[ssh.index("-i") + 1]).name == "hygon-261111.txt"
assert not any("proxy" in x.lower() or "controlmaster" in x.lower() for x in ssh)
ssh.insert(1, "-v")
payload = b'''from pathlib import Path
import datetime,json,socket
w=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-native-first-boundary-pair-v1')
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),namespace_exists=w.exists(),intent_exists=(w/'intent.json').is_file(),owner_exists=(w/'keeper-owner.json').is_file(),new_execution_attempted=False)))
'''
folder = D / "transport-diagnosis"
folder.mkdir(exist_ok=True)
tag = str(time.time_ns())
start = time.monotonic()
r = subprocess.run(ssh, input=payload, capture_output=True, timeout=35)
for name, raw in (("stdout.private.json", r.stdout), ("stderr.private.txt", r.stderr)):
    with (folder / (tag + "." + name)).open("xb") as f:
        f.write(raw)
err = r.stderr.decode("utf8", "replace").lower()
markers = ["connection established", "remote protocol version", "ssh2_msg_kexinit sent", "ssh2_msg_kexinit received", "server host key", "ssh2_msg_newkeys received", "authentications that can continue", "offering public key", "server accepts key", "authenticated to", "sending command", "exit status", "connection closed", "permission denied", "connection refused", "connection reset", "timed out"]
result = dict(transport_rc=r.returncode, RTT_seconds=time.monotonic()-start,
              phase_markers={x: x in err for x in markers},
              stdout_bytes=len(r.stdout), stderr_bytes=len(r.stderr),
              stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),
              stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),
              new_execution_attempted=False, replay_allowed=False)
if r.returncode == 0:
    result["value"] = json.loads(r.stdout)
with (folder / (tag + "-receipt.json")).open("x") as f:
    json.dump(result, f, indent=2)
print(json.dumps(dict(receipt=tag+"-receipt.json", **result)))
