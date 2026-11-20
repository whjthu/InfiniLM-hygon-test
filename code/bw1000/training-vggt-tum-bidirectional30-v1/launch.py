"""Prepare one published task and use root's bounded ordinary-SSH transport.

No background controller, SSH service, fallback transport or polling is created.
The callback signature is transport(body_bytes, stdout_file, stderr_file, timeout).
"""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REQUEST = "ab8a312813524d949608defa7124c3bf"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(publication):
    epoch = time.time()
    mono_deadline = time.monotonic() + 1800
    def left(reserve=15):
        value = mono_deadline-time.monotonic()-reserve
        if value <= 0:
            raise TimeoutError("Original task epoch exhausted")
        return value
    spec=importlib.util.spec_from_file_location("vggt_bidirectional30_package_verifier",HERE/"verify_package.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify(HERE, adopted=True)
    git = ["git", "--no-optional-locks", "-c", "safe.directory="+REPO.as_posix(), "-C", str(REPO)]
    def read(*args):
        return subprocess.check_output(git+list(args), timeout=min(12,left()))
    if read("ls-remote","--exit-code","origin","refs/heads/bw1000-experiments").decode().split() != [publication,"refs/heads/bw1000-experiments"]:
        raise ValueError("Remote publication mismatch")
    manifest_raw = (HERE/"SOURCE_FILES.json").read_bytes()
    relative = HERE.relative_to(REPO).as_posix()
    if read("cat-file","blob",publication+":"+relative+"/SOURCE_FILES.json") != manifest_raw:
        raise ValueError("Published task manifest differs")
    for name,pin in json.loads(manifest_raw)["files"].items():
        raw = read("cat-file","blob",publication+":"+relative+"/"+name)
        if raw != (HERE/name).read_bytes() or digest(raw) != pin["SHA256"]:
            raise ValueError("Published member mismatch: "+name)
    claim_raw = (HERE/"CLAIM.json").read_bytes()
    claim = json.loads(claim_raw)
    if claim["request"] != REQUEST:
        raise ValueError("Fixed published request differs")
    payload = dict(publication=publication,request=REQUEST,epoch=epoch,deadline=epoch+1800,
                   source_manifest_SHA256=digest(manifest_raw),claim_SHA256=digest(claim_raw),files={})
    for name,pin in claim["sources"].items():
        raw = read("cat-file","blob",publication+":"+pin["path"])
        if digest(raw) != pin["sha"] or raw != (REPO/pin["path"]).read_bytes():
            raise ValueError("Task dependency differs: "+name)
        payload["files"]["src/"+name] = dict(sha=pin["sha"],bytes=len(raw),b64=base64.b64encode(raw).decode("ascii"))
    payload["files"]["src/task-claim.json"] = dict(sha=digest(claim_raw),bytes=len(claim_raw),
                                                b64=base64.b64encode(claim_raw).decode("ascii"))
    program=("PAYLOAD="+repr(payload)+"\n").encode("ascii")+(HERE/"run_tum_continuous_once.py").read_bytes()
    destination=REPO/"local-only"/(HERE.name+"-original-dispatch")
    destination.mkdir()
    intent=dict(request=payload["request"],publication=publication,epoch=epoch,deadline=epoch+1800,
                body_SHA256=digest(program),source_manifest_SHA256=digest(manifest_raw),
                claim_SHA256=digest(claim_raw),unknown_replay_allowed=False)
    (destination/"intent.json").write_text(json.dumps(intent,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return program,destination,intent,left


def main(publication, transport):
    body,destination,intent,left = prepare(publication)
    attempt=dict(request=intent["request"],started_epoch=time.time(),attempted=True,timeout_seconds=left(5))
    (destination/"attempt.json").write_text(json.dumps(attempt,sort_keys=True)+"\n",encoding="utf8")
    rc=None
    error_type=None
    try:
        with (destination/"stdout.private").open("xb") as out,(destination/"stderr.private").open("xb") as err:
            rc=transport(body,out,err,left(5)).returncode
    except Exception as exc:
        error_type=type(exc).__name__
    receipt=dict(request=intent["request"],publication=publication,transport_returncode=rc,
                 error_type=error_type,execution_state="UNKNOWN" if rc is None or rc==255 else "RETURNED",
                 remote_terminal_not_implied_by_transport=True,replay_allowed=False,
                 epoch=intent["epoch"],deadline=intent["deadline"])
    (destination/"transport.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return 1 if rc is None else rc
