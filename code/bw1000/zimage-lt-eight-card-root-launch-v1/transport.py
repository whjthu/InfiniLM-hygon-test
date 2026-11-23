"""Use the existing pinned SCNet transport for one deployment or one O01 phase."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[4]
STAGE = ("/public/home/aclsylqidf/hygon-inference-refactor-20261014"
         "/incoming/O01-zimage-lt-eight-card-supervisor-v1-host-stage")
PYTHON = "/usr/local/python3.12/bin/python3"
PREFIX_SHA = "e45d0ac4be713c0244c109b5fc76a1532cfbe5ea792c790775e19d59e002bcd7"
REQUEST = "e23d24ea0c9c4d00af06154d82dc22e8"
HEX = re.compile("[0-9a-f]{64}")
INSTALLER = r"""
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys

stage = Path(sys.argv[1])
os.umask(0o077)
packet = json.loads(sys.stdin.buffer.read(524289))
manifest = base64.b64decode(packet['manifest'], validate=True)
if sha256(manifest).hexdigest() != sys.argv[2]:
    raise ValueError('manifest SHA differs')
plan = json.loads(manifest)
if plan['original_request'] != 'e23d24ea0c9c4d00af06154d82dc22e8':
    raise ValueError('original request differs')
files = {k: base64.b64decode(v, validate=True) for k, v in packet['files'].items()}
root = {'device_host.py','guarded-standby-bridge.py','host2105.json','host_run.py',
        'host_scope.py','node_entry.py','node_inventory.py','owner2108.json','pending.py','scheduler.py'}
member = {'CLAIM.json','RUNTIME_FILES.json','contract.py','linux_owner.py',
          'mask_preflight.py','standby_worker.py','supervisor.py','witness.py'}
if set(files) != set(plan['files']) or set(files) != root | {'runtime/' + value for value in member}:
    raise ValueError('exact stage membership differs')
for name, raw in files.items():
    if not 0 < len(raw) <= 65536 or {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()} != plan['files'][name]:
        raise ValueError('stage source pin differs')
    if name.endswith('.py'):
        compile(raw,name,'exec',dont_inherit=True)
stage.mkdir(mode=0o700)
(stage/'runtime').mkdir(mode=0o700)
for name, raw in {**files, 'HOST_STAGE_FILES.json':manifest}.items():
    with (stage/name).open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
for directory in (stage/'runtime', stage, stage.parent):
    fd=os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
print(json.dumps({'deployed': True, 'stage_members': len(files), 'manifest_SHA256': sys.argv[2]}))
"""
EXECUTOR = r"""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

stage=Path(sys.argv[1])
if stage.is_symlink():
    raise ValueError('unexpected HOST stage link')
manifest=(stage/'HOST_STAGE_FILES.json').read_bytes()
if len(manifest)>8192 or sha256(manifest).hexdigest()!=sys.argv[2]:
    raise ValueError('HOST stage manifest pin differs')
for name,item in json.loads(manifest)['files'].items():
    path=stage/name
    if path.is_symlink() or not path.is_file() or path.stat().st_size!=item['bytes']:
        raise ValueError('HOST stage source identity differs')
    if sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise ValueError('HOST stage source SHA differs')
interpreter=sys.executable
os.execv(interpreter,[interpreter,'-I','-S','-B',str(stage/'host_run.py'),*sys.argv[3:]])
"""


def ssh_args():
    raw = (ROOT / "run-dcu-standby-cycle-20261123.ps1").read_text(encoding="utf8")
    prefix = raw.split("@'\n", 1)[1].split("\nremote=", 1)[0]
    if sha256(prefix.encode()).hexdigest() != PREFIX_SHA:
        raise ValueError("existing authenticated transport setup changed")
    scope = {}
    exec(compile(prefix, "existing-pinned-transport", "exec"), scope)
    if Path.cwd() != ROOT:
        raise ValueError("transport must run from its existing workspace root")
    return scope["inner"]


def payload(path, expected):
    raw = (path / "HOST_STAGE_FILES.json").read_bytes()
    if len(raw) > 8192 or sha256(raw).hexdigest() != expected:
        raise ValueError("local stage manifest differs")
    manifest = json.loads(raw)
    if manifest["original_request"] != REQUEST:
        raise ValueError("different original request")
    files = {name: base64.b64encode((path / name).read_bytes()).decode() for name in manifest["files"]}
    result = json.dumps({"manifest": base64.b64encode(raw).decode(), "files": files}).encode()
    if len(result) > 524288:
        raise ValueError("stage packet exceeds byte bound")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("deploy", "observe", "reserve", "pre", "handoff", "gpu", "post", "grant"))
    parser.add_argument("--local-receipt", required=True, type=Path)
    parser.add_argument("--stage-sha", required=True)
    parser.add_argument("--rendered", type=Path)
    parser.add_argument("--reservation-SHA256")
    parser.add_argument("--boot-SHA256")
    parser.add_argument("--refill-request")
    args = parser.parse_args()
    if HEX.fullmatch(args.stage_sha) is None:
        raise ValueError("expected complete stage SHA")
    if args.phase == "deploy":
        body = payload(args.rendered, args.stage_sha)
        remote = [PYTHON, "-I", "-S", "-B", "-c", INSTALLER, STAGE, args.stage_sha]
    else:
        body = b""
        remote = [PYTHON, "-I", "-S", "-B", "-c", EXECUTOR, STAGE, args.stage_sha, args.phase]
        for name in ("reservation-SHA256", "boot-SHA256", "refill-request"):
            if value := getattr(args, name.replace("-", "_")):
                remote.extend(("--" + name, value))
    ssh = ssh_args()
    receipt = args.local_receipt
    receipt.mkdir()
    with (receipt / "intent.json").open("x", encoding="utf8") as intent:
        json.dump({"request": REQUEST, "phase": args.phase, "stage_SHA256": args.stage_sha,
                   "transport_replay_allowed": False}, intent, sort_keys=True)
    try:
        result = subprocess.run(ssh + [shlex.join(remote)], input=body, capture_output=True, timeout=94)
    except subprocess.TimeoutExpired as error:
        (receipt / "stdout.private").write_bytes(error.stdout or b"")
        (receipt / "stderr.private").write_bytes(error.stderr or b"")
        print(json.dumps({"phase": args.phase, "remote_outcome": "UNKNOWN"}))
        return 2
    (receipt / "stdout.private").write_bytes(result.stdout)
    (receipt / "stderr.private").write_bytes(result.stderr)
    if result.returncode:
        tail = result.stderr.decode("utf8", errors="replace").strip().splitlines()[-1:]
        # Print only known Python exception types and constant uppercase error labels.
        known = re.fullmatch(r"(ValueError|FileExistsError|BlockingIOError|KeyError): ([A-Z_0-9]+)",
                             tail[0]) if tail else None
        print(json.dumps({"phase": args.phase, "remote_outcome": "UNKNOWN", "rc": result.returncode,
                          "failure": known[0] if known else None, "stderr_SHA256": sha256(result.stderr).hexdigest()}))
        return result.returncode
    print(result.stdout.decode("utf8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
