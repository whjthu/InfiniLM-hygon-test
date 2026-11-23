"""Send exactly one 0-GRES host-memory observer for the running .2179 worker."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import shlex
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BOOT_EVIDENCE = (ROOT / "InfiniLM-hygon-test/local-only"
                 / "O01-zimage-hybrid-pid-none-0eac6c1ec15f-original.json")
BOOT_EVIDENCE_SHA = "048b0e55ee2149651abc7d6a47d534727faa621124b9f2e7be868fcd6cb4ca4a"
SSH_PREFIX_SHA = "e45d0ac4be713c0244c109b5fc76a1532cfbe5ea792c790775e19d59e002bcd7"
PYTHON = "/usr/local/python3.12/bin/python3"
REQUEST = "d97b876aac6643c581a7ef130ea728c0"

REMOTE = r"""
import base64, json, os, re, subprocess, sys, time
from hashlib import sha256
from pathlib import Path

PARENT = "1103606"
TARGET = "1103606.2179"
NODE = "f12r2n11"
REQUEST = "d97b876aac6643c581a7ef130ea728c0"
LABEL = "O01-refill-memory-observe-" + REQUEST[:12]
PYTHON = sys.executable
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
OUT = ROOT / "results" / LABEL
ATTESTED = ROOT / "incoming/O01-zimage-lt-eight-card-supervisor-v1-host-stage/node_entry.py"
COLUMNS = ("JobIDRaw","JobName%100","State","ExitCode","ElapsedRaw","NodeList","AllocTRES")

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def run(*argv):
    result = subprocess.run(argv, capture_output=True, timeout=7, check=False)
    require(result.returncode == 0 and len(result.stdout) <= 2*1024**2, "SCHEDULER_QUERY_FAILED")
    return result.stdout.decode("utf8")

def fields(raw):
    return {k:v for k,v in (item.split("=",1) for item in raw.split() if "=" in item)}

def account():
    raw = run("sacct", "-n", "-P", "-j", PARENT, "-o", ",".join(COLUMNS))
    rows = [line.split("|") for line in raw.splitlines() if line]
    require(rows and all(len(row) == len(COLUMNS) for row in rows), "ACCOUNTING_SCHEMA")
    return [row for row in rows if row[1] == LABEL]

def active_worker():
    row = fields(run("scontrol","-o","show","step",TARGET))
    require(row.get("StepId")==TARGET and row.get("State")=="RUNNING" and row.get("NodeList")==NODE
            and row.get("Name")=="training80-dcu-standby-8bfbd56e71bf-f12r2n11"
            and "gres/dcu=8" in row.get("TRES","").split(","), "ACTUAL_REFILL_WORKER_REQUIRED")
    return {k:row[k] for k in ("StepId","Name","State","NodeList","TRES")}

def private_file(path, data):
    with path.open("xb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())

def encoded(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode()

os.umask(0o077)
source_sha, boot_sha = sys.argv[1:]
require(all(re.fullmatch("[a-f0-9]{64}", value) for value in (source_sha,boot_sha)), "EXACT_SOURCE_AND_BOOT_SHA")
source = base64.b64decode(sys.stdin.buffer.read(100000), validate=True)
require(0 < len(source) <= 65536 and sha256(source).hexdigest()==source_sha, "OBSERVER_SOURCE_PIN")
compile(source,"observer.py","exec")
parent=fields(run("scontrol","-o","show","job",PARENT))
require(parent.get("JobId")==PARENT and parent.get("JobState")=="RUNNING" and
        parent.get("UserId","").endswith("("+str(os.getuid())+")") and "gres/dcu=320" in parent.get("AllocTRES","").split(","),
        "OWNED_RUNNING_PARENT_REQUIRED")
require(NODE in run("scontrol","show","hostnames",parent["NodeList"]).splitlines(), "TARGET_NOT_IN_PARENT")
before=active_worker()
require(not account(), "OBSERVER_REQUEST_ALREADY_HAS_ACCOUNTING")
require(sha256(ATTESTED.read_bytes()).hexdigest()=="c7f13669918f720ea195cb49d0be843681d3155b025520caa8b1887babbd6ee6",
        "ATTESTED_SOURCE_DIFFERS")
OUT.mkdir(mode=0o700)
local_source=OUT/"observer.py"
private_file(local_source,source)
argv=["srun","--jobid="+PARENT,"--job-name="+LABEL,"--exact","--overlap","--immediate=5",
      "--nodes=1","--ntasks=1","--nodelist="+NODE,"--cpus-per-task=1","--threads-per-core=1","--cpu-bind=threads",
      "--gres=none","--mem=256M",
      "--time=00:01:00","--kill-on-bad-exit=1",PYTHON,"-I","-S","-B","-u",str(local_source),
      "--source-sha256",source_sha,"--attested-source",str(ATTESTED),"--boot-sha256",boot_sha,
      "--previous-worker-pid","982368","--previous-worker-startticks","356482943","--out",str(OUT/"observer.json")]
original={"request":REQUEST,"label":LABEL,"planned_GRES":0,"target_worker":before,
          "source_SHA256":source_sha,"argv_SHA256":sha256(encoded(argv)).hexdigest(),"replay_allowed":False}
private_file(OUT/"intent.json",encoded(original))
fd=os.open(OUT,os.O_RDONLY|os.O_DIRECTORY)
try: os.fsync(fd)
finally: os.close(fd)
with (OUT/"launcher.stdout").open("xb") as stdout, (OUT/"launcher.stderr").open("xb") as stderr:
    result=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,timeout=36,check=False)
deadline=time.monotonic()+8
observed=account()
while time.monotonic()<deadline and (not observed or observed[0][2]=="RUNNING"):
    time.sleep(0.3)
    observed=account()
receipt=dict(original, actual_accounting_columns=COLUMNS,actual_accounting=observed,
             srun_rc=result.returncode,worker_after=active_worker(),observed_epoch=time.time())
evidence=OUT/"observer.json"
if evidence.is_file() and not evidence.is_symlink() and 0<evidence.stat().st_size<=16384:
    raw=evidence.read_bytes()
    receipt.update(observer_SHA256=sha256(raw).hexdigest(),observer=json.loads(raw))
private_file(OUT/"result.json",encoded(receipt))
print(encoded(receipt).decode(),end="")
"""


def linked_boot():
    raw = BOOT_EVIDENCE.read_bytes()
    if sha256(raw).hexdigest() != BOOT_EVIDENCE_SHA:
        raise ValueError("Original target-boot evidence differs")
    result = json.loads(raw)["result"]
    owner = result["receipt"]["owner"]
    if result["srun_rc"] != 0 or (owner["step"], owner["node"]) != ("1103606.2108", "f12r2n11"):
        raise ValueError("Original target-boot identity differs")
    return owner["boot_SHA256"]


def ssh_args():
    if Path.cwd() != ROOT:
        raise ValueError("Use the existing authenticated workspace")
    ps = (ROOT / "run-dcu-standby-cycle-20261123.ps1").read_text(encoding="utf8")
    source = ps.split("@'\n", 1)[1].split("\nremote=", 1)[0]
    if sha256(source.encode()).hexdigest() != SSH_PREFIX_SHA:
        raise ValueError("Existing SSH source differs")
    built = {}
    exec(compile(source, "existing-pinned-transport", "exec"), built)
    return built["inner"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    args = parser.parse_args()
    data = (HERE / "observer.py").read_bytes()
    if sha256(data).hexdigest() != args.source_sha256:
        raise ValueError("Published observer source SHA differs")
    command = shlex.join([PYTHON, "-I", "-S", "-B", "-c", REMOTE, args.source_sha256, linked_boot()])
    ssh = ssh_args()
    args.receipt_dir.mkdir(mode=0o700)
    with (args.receipt_dir / "intent.json").open("x", encoding="utf8") as intent:
        json.dump({"request": REQUEST, "source_SHA256": args.source_sha256, "replay_allowed": False}, intent)
    try:
        remote = subprocess.run(ssh + [command], input=base64.b64encode(data), capture_output=True, timeout=70, check=False)
    except subprocess.TimeoutExpired:
        print(json.dumps({"request": REQUEST, "remote_outcome": "UNKNOWN", "transport_observation_timeout": True}))
        return 2
    (args.receipt_dir / "stdout.private").write_bytes(remote.stdout)
    (args.receipt_dir / "stderr.private").write_bytes(remote.stderr)
    if remote.returncode:
        print(json.dumps({"request": REQUEST, "remote_outcome": "UNKNOWN", "rc": remote.returncode,
                          "stderr_SHA256": sha256(remote.stderr).hexdigest()}))
    else:
        data = json.loads(remote.stdout)
        print(json.dumps({key: data.get(key) for key in
                          ("request", "actual_accounting", "srun_rc", "worker_after", "observer_SHA256")}))
    return remote.returncode


if __name__ == "__main__":
    raise SystemExit(main())
