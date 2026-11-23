"""Dispatch one published 0-GRES range scan through the existing SCNet connection."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import shlex
import subprocess


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
PYTHON = "/usr/local/python3.12/bin/python3"
SSH_PREFIX_SHA = "e45d0ac4be713c0244c109b5fc76a1532cfbe5ea792c790775e19d59e002bcd7"
CLAIM_SHA = "a21d157131c58718eed60dff56a80c39b4eca790571e4909708117a9517eabd9"
REMOTE = r"""
from pathlib import Path
from hashlib import sha256
import base64,json,os,subprocess,sys,time
ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
REQUEST="6cddd9bdb8da491996902eeb4b394a70"
LABEL="Deep-CSA-16-range-pins-6cddd9bdb8da"
PARENT="1194900"
NODE="f12r3n17"
PRE="1194900.525"
SOURCE_SHA="6af479a6bd0351a1ab9b0c5defb80acce61dff329b3ca986e33205614e72d212"
META=ROOT/"results/Deep-CSA-layer2-23-weight-metadata-v1.json"
STAGE=ROOT/"incoming"/LABEL
OUT=ROOT/"results"/(LABEL+".json")
CHECKPOINT="/public/home/aclsylqidf/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8"
C=["JobIDRaw","JobName%100","State","ExitCode","ElapsedRaw","NodeList","AllocTRES"]

def query(*argv):
    q=subprocess.run(argv,capture_output=True,text=True,timeout=8,check=False)
    if q.returncode or len(q.stdout)>2*1024**2:raise ValueError("SCHEDULER_QUERY_FAILED")
    return q.stdout

def fields(line):
    return dict(x.split("=",1) for x in line.split() if "=" in x)

def exact_standby(rows):
    expected={"StepId":PRE,"Name":"training80-dcu-standby-3a23feb32d2f-f12r3n17","State":"RUNNING",
              "NodeList":NODE,"TRES":"cpu=8,gres/dcu=8,mem=16G,node=1"}
    matches=[row for row in rows if row.get("StepId")==PRE]
    if len(matches)!=1 or any(matches[0].get(k)!=v for k,v in expected.items()):
        raise ValueError("EXACT_PREVIOUS_STANDBY_REQUIRED")
    return expected

def current():
    return exact_standby(map(fields,query("scontrol","-o","show","step",PARENT).splitlines()))

def accounting():
    raw=query("sacct","-n","-P","-j",PARENT,"-o",",".join(C))
    values=[line.split("|") for line in raw.splitlines() if line]
    assert values and all(len(x)==len(C) for x in values)
    return [x for x in values if x[1]==LABEL]

def write(path,data):
    with path.open("xb") as handle:
        handle.write(data);handle.flush();os.fsync(handle.fileno())

def dump(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+"\n").encode()
def pin(value):return sha256(value).hexdigest()

def main():
    os.umask(0o077)
    source=base64.b64decode(sys.stdin.buffer.read(40000),validate=True)
    assert pin(source)==SOURCE_SHA
    compile(source,"read_ranges.py","exec")
    parent=fields(query("scontrol","-o","show","job",PARENT))
    assert parent["JobId"]==PARENT and parent["JobState"]=="RUNNING" and parent["UserId"].endswith("("+str(os.getuid())+")")
    assert NODE in query("scontrol","show","hostnames",parent["NodeList"]).splitlines()
    with META.open("rb") as metadata:
        meta=metadata.read(6072)
    assert len(meta)==6071 and pin(meta)=="39e2af65e0f30502d4a0d1f97df54fc27f0023f2ba1861810a634b1065a7e4f7"
    before=current()
    assert not accounting() and not os.path.lexists(STAGE) and not os.path.lexists(OUT)
    STAGE.mkdir(mode=0o700)
    write(STAGE/"read_ranges.py",source)
    argv=["srun","--jobid="+PARENT,"--job-name="+LABEL,"--exact","--overlap","--immediate=5","--nodes=1","--ntasks=1",
          "--nodelist="+NODE,"--cpus-per-task=1","--threads-per-core=1","--cpu-bind=threads","--mem=1024M",
          "--gres=none","--time=00:01:30","--kill-on-bad-exit=1",sys.executable,"-I","-S","-B","-u",str(STAGE/"read_ranges.py"),
          "--checkpoint",CHECKPOINT,"--metadata",str(META),"--output",str(OUT)]
    record={"request":REQUEST,"label":LABEL,"epoch":time.time(),"source_SHA256":SOURCE_SHA,
            "argv_SHA256":pin(dump(argv)),"worker_before":before,"replay_allowed":False}
    write(STAGE/"intent.json",dump(record))
    with (STAGE/"slurm.stdout").open("xb") as out,(STAGE/"slurm.stderr").open("xb") as err:
        q=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=64,check=False)
    until=time.monotonic()+10
    actual=accounting()
    while (not actual or actual[0][2]=="RUNNING") and time.monotonic()<until:
        time.sleep(.3);actual=accounting()
    record.update(actual_accounting_columns=C,actual_accounting=actual,worker_after=current(),srun_rc=q.returncode,
                  observed_epoch=time.time(),stderr_SHA256=pin((STAGE/"slurm.stderr").read_bytes()))
    if q.returncode==0:
        private=OUT.read_bytes()
        assert 0<len(private)<=16384
        detail=json.loads(private)
        assert detail["actual_value_bytes_read"]==157460736 and detail["indexer_payload_bytes_read"]==0
        assert len(detail["weights"])==16 and not any(k.startswith("indexer.") for k in detail["weights"])
        assert len(actual)==1 and actual[0][:4]==[detail["step"],LABEL,"COMPLETED","0:0"]
        assert actual[0][5]==NODE and actual[0][6] in ("cpu=1,mem=1G,node=1","cpu=1,mem=1024M,node=1")
        record.update(result=dict(receipt_SHA256=pin(private),receipt_bytes=len(private),partial_weights=16,
                                 actual_value_bytes_read=detail["actual_value_bytes_read"],indexer_payload_bytes_read=0,
                                 actual_GPU_calls=0,full_checkpoint_bound=False,elapsed_monotonic_s=detail["elapsed_monotonic_s"]))
    write(STAGE/"result.json",dump(record))
    print(dump(record).decode(),end="")

if __name__=="__main__":main()
"""


def digest(raw):
    return sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt_directory", type=Path)
    args = parser.parse_args()
    if Path.cwd() != WORKSPACE:
        raise ValueError("Run from the existing SCNet workspace")
    claim_raw = (HERE / "CLAIM.json").read_bytes()
    claim = json.loads(claim_raw)
    source = (HERE / "read_ranges.py").read_bytes()
    if digest(claim_raw) != CLAIM_SHA or digest(source) != claim["source_SHA256"]:
        raise ValueError("Published range scan differs")
    script = (WORKSPACE / "run-dcu-standby-cycle-20261123.ps1").read_text(encoding="utf8")
    prefix = script.split("@'\n", 1)[1].split("\nremote=", 1)[0]
    if digest(prefix.encode()) != SSH_PREFIX_SHA:
        raise ValueError("Existing transport source differs")
    namespace = {}
    exec(compile(prefix, "existing-pinned-SCNet-config", "exec"), namespace)
    out = args.receipt_directory
    out.mkdir()
    (out / "intent.json").write_text(json.dumps({"request": claim["request"], "replay_allowed": False}), encoding="utf8")
    command = shlex.join([PYTHON, "-I", "-S", "-B", "-c", REMOTE])
    try:
        q = subprocess.run(namespace["inner"] + [command], input=base64.b64encode(source),
                           capture_output=True, timeout=100, check=False)
    except subprocess.TimeoutExpired:
        print(json.dumps({"request": claim["request"], "remote_outcome": "UNKNOWN"}))
        return 2
    (out / "stdout.private").write_bytes(q.stdout)
    (out / "stderr.private").write_bytes(q.stderr)
    if q.returncode:
        print(json.dumps({"rc": q.returncode, "stderr_SHA256": digest(q.stderr), "remote_outcome": "UNKNOWN"}))
    else:
        value = json.loads(q.stdout)
        print(json.dumps(value | {"private_stdout_SHA256": digest(q.stdout)}, sort_keys=True))
    return q.returncode


if __name__ == "__main__":
    raise SystemExit(main())
