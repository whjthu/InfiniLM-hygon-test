"""Identify current KFD holders on the occupied T11 node using a CPU-only step."""

import json
from pathlib import Path
import subprocess

PROBE = r'''
import hashlib,json,os
from pathlib import Path
import socket

assert socket.gethostname().split(".")[0]=="f11r2n15"
rows=[]
for entry in Path("/sys/class/kfd/kfd/proc").iterdir():
    assert entry.name.isdecimal()
    p=Path("/proc")/entry.name
    row={"pid":int(entry.name),"proc_visible":p.is_dir()}
    if p.is_dir():
        stat=(p/"stat").read_text()
        tail=stat[stat.rfind(")")+2:].split()
        row.update(state=tail[0],start_ticks=tail[19],owner_uid=p.stat().st_uid,
                   same_uid=p.stat().st_uid==os.getuid(),process_name=(p/"comm").read_text().strip())
        try:
            env=dict(line.split(b"=",1) for line in (p/"environ").read_bytes().split(b"\0") if b"=" in line)
            row["slurm"]={key:env[key.encode()].decode() for key in ("SLURM_JOB_ID","SLURM_STEP_ID") if key.encode() in env}
            command=(p/"cmdline").read_bytes()
            row["command_SHA256"]=hashlib.sha256(command).hexdigest()
            project=b"/public/home/aclsylqidf/hygon-inference-refactor-20261014"
            row["command_in_project"]=project in command
            row["cwd_in_project"]=str((p/"cwd").resolve()).startswith(project.decode()+"/")
        except PermissionError:
            row["process_details_restricted"]=True
    rows.append(row)
print(json.dumps({"node":"f11r2n15","probe_step":"1087656."+os.environ["SLURM_STEP_ID"],"holders":rows}),flush=True)
'''
base=Path(__file__).resolve().parent
prelude=(base.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf-8")
namespace={"__file__":str(Path(__file__).resolve())}
exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),namespace)
remote=r'''
import hashlib,json,subprocess,time
from pathlib import Path
label="training40-occupied-t11-owner-v1"
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training40-resource-ledger-v2")
def accounting():
    rows=subprocess.check_output(["sacct","-n","-P","-j","1087656","--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw"],
                                  text=True,timeout=12)
    return [line.split("|") for line in rows.splitlines() if "|"+label+"|" in line]
assert not accounting()
assert not (root/"occupied-owner-probe.log").exists()
with (root/"occupied-owner-probe.log").open("xb") as log:
    result=subprocess.run(["srun","--jobid=1087656","--job-name="+label,"--exact","--nodes=1","--ntasks=1",
                           "--nodelist=f11r2n15","--cpus-per-task=1","--mem=1024M","--gres=none",
                           "--immediate=5","--time=00:01:00","/usr/local/python3.12/bin/python3","-B","-c",CODE],
                          stdout=log,stderr=subprocess.STDOUT,timeout=70)
rows=[]
for _ in range(20):
    rows=accounting()
    if len(rows)==1 and rows[0][2] in ("COMPLETED","FAILED","TIMEOUT","CANCELLED"):
        break
    time.sleep(.5)
raw=(root/"occupied-owner-probe.log").read_bytes()
assert len(raw)<8192
data=[json.loads(line) for line in raw.splitlines() if line.startswith(b"{")]
print(json.dumps({"rc":result.returncode,"accounting":rows,"record":data[0] if len(data)==1 else None,
                  "log_SHA256":hashlib.sha256(raw).hexdigest()}))
'''
reply=subprocess.run(namespace["ssh"],input=f"CODE={PROBE!r}\n".encode()+remote.encode(),capture_output=True,timeout=95)
rows=[json.loads(line) for line in reply.stdout.splitlines() if line.startswith(b"{")]
assert reply.returncode==0 and len(rows)==1, f"rc={reply.returncode},rows={len(rows)}"
with (base/"occupied-t11-owner-original.json").open("x",encoding="utf-8") as stream:
    json.dump(rows[0],stream,indent=2)
print(json.dumps(rows[0]))
