"""Collect one finite 40-node CPU metadata step under the existing allocation."""

import base64
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training40-resource-ledger-v2")
label="training40-resource-ledger-v2"
assert not root.exists()

def query(*args):
    return subprocess.check_output(args,text=True,timeout=12).strip()

def accounting():
    return [line.split("|") for line in query("sacct","-n","-P","-j","1087656",
            "--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw").splitlines()
            if "|"+label+"|" in line]

parent=query("squeue","-h","-j","1087656","-o","%i|%T|%D|%L|%N").split("|")
assert parent[:3]==["1087656","RUNNING","40"]
nodes=query("scontrol","show","hostnames",parent[-1]).splitlines()
assert len(set(nodes))==40 and set(nodes)==set(EXPECTED_NODES) and not accounting()
for line in query("squeue","--steps","-h","-j","1087656","-o","%i").splitlines():
    assert line in ("1087656.batch","1087656.extern")
epoch=time.time()
deadline=epoch+180
root.mkdir(mode=0o700)
(root/"nodes").mkdir(mode=0o700)
raw=base64.b64decode(SOURCE["b64"],validate=True)
assert hashlib.sha256(raw).hexdigest()==SOURCE["SHA256"]
with (root/"snapshot_nodes_v2.py").open("xb") as stream:
    stream.write(raw)
with (root/"intent.json").open("x") as stream:
    json.dump({"claim_commit":CLAIM,"nodes":nodes,"epoch":epoch,"deadline":deadline,"GPU_work":False},stream)
args=["srun","--jobid=1087656","--job-name="+label,"--exact","--nodes=40","--ntasks=40",
      "--ntasks-per-node=1","--cpus-per-task=1","--mem=1024M","--gres=none",
      "--immediate=5","--time=00:02:00","/usr/local/python3.12/bin/python3","-B",str(root/"snapshot_nodes_v2.py"),str(root/"nodes")]
with (root/"snapshot.log").open("xb") as log:
    process=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()-5))
rows=[]
for _ in range(20):
    rows=accounting()
    if len(rows)==1 and rows[0][2] in ("COMPLETED","FAILED","TIMEOUT","CANCELLED","OUT_OF_MEMORY","NODE_FAIL"):
        break
    time.sleep(.5)
raw=(root/"snapshot.log").read_bytes()
assert len(raw)<200000
snapshots=[]
snapshot_files={}
if process.returncode==0:
    for node in nodes:
        path=root/"nodes"/(node+".json")
        data=path.read_bytes()
        assert 0<len(data)<8192
        value=json.loads(data)
        assert value["node"]==node
        snapshots.append(value)
        snapshot_files[node]={"bytes":len(data),"SHA256":hashlib.sha256(data).hexdigest()}
result={"claim_commit":CLAIM,"parent":parent[:4],"returncode":process.returncode,"accounting":rows,
        "source_SHA256":SOURCE["SHA256"],"epoch":epoch,"deadline":deadline,
        "wall_seconds":time.time()-epoch,"snapshots":snapshots,"snapshot_files":snapshot_files,"log_SHA256":hashlib.sha256(raw).hexdigest(),
        "errors":[line[-180:] for line in raw.decode(errors="replace").splitlines()
                  if line.startswith(("srun:","AssertionError","RuntimeError","PermissionError","FileNotFoundError"))][-8:]}
if process.returncode==0:
    assert len(rows)==1 and rows[0][2:4]==["COMPLETED","0:0"]
    assert len(snapshots)==40 and {row["node"] for row in snapshots}==set(nodes)
    assert {row["rank"] for row in snapshots}==set(range(40))
    assert all(row["step"]==rows[0][0] for row in snapshots)
    result["unique_node_PCI_pairs"]=len({(row["node"],card["pci"]) for row in snapshots for card in row["cards"]})
    assert result["unique_node_PCI_pairs"]==320
    result["all_KFD_holder_counts_zero"]=all(row["kfd_process_count"]==0 for row in snapshots)
with (root/"outcome.json").open("x") as stream:
    json.dump(result,stream,indent=2)
print(json.dumps(result))
