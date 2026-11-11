"""Independently verify the exact terminal forty-node result and its node files."""

import json
from pathlib import Path
import subprocess

base=Path(__file__).resolve().parent
prelude=(base.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf-8")
namespace={"__file__":str(Path(__file__).resolve())}
exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),namespace)
remote="""
import hashlib,json,subprocess
from pathlib import Path
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training40-resource-ledger-v2")
raw=(root/"outcome.json").read_bytes()
assert len(raw)<200000
value=json.loads(raw)
assert value["claim_commit"]=="3318f258d37a103dae2645135ada50aa3d7ce3c8"
rows=subprocess.check_output(["sacct","-n","-P","-j","1087656.112","--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw"],
                             text=True,timeout=12)
accounting=[line.split("|") for line in rows.splitlines() if line.startswith("1087656.112|")]
assert accounting==value["accounting"]==[["1087656.112","training40-resource-ledger-v2","COMPLETED","0:0","0"]]
assert value["returncode"]==0
nodes={row["node"] for row in value["snapshots"]}
assert len(nodes)==40 and nodes==set(value["snapshot_files"])
for name,expected in value["snapshot_files"].items():
    data=(root/"nodes"/(name+".json")).read_bytes()
    assert len(data)==expected["bytes"] and hashlib.sha256(data).hexdigest()==expected["SHA256"]
    assert [row for row in value["snapshots"] if row["node"]==name]==[json.loads(data)]
assert len({(row["node"],card["pci"]) for row in value["snapshots"] for card in row["cards"]})==320
print(json.dumps({"original_outcome_SHA256":hashlib.sha256(raw).hexdigest(),"value":value,
                  "independent_exact_node_files_verified":40}))
"""
reply=subprocess.run(namespace["ssh"],input=remote.encode(),capture_output=True,timeout=40)
rows=[json.loads(line) for line in reply.stdout.splitlines() if line.startswith(b"{")]
assert reply.returncode==0 and len(rows)==1, f"rc={reply.returncode},rows={len(rows)}"
value=rows[0]
with (base/"terminal-v2-independent.json").open("x",encoding="utf-8") as stream:
    json.dump(value,stream,indent=2)
print(json.dumps({k:v for k,v in value.items() if k!="value"}))
