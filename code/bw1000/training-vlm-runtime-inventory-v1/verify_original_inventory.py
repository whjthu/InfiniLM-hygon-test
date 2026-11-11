"""Independently consume the original VLM CPU runtime inventory."""

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
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-vlm-runtime-inventory-v1")
raw=(root/"outcome.json").read_bytes()
assert len(raw)<131072
value=json.loads(raw)
assert value["claim_commit"]=="7ea73f9eeb9b6d1c072d1dde281ba9cde1f33611"
rows=subprocess.check_output(["sacct","-n","-P","-j","1087656.114","--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw"],
                             text=True,timeout=12)
accounting=[line.split("|") for line in rows.splitlines() if line.startswith("1087656.114|")]
assert accounting==value["accounting"]==[["1087656.114","training-vlm-runtime-inventory-v1","COMPLETED","0:0","28"]]
assert value["returncode"]==0
data=(root/"inventory.json").read_bytes()
assert hashlib.sha256(data).hexdigest()==value["inventory"]["SHA256"]
assert json.loads(data)==value["inventory"]["value"]
assert value["inventory"]["value"]["runtime"]["pytorch_device_context_initialized"] is False
print(json.dumps({"original_outcome_SHA256":hashlib.sha256(raw).hexdigest(),
                  "original_inventory_SHA256":hashlib.sha256(data).hexdigest(),"value":value}))
"""
reply=subprocess.run(namespace["ssh"],input=remote.encode(),capture_output=True,timeout=40)
rows=[json.loads(line) for line in reply.stdout.splitlines() if line.startswith(b"{")]
assert reply.returncode==0 and len(rows)==1, f"rc={reply.returncode},rows={len(rows)}"
value=rows[0]
with (base/"terminal-independent.json").open("x",encoding="utf-8") as stream:
    json.dump(value,stream,indent=2)
print(json.dumps({k:v for k,v in value.items() if k!="value"}))
