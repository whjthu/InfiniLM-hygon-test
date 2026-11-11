"""Independently inspect the exact staged public files and original offline CPU step."""

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
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
stage=root/"training-vlm-swift-public-artifacts-v1"
work=root/"training-vlm-swift-offline-runtime-v1"
def get(path):
    raw=path.read_bytes()
    assert len(raw)<300000
    return {"SHA256":hashlib.sha256(raw).hexdigest(),"value":json.loads(raw)}
public=get(stage/"receipt.json")
assert public["value"]["claim_commit"]=="8d7f5d8f46568569da9873ab8654150a1ad6f99f"
for name,row in public["value"]["files"].items():
    path=stage/"packages"/name
    assert path.stat().st_size==row["bytes"]
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest()==row["SHA256"]
answer={"public_receipt":public,"public_files_rehashed":len(public["value"]["files"]),"work_exists":work.exists()}
if (work/"outcome.json").is_file():
    original=get(work/"outcome.json")
    value=original["value"]
    assert value["claim_commit"]=="8d7f5d8f46568569da9873ab8654150a1ad6f99f"
    rows=subprocess.check_output(["sacct","-n","-P","-j","1087656","--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw"],
                                 text=True,timeout=12)
    accounting=[line.split("|") for line in rows.splitlines() if "|training-vlm-swift-offline-runtime-v1|" in line]
    assert len(accounting)==1 and accounting==value["accounting"]
    assert accounting[0][2] in ("COMPLETED","FAILED","TIMEOUT","CANCELLED","OUT_OF_MEMORY","NODE_FAIL")
    summary=get(work/"summary.json")
    assert summary["SHA256"]==value["resolution"]["SHA256"] and summary["value"]==value["resolution"]["value"]
    if (work/"inventory.json").is_file():
        inventory=get(work/"inventory.json")
        assert inventory["SHA256"]==summary["value"]["inventory_SHA256"]
        assert inventory["value"]==summary["value"]["inventory"]
    answer.update(original_outcome=original,original_summary=summary,accounting=accounting)
print(json.dumps(answer))
"""
reply=subprocess.run(namespace["ssh"],input=remote.encode(),capture_output=True,timeout=110)
rows=[json.loads(line) for line in reply.stdout.splitlines() if line.startswith(b"{")]
if reply.returncode or len(rows)!=1:
    (base/"offline-reader-failure.stderr").write_bytes(reply.stderr)
    print(json.dumps({"remote_traceback":[line[-200:] for line in reply.stderr.decode(errors="replace").splitlines()
                      if line.startswith(("  File ","AssertionError","KeyError","TypeError","RuntimeError"))][-5:]}))
assert reply.returncode==0 and len(rows)==1, f"rc={reply.returncode},rows={len(rows)}"
value=rows[0]
filename="offline-terminal-independent.json" if "original_outcome" in value else "public-stage-independent.json"
with (base/filename).open("x",encoding="utf-8") as stream:
    json.dump(value,stream,indent=2)
print(json.dumps({"public_files_rehashed":value["public_files_rehashed"],"work_exists":value["work_exists"],
                  "accounting":value.get("accounting"),"public_receipt_SHA256":value["public_receipt"]["SHA256"],
                  "original_outcome_SHA256":value.get("original_outcome",{}).get("SHA256")}))
