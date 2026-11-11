"""Independently read the original staging receipt and rehash exact inputs."""
import hashlib,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
plan=json.loads((BASE/"TRANSFER_PLAN.json").read_bytes())
prelude=(BASE.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf8")
ns={"__file__":str(Path(__file__).resolve())}
exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),ns)
REMOTE=r"""
import datetime,hashlib,json
from pathlib import Path,PurePosixPath
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-vlm-original-assets-v1")
receipt_raw=(root/"receipt.json").read_bytes()
receipt=json.loads(receipt_raw)
assert receipt["claim_commit"]=="bb516defd368461003d4740a7fd63004f5686ff5"
verified=[]
for item in PLAN["files"]:
    p=root/item["name"]
    assert p.is_file() and not p.is_symlink()
    raw=p.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    assert len(raw)==item["bytes"] and digest==item["SHA256"]==receipt["files_SHA256"][item["name"]]
    verified.append(dict(name=item["name"],bytes=len(raw),SHA256=digest))
original=[json.loads(x) for x in (root/"original.cuda.jsonl").read_text().splitlines()]
target=(root/"vlm_report_coco64.hip.jsonl").read_bytes()
rewritten=[json.loads(x) for x in target.splitlines()]
assert len(original)==len(rewritten)==64
canonical=[]
roles=[]
for source,dest in zip(original,rewritten):
    a={k:v for k,v in source.items() if k!="images"}
    b={k:v for k,v in dest.items() if k!="images"}
    assert a==b
    image_roles=[PurePosixPath(x).relative_to("/mnt/data/nvidia-baseline-20261010/inputs/datasets").as_posix() for x in source["images"]]
    assert dest["images"]==[str(root/"datasets"/role) for role in image_roles]
    canonical.append({**source,"images":image_roles})
    roles.extend(image_roles)
assert roles==PLAN["ordered_image_roles"]
canonical_sha=hashlib.sha256((json.dumps(canonical,sort_keys=True,ensure_ascii=False,separators=(",",":"))+"\n").encode()).hexdigest()
assert canonical_sha==PLAN["canonical_rows_SHA256"]==receipt["canonical_rows_SHA256"]
assert hashlib.sha256(target).hexdigest()==receipt["rewritten_dataset"]["SHA256"]
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    claim_commit=receipt["claim_commit"],original_receipt_SHA256=hashlib.sha256(receipt_raw).hexdigest(),
    files_verified=len(verified),bytes_verified=sum(x["bytes"] for x in verified),image_count=len(roles),row_count=len(original),
    canonical_rows_SHA256=canonical_sha,nonimage_fields_preserved=True,image_paths_exact=True,
    rewritten_dataset=receipt["rewritten_dataset"],plugins=[x for x in verified if x["name"].startswith("plugins/")],
    model_weights_copied=False,training_executed=False,GPU_qualified=False)))
"""
reply=subprocess.run(ns["ssh"],input=("PLAN="+repr(plan)+"\n"+REMOTE).encode(),capture_output=True,timeout=45)
with (BASE/"independent-read.stderr").open("xb") as f:f.write(reply.stderr)
assert reply.returncode==0,f"independent reader rc={reply.returncode}"
value=json.loads(reply.stdout)
with (BASE/"independent-input-verification.json").open("x") as f:json.dump(value,f,indent=2)
print(json.dumps(value))
