"""Receive pinned small training inputs, then rewrite only image path prefixes."""
import datetime,hashlib,io,json,sys,tarfile
from pathlib import Path,PurePosixPath
ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-vlm-original-assets-v1")
assert not ROOT.exists(),"destination already exists; reconcile original receipt instead of replay"
ROOT.mkdir(mode=0o700)
(ROOT/"intent.json").write_text(json.dumps({"claim_commit":CLAIM,"started_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scope":"COCO64 and plugins, no weights/GPU"}))
expected={x["name"]:x for x in PLAN["files"]}
assert len(expected)==67
archive_raw=sys.stdin.buffer.read(12000001)
assert len(archive_raw)<=12000000
assert hashlib.sha256(archive_raw).hexdigest()==ARCHIVE_SHA
seen=set()
with tarfile.open(fileobj=io.BytesIO(archive_raw),mode="r:") as archive:
    for member in archive:
        assert member.isfile() and member.name in expected and member.name not in seen
        item=expected[member.name]
        assert member.size==item["bytes"]
        raw=archive.extractfile(member).read()
        assert hashlib.sha256(raw).hexdigest()==item["SHA256"]
        path=ROOT/member.name
        assert not PurePosixPath(member.name).is_absolute() and ".." not in PurePosixPath(member.name).parts
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as f:f.write(raw)
        seen.add(member.name)
assert seen==set(expected)
rows=[json.loads(x) for x in (ROOT/"original.cuda.jsonl").read_text().splitlines()]
assert len(rows)==64
canonical=[]
roles=[]
for row in rows:
    image_roles=[PurePosixPath(x).relative_to("/mnt/data/nvidia-baseline-20261010/inputs/datasets").as_posix() for x in row["images"]]
    canonical.append({**row,"images":image_roles})
    roles.extend(image_roles)
assert roles==PLAN["ordered_image_roles"]
canonical_raw=(json.dumps(canonical,sort_keys=True,ensure_ascii=False,separators=(",",":"))+"\n").encode()
assert hashlib.sha256(canonical_raw).hexdigest()==PLAN["canonical_rows_SHA256"]
rewritten=[{**row,"images":[str(ROOT/"datasets"/role) for role in canon["images"]]} for row,canon in zip(rows,canonical)]
output=("".join(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n" for row in rewritten)).encode()
with (ROOT/"vlm_report_coco64.hip.jsonl").open("xb") as f:f.write(output)
result={"claim_commit":CLAIM,"completed_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files":len(seen),"source_bytes":sum(x["bytes"] for x in expected.values()),"archive_SHA256":ARCHIVE_SHA,
        "canonical_rows_SHA256":PLAN["canonical_rows_SHA256"],"images":len(roles),
        "rewritten_dataset":{"path":str(ROOT/"vlm_report_coco64.hip.jsonl"),"bytes":len(output),"SHA256":hashlib.sha256(output).hexdigest()},
        "files_SHA256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(seen)},
        "GPU_execution":False,"weights_read":False,"training_qualified":False}
with (ROOT/"receipt.json").open("x") as f:json.dump(result,f,indent=2)
print(json.dumps(result))
