"""Independently rehash exact MiniCPM inputs after the original stream completes."""
import hashlib,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
pins=json.loads((BASE/"MODEL_FILES.json").read_bytes())["files"]
prelude=(BASE.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf8")
ns={"__file__":str(Path(__file__).resolve())}
exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),ns)
REMOTE=r"""
import datetime,hashlib,json,signal,time
from pathlib import Path
signal.alarm(180)
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-original-model-v1")
receipt_raw=(root/"receipt.json").read_bytes();receipt=json.loads(receipt_raw)
assert receipt["state"]=="VERIFIED_ORIGINAL_MODEL_FILES" and receipt["files"]==PINS
model=root/"MiniCPM-V-4.6"
assert sorted(p.name for p in model.iterdir())==sorted(PINS)
start=time.monotonic();rows={}
for name,item in PINS.items():
    p=model/name
    assert p.is_file() and not p.is_symlink() and p.stat().st_size==item["bytes"]
    h=hashlib.sha256()
    with p.open("rb") as f:
        for raw in iter(lambda:f.read(1024*1024),b""):h.update(raw)
    assert h.hexdigest()==item["SHA256"],name
    rows[name]=item
print(json.dumps(dict(state="INDEPENDENT_MODEL_BYTES_VERIFIED",claim_commit=receipt["claim_commit"],
    observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=rows,
    bytes=sum(x["bytes"] for x in rows.values()),original_receipt_SHA256=hashlib.sha256(receipt_raw).hexdigest(),
    independent_hash_seconds=time.monotonic()-start,model_directory=str(model),
    model_imported=False,GPU_execution=False,training_numerics_qualified=False)))
"""
reply=subprocess.run(ns["ssh"],input=("PINS="+repr(pins)+"\n"+REMOTE).encode(),capture_output=True,timeout=195)
with (BASE/"independent-read.stderr").open("xb") as f:f.write(reply.stderr)
assert reply.returncode==0,f"rc={reply.returncode}"
value=json.loads(reply.stdout)
with (BASE/"independent-original-model.json").open("x") as f:json.dump(value,f,indent=2)
print(json.dumps({k:v for k,v in value.items() if k!="files"}))
