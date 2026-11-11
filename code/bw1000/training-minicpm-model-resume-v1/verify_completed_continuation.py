"""Independently verify completed continuation and preserve the original failed attempt."""
import hashlib,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
pins=json.loads((BASE/"MODEL_FILES.json").read_bytes())["files"]
intent=json.loads((BASE/"continuation-intent.json").read_bytes())
ns={"__file__":str(Path(__file__).resolve())}
prelude=(BASE.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf8")
exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),ns)
code=r"""
import datetime,hashlib,json
from pathlib import Path
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-original-model-v1")
raw=(root/"resume-v1-receipt.json").read_bytes();receipt=json.loads(raw)
assert receipt["claim_commit"]==INTENT["claim_commit"] and receipt["state"]=="CONTINUED_ORIGINAL_MODEL_FILES_VERIFIED"
assert receipt["epoch"]==INTENT["epoch"] and receipt["deadline"]==INTENT["deadline"]
assert receipt["deadline"]-receipt["epoch"]==10800 and receipt["files"]==PINS
assert hashlib.sha256((root/"failure.json").read_bytes()).hexdigest()=="bc3442cd990d713dfd62951eb41cba50509c480e2f082ce80d647795cbf3d3c3"
model=root/"MiniCPM-V-4.6"
assert sorted(x.name for x in model.iterdir())==sorted(PINS)
for name,row in PINS.items():
    p=model/name
    assert p.is_file() and not p.is_symlink() and p.stat().st_size==row["bytes"]
    h=hashlib.sha256()
    with p.open("rb") as f:
        for data in iter(lambda:f.read(1024*1024),b""):h.update(data)
    assert h.hexdigest()==row["SHA256"],name
print(json.dumps({"state":"INDEPENDENT_CONTINUED_ORIGINAL_MODEL_VERIFIED","observed_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "claim_commit":receipt["claim_commit"],"receipt_SHA256":hashlib.sha256(raw).hexdigest(),"files":PINS,
    "prefix_bytes":receipt["copy"]["retained_prefix_bytes"],"continued_bytes":receipt["copy"]["continued_bytes"],
    "original_failure_preserved":True,"model_directory":str(model),"GPU_execution":False,"model_loaded":False}))
"""
r=subprocess.run(ns["ssh"],input=("PINS="+repr(pins)+"\nINTENT="+repr(intent)+"\n"+code).encode(),capture_output=True,timeout=180)
with (BASE/"independent-reader.stderr").open("xb") as f:f.write(r.stderr)
assert r.returncode==0,f"reader rc={r.returncode}"
value=json.loads(r.stdout)
with (BASE/"independent-completed-model.json").open("x") as f:json.dump(value,f,indent=2)
print(json.dumps({k:v for k,v in value.items() if k!="files"}))
