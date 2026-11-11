"""Independently verify original fast overlay software bytes and masked import evidence."""
import hashlib,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
plan=json.loads((BASE/"OVERLAY_FILES.json").read_bytes())
config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
code=r"""
import datetime,hashlib,json
from pathlib import Path
root=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-fast-overlay-v1")
receipt_raw=(root/"direct-v2/receipt.json").read_bytes();receipt=json.loads(receipt_raw)
assert receipt["claim_commit"]=="59aecdb5649b2381011222c20e46359e581f0cf0"
assert receipt["deadline"]-receipt["epoch"]==600
manifest=(root/"PLAN.json").read_bytes()
assert hashlib.sha256(manifest).hexdigest()==receipt["PLAN_SHA256"]
assert json.loads(manifest)==PLAN and receipt["files"]==len(PLAN["files"])
for name,item in PLAN["files"].items():
    p=root/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==item["bytes"]
    h=hashlib.sha256()
    with p.open("rb") as f:
        for raw in iter(lambda:f.read(1024*1024),b""):h.update(raw)
    assert h.hexdigest()==item["SHA256"],name
assert hashlib.sha256((root/"intent.json").read_bytes()).hexdigest()=="b65ea7b133b49db3ed38709de5c6c3a8ac1401bd92782e284139b0445cb42c2c"
assert receipt["source_rc"]==0
out={"observed_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"state":receipt["state"],
    "receipt_SHA256":hashlib.sha256(receipt_raw).hexdigest(),"files_rehashed":len(PLAN["files"]),
    "bytes_rehashed":sum(x["bytes"] for x in PLAN["files"].values()),"receipt":receipt}
if (root/"direct-v2/verification.json").is_file():
    raw=(root/"direct-v2/verification.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==receipt["verification"]["SHA256"]
    assert json.loads(raw)==receipt["verification"]["value"]
print(json.dumps(out))
"""
r=subprocess.run(["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),"a100-26","python3"],input=("PLAN="+repr(plan)+"\n"+code).encode(),capture_output=True,timeout=120)
with (BASE/"independent-reader.stderr").open("xb") as f:f.write(r.stderr)
assert r.returncode==0,f"rc={r.returncode}"
v=json.loads(r.stdout)
with (BASE/"independent-direct-overlay.json").open("x") as f:json.dump(v,f,indent=2)
print(json.dumps({k:x for k,x in v.items() if k!="receipt"}))
print(json.dumps(v["receipt"].get("verification",{})))
