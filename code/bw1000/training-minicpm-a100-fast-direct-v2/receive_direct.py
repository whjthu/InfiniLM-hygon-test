"""Stage one published direct-copy controller in the reconciled original directory."""
import base64,hashlib,json,subprocess
from pathlib import Path
ROOT=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-fast-overlay-v1")
assert sorted(p.name for p in ROOT.iterdir())==["intent.json"]
assert hashlib.sha256((ROOT/"intent.json").read_bytes()).hexdigest()=="b65ea7b133b49db3ed38709de5c6c3a8ac1401bd92782e284139b0445cb42c2c"
CONTROL=ROOT/"direct-v2";CONTROL.mkdir(mode=0o700)
src=CONTROL/"src";src.mkdir(mode=0o700)
pins={}
for name,item in FILES.items():
    assert name in ("OVERLAY_FILES.json","export_overlay.py","verify_overlay.py","direct_copy.py")
    raw=base64.b64decode(item["b64"],validate=True);assert hashlib.sha256(raw).hexdigest()==item["SHA256"]
    with (src/name).open("xb") as f:f.write(raw)
    pins[name]=item["SHA256"]
with (CONTROL/"installed.json").open("x") as f:json.dump({"claim_commit":CLAIM,"files":pins},f)
reply=subprocess.run(["python3","-B","-u",str(src/"direct_copy.py")],capture_output=True,timeout=620)
with (CONTROL/"controller.stdout").open("xb") as f:f.write(reply.stdout)
with (CONTROL/"controller.stderr").open("xb") as f:f.write(reply.stderr)
rows=[json.loads(x) for x in reply.stdout.splitlines() if x.startswith(b"{")]
value={"controller_rc":reply.returncode,"stdout_SHA256":hashlib.sha256(reply.stdout).hexdigest(),"stderr_SHA256":hashlib.sha256(reply.stderr).hexdigest()}
if len(rows)==1:value["receipt"]=rows[0]
print(json.dumps(value))
