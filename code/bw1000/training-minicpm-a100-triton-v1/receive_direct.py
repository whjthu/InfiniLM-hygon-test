"""Stage one published direct-copy controller in the reconciled original directory."""
import base64,hashlib,json,subprocess
from pathlib import Path
ROOT=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-fast-overlay-v1")
assert hashlib.sha256((ROOT/"direct-v2/receipt.json").read_bytes()).hexdigest()=="f15515aed67a4ab27879f7b32225c9aa59742f589b749c4360a17b8c0dca0ed6"
CONTROL=ROOT/"triton-v1";CONTROL.mkdir(mode=0o700)
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
