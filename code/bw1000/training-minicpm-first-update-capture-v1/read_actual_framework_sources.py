"""Bind the observer to actual original framework source identities without GPU imports."""
import hashlib,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
code=r"""
import datetime,hashlib,importlib.metadata as m,json
from pathlib import Path
out={"observed_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"versions":{},"sources":{}}
for package,roles in {
    "transformers":["transformers/trainer.py"],
    "ms-swift":["swift/trainers/mixin.py","swift/callbacks/base.py","swift/model/model_arch.py"]
}.items():
    d=m.distribution(package);out["versions"][package]=d.version
    for role in roles:
        p=Path(d.locate_file(role));raw=p.read_bytes()
        out["sources"][role]={"path":str(p),"bytes":len(raw),"SHA256":hashlib.sha256(raw).hexdigest()}
print(json.dumps(out))
"""
r=subprocess.run(["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),"a100-27","/home/whj/nvidia-baseline/env-vlm-fast/bin/python -B -"],input=code.encode(),capture_output=True,timeout=30)
assert r.returncode==0
v=json.loads(r.stdout)
with (BASE/"actual-framework-sources.json").open("x") as f:json.dump(v,f,indent=2)
proof=json.loads((BASE/"SOURCE_PROOF.json").read_bytes())
expected={**proof["pinned_public_Swift_sources"],"transformers/trainer.py":proof["pinned_public_Trainer_source_SHA256"]}
exact={name:v["sources"][name]["SHA256"]==pin for name,pin in expected.items()}
print(json.dumps({"versions":v["versions"],"source_SHA_matches":exact}))
