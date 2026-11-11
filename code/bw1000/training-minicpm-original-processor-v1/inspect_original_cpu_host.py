import hashlib,json,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent
config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
out={}
code="import json;from pathlib import Path; print(json.dumps({p:Path(p).exists() for p in ['/home/whj/nvidia-baseline/env-vlm-fast/bin/python','/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-original-processor-v1']}))"
for host in ("a100-26","a100-27"):
    r=subprocess.run(["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),host,"python3"],input=code.encode(),capture_output=True,timeout=25)
    assert r.returncode==0
    out[host]=json.loads(r.stdout)
with (B/"original-environment-hosts.json").open("x") as f:json.dump(out,f,indent=2)
print(json.dumps(out))
