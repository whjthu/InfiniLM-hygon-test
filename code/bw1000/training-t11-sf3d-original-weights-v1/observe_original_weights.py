"""Independently inspect only the intended original SF3D staging directory."""

import datetime
import hashlib
import json
from pathlib import Path
import subprocess

base=Path(__file__).resolve().parent
prelude=(base.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf-8")
namespace={"__file__":str(Path(__file__).resolve())}
exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),namespace)
remote="""
import hashlib,json,shutil
from pathlib import Path
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-t11-sf3d-original-weights-v1")
result={"exists":root.exists(),"free_bytes":shutil.disk_usage(root.parent).free}
for name in ("intent.json","receipt.json","failure.json"):
    path=root/name
    if path.is_file():
        raw=path.read_bytes()
        assert 0<len(raw)<8192
        result[name]={"SHA256":hashlib.sha256(raw).hexdigest(),"value":json.loads(raw)}
if "receipt.json" in result:
    for name,row in result["receipt.json"]["value"]["files"].items():
        assert name in ("model.safetensors","config.yaml")
        path=root/name
        assert path.stat().st_size==row["bytes"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream,"sha256").hexdigest()==row["SHA256"]
    result["independent_file_hashes_match"]=True
print(json.dumps(result))
"""
reply=subprocess.run(namespace["ssh"],input=remote.encode(),capture_output=True,timeout=110)
rows=[json.loads(line) for line in reply.stdout.splitlines() if line.startswith(b"{")]
assert reply.returncode==0 and len(rows)==1, f"rc={reply.returncode}, rows={len(rows)}"
result={"checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),**rows[0]}
name="original-weight-terminal-independent.json" if "receipt.json" in result else "original-weight-preflight.json"
with (base/name).open("x",encoding="utf-8") as f:
    json.dump(result,f,indent=2)
print(json.dumps(result))
