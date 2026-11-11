"""Invoke the published direct internal software-copy attempt exactly once."""
import base64,hashlib,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;G=Path(sys.argv[1]);claim=sys.argv[2]
def blob(path):return subprocess.check_output(["git","-c","safe.directory="+G.as_posix(),"-C",str(G),"show",claim+":"+path])
task=json.loads(blob("coordination/tasks/training-minicpm-a100-fast-direct-v2.json"))
sources={}
for name,pin in task["source_SHA256"].items():
    raw=blob(task["source"]+"/"+name);assert hashlib.sha256(raw).hexdigest()==pin
    sources[name]=raw
files={n:{"b64":base64.b64encode(sources[n]).decode(),"SHA256":task["source_SHA256"][n]} for n in ("OVERLAY_FILES.json","export_overlay.py","verify_overlay.py","direct_copy.py")}
code=f"CLAIM={claim!r}\nFILES={files!r}\n".encode()+sources["receive_direct.py"]
config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
with (BASE/"launch-intent.json").open("x") as f:json.dump({"claim_commit":claim},f)
r=subprocess.run(["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),"a100-26","python3 -B -u -"],input=code,capture_output=True,timeout=660)
with (BASE/"original.stdout").open("xb") as f:f.write(r.stdout)
with (BASE/"original.stderr").open("xb") as f:f.write(r.stderr)
out={"transport_rc":r.returncode,"stdout_SHA256":hashlib.sha256(r.stdout).hexdigest(),"stderr_SHA256":hashlib.sha256(r.stderr).hexdigest()}
if r.returncode==0:out["value"]=json.loads(r.stdout)
with (BASE/"outcome.json").open("x") as f:json.dump(out,f,indent=2)
print(json.dumps(out))
