"""Execute the published CPU-only original MiniCPM template capture once."""
import hashlib,json,shlex,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
G=Path(sys.argv[1]);claim=sys.argv[2]
def blob(path):return subprocess.check_output(["git","-c","safe.directory="+G.as_posix(),"-C",str(G),"show",claim+":"+path])
task=json.loads(blob("coordination/tasks/training-minicpm-original-processor-v1.json"))
code=blob(task["source"]+"/capture_original_processor.py")
assert hashlib.sha256(code).hexdigest()==task["source_SHA256"]["capture_original_processor.py"]
config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
script=f"CLAIM={claim!r}\n"+code.decode()
command="env CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 /home/whj/nvidia-baseline/env-vlm-fast/bin/python -B -u -c "+shlex.quote(script)
ssh=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),"a100-26",command]
with (BASE/"launch-intent.json").open("x") as f:json.dump({"claim_commit":claim},f)
r=subprocess.run(ssh,capture_output=True,timeout=200)
with (BASE/"original.stdout").open("xb") as f:f.write(r.stdout)
with (BASE/"original.stderr").open("xb") as f:f.write(r.stderr)
rows=[json.loads(x) for x in r.stdout.splitlines() if x.startswith(b"{")]
value={"rc":r.returncode,"stdout_SHA256":hashlib.sha256(r.stdout).hexdigest(),"stderr_SHA256":hashlib.sha256(r.stderr).hexdigest(),"rows":len(rows)}
if len(rows)==1:value["receipt"]=rows[0]
with (BASE/"outcome.json").open("x") as f:json.dump(value,f,indent=2)
print(json.dumps(value))
