"""Stage processor-only inputs and launch the published Hygon CPU comparison once."""
import base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;G=Path(sys.argv[1]);claim=sys.argv[2]
def blob(name):return subprocess.check_output(["git","-c","safe.directory="+G.as_posix(),"-C",str(G),"show",claim+":"+name])
task=json.loads(blob("coordination/tasks/training-minicpm-hygon-processor-v1.json"))
sources={}
for name,pin in task["source_SHA256"].items():
    data=blob(task["source"]+"/"+name);assert hashlib.sha256(data).hexdigest()==pin
    sources[name]=data
plan=json.loads(sources["PROCESSOR_INPUTS.json"])
config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
export=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),"a100-27","python3 -B -u -c "+shlex.quote(sources["export_processor_inputs.py"].decode())]
with (BASE/"launch-intent.json").open("x") as f:json.dump({"claim_commit":claim,"node":task["node"]},f)
r=subprocess.run(export,input=json.dumps(plan).encode(),capture_output=True,timeout=60)
with (BASE/"export.stderr").open("xb") as f:f.write(r.stderr)
assert r.returncode==0 and len(r.stdout)<=25000000
payload=r.stdout;digest=hashlib.sha256(payload).hexdigest()
ns={"__file__":str(Path(__file__).resolve())}
prelude=(BASE.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf8")
exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),ns)
bound={name:dict(b64=base64.b64encode(sources[name]).decode(),SHA256=task["source_SHA256"][name]) for name in ("capture_hygon_processor.py","compare_encoded_inputs.py")}
code=f"CLAIM={claim!r}\nPLAN={plan!r}\nSOURCES={bound!r}\nARCHIVE_SHA={digest!r}\n"+sources["receive_and_run_processor.py"].decode()
header=(json.dumps({"code_b64":base64.b64encode(code.encode()).decode()})+"\n").encode()
assert len(header)<=131072
bootstrap="import base64,json,sys;line=sys.stdin.buffer.readline(131073);assert len(line)<=131072 and line.endswith(b'\\n');exec(compile(base64.b64decode(json.loads(line)['code_b64']),'<frozen_cpu_receiver>','exec'))"
ssh=ns["ssh"].copy();ssh[-1]="/usr/local/python3.12/bin/python3 -B -u -c "+shlex.quote(bootstrap)
reply=subprocess.run(ssh,input=header+payload,capture_output=True,timeout=360)
with (BASE/"original.stdout").open("xb") as f:f.write(reply.stdout)
with (BASE/"original.stderr").open("xb") as f:f.write(reply.stderr)
value={"transport_rc":reply.returncode,"archive_bytes":len(payload),"archive_SHA256":digest,"stdout_SHA256":hashlib.sha256(reply.stdout).hexdigest(),"stderr_SHA256":hashlib.sha256(reply.stderr).hexdigest()}
if reply.returncode==0:value["outcome"]=json.loads(reply.stdout)
with (BASE/"outcome.json").open("x") as f:json.dump(value,f,indent=2)
print(json.dumps(value))
