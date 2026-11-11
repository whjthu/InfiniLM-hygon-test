"""Stage frozen small training inputs over the two existing SSH configurations."""
import hashlib,json,shlex,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
B=BASE.parent
A=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline")
G=Path(sys.argv[1]);CLAIM=sys.argv[2]
def blob(name):
    return subprocess.check_output(["git","-c","safe.directory="+G.as_posix(),"-C",str(G),"show",CLAIM+":"+name])
task=json.loads(blob("coordination/tasks/training-vlm-original-assets-v1.json"))
sources={}
for name,pin in task["source_SHA256"].items():
    raw=blob(task["source"]+"/"+name)
    assert hashlib.sha256(raw).hexdigest()==pin
    sources[name]=raw
plan=json.loads(sources["TRANSFER_PLAN.json"])
with (BASE/"stage-intent.json").open("x") as f:json.dump({"claim_commit":CLAIM},f)
a100=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(A/"ssh_config"),"a100-26","python3 -c "+shlex.quote(sources["export_original_inputs.py"].decode())]
export=subprocess.run(a100,input=json.dumps(plan).encode(),capture_output=True,timeout=90)
with (BASE/"export.stderr").open("xb") as f:f.write(export.stderr)
assert export.returncode==0 and len(export.stdout)<=12000000
archive=export.stdout
archive_sha=hashlib.sha256(archive).hexdigest()
prelude=(B/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf8")
ns={"__file__":str(Path(__file__).resolve())}
exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),ns)
code=f"CLAIM={CLAIM!r}\nPLAN={plan!r}\nARCHIVE_SHA={archive_sha!r}\n"+sources["receive_original_inputs.py"].decode()
ssh=ns["ssh"].copy()
ssh[-1]="/usr/local/python3.12/bin/python3 -B -u -c "+shlex.quote(code)
reply=subprocess.run(ssh,input=archive,capture_output=True,timeout=90)
with (BASE/"stage.stdout").open("xb") as f:f.write(reply.stdout)
with (BASE/"stage.stderr").open("xb") as f:f.write(reply.stderr)
status={"transport_rc":reply.returncode,"archive_bytes":len(archive),"archive_SHA256":archive_sha,"stdout_SHA256":hashlib.sha256(reply.stdout).hexdigest(),"stderr_SHA256":hashlib.sha256(reply.stderr).hexdigest()}
if reply.returncode==0:status["receipt"]=json.loads(reply.stdout)
with (BASE/"stage-outcome.json").open("x") as f:json.dump(status,f,indent=2)
print(json.dumps({k:v for k,v in status.items() if k!="receipt"}))
assert reply.returncode==0
print(json.dumps({k:v for k,v in status["receipt"].items() if k!="files_SHA256"}))
