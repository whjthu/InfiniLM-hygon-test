"""Copy only the published original fast SDK overlay from A100-27 to A100-26."""
import base64,hashlib,json,shlex,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;G=Path(sys.argv[1]);claim=sys.argv[2]
def blob(path):return subprocess.check_output(["git","-c","safe.directory="+G.as_posix(),"-C",str(G),"show",claim+":"+path])
task=json.loads(blob("coordination/tasks/training-minicpm-a100-fast-overlay-v1.json"))
sources={}
for name,pin in task["source_SHA256"].items():
    data=blob(task["source"]+"/"+name);assert hashlib.sha256(data).hexdigest()==pin
    sources[name]=data
plan=sources["OVERLAY_FILES.json"];plan_sha=hashlib.sha256(plan).hexdigest()
config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
epoch=time.time();deadline=epoch+600
with (BASE/"transfer-intent.json").open("x") as f:json.dump({"claim_commit":claim,"epoch":epoch,"deadline":deadline},f)
sender_code=f"DEADLINE={deadline!r}\nPLAN_SHA={plan_sha!r}\n"+sources["export_overlay.py"].decode()
sender=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),"a100-27","python3 -B -u -c "+shlex.quote(sender_code)]
r=subprocess.run(sender,input=plan,capture_output=True,timeout=max(1,min(180,deadline-time.time()-5)))
with (BASE/"sender.stderr").open("xb") as f:f.write(r.stderr)
assert r.returncode==0 and len(r.stdout)<=256*1024**2
archive_sha=hashlib.sha256(r.stdout).hexdigest()
code=f"CLAIM={claim!r}\nEPOCH={epoch!r}\nDEADLINE={deadline!r}\nPLAN_SHA={plan_sha!r}\nARCHIVE_SHA={archive_sha!r}\nVERIFIER_B64={base64.b64encode(sources['verify_overlay.py']).decode()!r}\nVERIFIER_SHA={task['source_SHA256']['verify_overlay.py']!r}\n"+sources["receive_overlay.py"].decode()
header=(json.dumps({"code_b64":base64.b64encode(code.encode()).decode()})+"\n").encode();assert len(header)<131072
bootstrap="import base64,json,sys;line=sys.stdin.buffer.readline(131073);assert len(line)<131072;exec(compile(base64.b64decode(json.loads(line)['code_b64']),'<exact_overlay_receiver>','exec'))"
receiver=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),"a100-26","python3 -B -u -c "+shlex.quote(bootstrap)]
reply=subprocess.run(receiver,input=header+r.stdout,capture_output=True,timeout=max(1,deadline-time.time()-5))
with (BASE/"receiver.stdout").open("xb") as f:f.write(reply.stdout)
with (BASE/"receiver.stderr").open("xb") as f:f.write(reply.stderr)
out={"transport_rc":reply.returncode,"epoch":epoch,"deadline":deadline,"archive_bytes":len(r.stdout),"archive_SHA256":archive_sha,"stderr_SHA256":hashlib.sha256(reply.stderr).hexdigest()}
if reply.returncode==0:out["receipt"]=json.loads(reply.stdout)
with (BASE/"outcome.json").open("x") as f:json.dump(out,f,indent=2)
print(json.dumps(out))
