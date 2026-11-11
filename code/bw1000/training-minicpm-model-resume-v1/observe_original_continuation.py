"""Observe only the two saved original continuation processes and target byte count."""
import hashlib,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
intent=json.loads((BASE/"continuation-intent.json").read_bytes())
config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
ns={"__file__":str(Path(__file__).resolve())}
prelude=(BASE.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf8")
exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),ns)
code=r"""
import datetime,hashlib,json,os
from pathlib import Path
root=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-model-resume-v1") if PLATFORM=="a100" else Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-original-model-v1")
name="intent.json" if PLATFORM=="a100" else "resume-v1-intent.json"
raw=(root/name).read_bytes();intent=json.loads(raw)
assert intent["claim_commit"]==EXPECTED["claim_commit"] and intent["epoch"]==EXPECTED["epoch"] and intent["deadline"]==EXPECTED["deadline"]
pid=intent["PID"];p=Path("/proc")/str(pid);actual=None
try:
    values=(p/"stat").read_text().split(") ",1)[1].split()
    argv=(p/"cmdline").read_bytes().split(b"\0");i=argv.index(b"-c")
    actual={"PID":pid,"startticks":int(values[19]),"PGID":int(values[2]),"process_state":values[0],
            "code_SHA256":hashlib.sha256(argv[i+1]).hexdigest(),"same_UID":p.stat().st_uid==os.getuid()}
except (FileNotFoundError,ProcessLookupError):pass
if actual:
    expected_hash=EXPECTED["source_code_SHA256"] if PLATFORM=="a100" else EXPECTED["receiver_code_SHA256"]
    assert actual["startticks"]==intent["startticks"] and actual["PGID"]==intent["PGID"] and actual["code_SHA256"]==expected_hash and actual["same_UID"]
out={"observed_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"platform":PLATFORM,"intent_SHA256":hashlib.sha256(raw).hexdigest(),"intent":intent,"actual_process":actual}
names=("receipt.json","failure.json") if PLATFORM=="a100" else ("resume-v1-receipt.json","resume-v1-failure.json")
for name in names:
    p=root/name
    if p.is_file():
        raw=p.read_bytes();out[name]={"SHA256":hashlib.sha256(raw).hexdigest(),"value":json.loads(raw)}
if PLATFORM=="hygon":
    model=root/"MiniCPM-V-4.6"
    out["files"]={p.name:p.stat().st_size for p in model.iterdir()}
print(json.dumps(out))
"""
out={}
for platform in ("a100","hygon"):
    ssh=ns["ssh"] if platform=="hygon" else ["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),"a100-26","python3"]
    r=subprocess.run(ssh,input=(f"PLATFORM={platform!r}\nEXPECTED={intent!r}\n"+code).encode(),capture_output=True,timeout=30)
    assert r.returncode==0,f"{platform} observer rc={r.returncode}"
    out[platform]=json.loads(r.stdout)
name=sys.argv[1] if len(sys.argv)>1 else "original-continuation-first-observation.json"
assert Path(name).name==name and name.startswith("original-continuation-") and name.endswith(".json")
with (BASE/name).open("x") as f:json.dump(out,f,indent=2)
print(json.dumps(out))
