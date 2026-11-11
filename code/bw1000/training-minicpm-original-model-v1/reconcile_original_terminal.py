"""Reconcile the exact original transfer after its local handle terminates."""
import ast,hashlib,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
tree=ast.parse((BASE/"transfer_original_model.py").read_bytes())
constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ("SENDER","RECEIVER")}
pins=json.loads((BASE/"MODEL_FILES.json").read_bytes())["files"]
claim="f206faea84a8022b2afcc4d51a2bca505bef9b28"
hashes={"a100":hashlib.sha256((f"PINS={pins!r}\n"+constants["SENDER"]).encode()).hexdigest(),
        "hygon":hashlib.sha256((f"PINS={pins!r}\nCLAIM={claim!r}\n"+constants["RECEIVER"]).encode()).hexdigest()}
code=r"""
import datetime,hashlib,json,os
from pathlib import Path
matches=[]
for p in Path("/proc").iterdir():
    if not p.name.isdigit():continue
    try:
        if p.stat().st_uid!=os.getuid():continue
        argv=(p/"cmdline").read_bytes().split(b"\0")
        if not argv or b"-c" not in argv:continue
        i=argv.index(b"-c")
        if i+1<len(argv) and hashlib.sha256(argv[i+1]).hexdigest()==EXPECTED_CODE_SHA:
            stat=(p/"stat").read_text().split(") ",1)[1].split()
            matches.append({"PID":int(p.name),"startticks":int(stat[19])})
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
out={"observed_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"matching_original_processes":matches}
if PLATFORM=="hygon":
    root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-original-model-v1")
    out["root_exists"]=root.exists()
    for name in ("intent.json","receipt.json","failure.json"):
        path=root/name
        if path.is_file():
            raw=path.read_bytes();out[name]={"SHA256":hashlib.sha256(raw).hexdigest(),"value":json.loads(raw)}
    model=root/"MiniCPM-V-4.6"
    if model.is_dir():out["files"]={p.name:p.stat().st_size for p in model.iterdir()}
print(json.dumps(out))
"""
ns={"__file__":str(Path(__file__).resolve())}
prelude=(BASE.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf8")
exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),ns)
config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
result={}
for platform in ("a100","hygon"):
    ssh=ns["ssh"] if platform=="hygon" else ["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),"a100-26","python3"]
    r=subprocess.run(ssh,input=(f"PLATFORM={platform!r}\nEXPECTED_CODE_SHA={hashes[platform]!r}\n"+code).encode(),capture_output=True,timeout=35)
    assert r.returncode==0
    result[platform]=json.loads(r.stdout)
with (BASE/"original-transfer-terminal-reconcile.json").open("x") as f:json.dump(result,f,indent=2)
print(json.dumps(result))
