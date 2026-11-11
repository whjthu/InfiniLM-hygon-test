"""Verify the existing exact partial prefix against the original A100 source."""
import ast,hashlib,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent;B=BASE.parent
old=B/"training-minicpm-original-model-v1"
tree=ast.parse((old/"transfer_original_model.py").read_bytes())
constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ("SENDER","RECEIVER")}
pins=json.loads((old/"MODEL_FILES.json").read_bytes())["files"];claim="f206faea84a8022b2afcc4d51a2bca505bef9b28"
old_hashes={"a100":hashlib.sha256((f"PINS={pins!r}\n"+constants["SENDER"]).encode()).hexdigest(),
            "hygon":hashlib.sha256((f"PINS={pins!r}\nCLAIM={claim!r}\n"+constants["RECEIVER"]).encode()).hexdigest()}
code=r"""
import datetime,hashlib,json,os,signal,time
from pathlib import Path
signal.alarm(180)
live=[]
for p in Path("/proc").iterdir():
    if not p.name.isdigit():continue
    try:
        if p.stat().st_uid!=os.getuid():continue
        args=(p/"cmdline").read_bytes().split(b"\0")
        if b"-c" not in args:continue
        i=args.index(b"-c")
        if i+1<len(args) and hashlib.sha256(args[i+1]).hexdigest()==OLD_CODE_SHA:
            live.append(int(p.name))
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not live,"Original exact transfer process still live"
if PLATFORM=="a100":
    path=Path("/mnt/data/nvidia-baseline-20261010/inputs/models/MiniCPM-V-4.6/model.safetensors")
    assert path.stat().st_size==2600957528
else:
    root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-original-model-v1")
    failure=(root/"failure.json").read_bytes()
    assert hashlib.sha256(failure).hexdigest()=="bc3442cd990d713dfd62951eb41cba50509c480e2f082ce80d647795cbf3d3c3"
    assert not (root/"receipt.json").exists() and not (root/"resume-v1-intent.json").exists()
    path=root/"MiniCPM-V-4.6/model.safetensors.partial"
    assert path.stat().st_size==315947660
before=path.stat();start=time.monotonic();left=315947660;h=hashlib.sha256()
with path.open("rb") as f:
    while left:
        raw=f.read(min(left,1024*1024));assert raw
        h.update(raw);left-=len(raw)
after=path.stat()
assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
print(json.dumps({"observed_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"platform":PLATFORM,
    "original_exact_processes_live":live,"prefix_bytes":315947660,"prefix_SHA256":h.hexdigest(),"file_bytes":before.st_size,
    "stat":{"device":before.st_dev,"inode":before.st_ino,"mtime_ns":before.st_mtime_ns},"hash_seconds":time.monotonic()-start}))
"""
ns={"__file__":str(Path(__file__).resolve())}
prelude=(B/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf8")
exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),ns)
config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
result={}
for platform in ("a100","hygon"):
    ssh=ns["ssh"] if platform=="hygon" else ["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),"a100-26","python3"]
    r=subprocess.run(ssh,input=(f"PLATFORM={platform!r}\nOLD_CODE_SHA={old_hashes[platform]!r}\n"+code).encode(),capture_output=True,timeout=195)
    with (BASE/(platform+"-prefix.stderr")).open("xb") as f:f.write(r.stderr)
    assert r.returncode==0,f"{platform} prefix rc={r.returncode}"
    result[platform]=json.loads(r.stdout)
assert result["a100"]["prefix_SHA256"]==result["hygon"]["prefix_SHA256"]
with (BASE/"original-prefix-verification.json").open("x") as f:json.dump(result,f,indent=2)
print(json.dumps(result))
