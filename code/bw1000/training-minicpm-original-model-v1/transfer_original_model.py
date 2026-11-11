"""Copy exact original MiniCPM model files using existing finite SSH commands."""
import datetime,hashlib,json,shlex,subprocess,sys,threading,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
SENDER=r"""
import hashlib,json,signal,sys
from pathlib import Path
signal.alarm(900)
root=Path("/mnt/data/nvidia-baseline-20261010/inputs/models/MiniCPM-V-4.6")
for name,item in PINS.items():
    path=root/name
    assert path.is_file() and not path.is_symlink() and path.stat().st_size==item["bytes"]
    sys.stdout.buffer.write((json.dumps({"name":name,**item})+"\n").encode());sys.stdout.buffer.flush()
    h=hashlib.sha256()
    with path.open("rb") as f:
        left=item["bytes"]
        while left:
            raw=f.read(min(left,1024*1024));assert raw
            h.update(raw);sys.stdout.buffer.write(raw);left-=len(raw)
        assert not f.read(1)
    sys.stdout.buffer.flush()
    assert h.hexdigest()==item["SHA256"],name
"""
RECEIVER=r"""
import datetime,hashlib,json,os,shutil,signal,sys,time
from pathlib import Path
signal.alarm(900)
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-original-model-v1")
assert not root.exists(),"Reconcile existing exact intent and receipt before any retry"
assert shutil.disk_usage(root.parent).free>sum(x["bytes"] for x in PINS.values())+1024*1024*1024
root.mkdir(mode=0o700)
model=root/"MiniCPM-V-4.6";model.mkdir(mode=0o700)
start=time.monotonic()
intent={"claim_commit":CLAIM,"state":"COPYING","expected":PINS,"UTC":datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (root/"intent.json").open("x") as f:json.dump(intent,f)
done={}
try:
    for name,item in PINS.items():
        header=sys.stdin.buffer.readline(512)
        assert header.endswith(b"\n") and json.loads(header)=={"name":name,**item}
        assert Path(name).name==name and name not in (".","..")
        temp=model/(name+".partial")
        h=hashlib.sha256();left=item["bytes"]
        with temp.open("xb") as f:
            while left:
                raw=sys.stdin.buffer.read(min(left,1024*1024));assert raw
                f.write(raw);h.update(raw);left-=len(raw)
            f.flush();os.fsync(f.fileno())
        assert h.hexdigest()==item["SHA256"] and temp.stat().st_size==item["bytes"],name
        temp.rename(model/name);done[name]=item
    assert not sys.stdin.buffer.read(1)
    result={"state":"VERIFIED_ORIGINAL_MODEL_FILES","claim_commit":CLAIM,"files":done,
            "bytes":sum(x["bytes"] for x in done.values()),"wall_seconds":time.monotonic()-start,
            "model_directory":str(model),"GPU_execution":False,"model_loaded":False}
    with (root/"receipt.json").open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
except BaseException as e:
    with (root/"failure.json").open("x") as f:json.dump({"state":"COPY_FAILED","error_type":type(e).__name__,"completed_files":done},f)
    raise
"""
def main():
    G=Path(sys.argv[1]);claim=sys.argv[2]
    def blob(path):return subprocess.check_output(["git","-c","safe.directory="+G.as_posix(),"-C",str(G),"show",claim+":"+path])
    task=json.loads(blob("coordination/tasks/training-minicpm-original-model-v1.json"))
    source=blob(task["source"]+"/transfer_original_model.py")
    assert hashlib.sha256(source).hexdigest()==task["source_SHA256"]["transfer_original_model.py"]
    manifest=blob(task["source"]+"/MODEL_FILES.json")
    assert hashlib.sha256(manifest).hexdigest()==task["source_SHA256"]["MODEL_FILES.json"]
    pins=json.loads(manifest)["files"]
    prelude=(BASE.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf8")
    ns={"__file__":str(Path(__file__).resolve())}
    exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),ns)
    receiver=ns["ssh"].copy()
    receiver[-1]="/usr/local/python3.12/bin/python3 -B -u -c "+shlex.quote(f"PINS={pins!r}\nCLAIM={claim!r}\n"+RECEIVER)
    config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
    sender=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),"a100-26","python3 -B -u -c "+shlex.quote(f"PINS={pins!r}\n"+SENDER)]
    with (BASE/"transfer-intent.json").open("x") as f:json.dump({"claim_commit":claim,"local_epoch":time.time(),"budget_seconds":900},f)
    start=time.monotonic();copied=0;next_progress=64*1024*1024
    with (BASE/"sender.stderr").open("xb") as se,(BASE/"receiver.stderr").open("xb") as re,(BASE/"receiver.stdout").open("xb") as ro:
        dest=subprocess.Popen(receiver,stdin=subprocess.PIPE,stdout=ro,stderr=re)
        orig=subprocess.Popen(sender,stdout=subprocess.PIPE,stderr=se)
        def timeout():
            for p in (orig,dest):
                if p.poll() is None:p.terminate()
        timer=threading.Timer(900,timeout);timer.daemon=True;timer.start()
        try:
            while True:
                raw=orig.stdout.read(1024*1024)
                if not raw:break
                dest.stdin.write(raw);copied+=len(raw)
                if copied>=next_progress:
                    progress={"copied_bytes":copied,"wall_seconds":time.monotonic()-start,"sender_PID":orig.pid,"receiver_PID":dest.pid,"sender_live":orig.poll() is None,"receiver_live":dest.poll() is None}
                    temp=BASE/"transfer-progress.tmp"
                    temp.write_bytes(json.dumps(progress).encode());temp.replace(BASE/"transfer-progress.json")
                    print(json.dumps(progress),flush=True);next_progress+=64*1024*1024
            dest.stdin.close();source_rc=orig.wait(timeout=20);dest_rc=dest.wait(timeout=20)
        except BaseException:
            timeout()
            for p in (orig,dest):
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            raise
        finally:timer.cancel()
    raw=(BASE/"receiver.stdout").read_bytes()
    result={"claim_commit":claim,"source_rc":source_rc,"destination_rc":dest_rc,"stream_bytes":copied,
            "wall_seconds":time.monotonic()-start,"stdout_SHA256":hashlib.sha256(raw).hexdigest()}
    if dest_rc==0:result["receipt"]=json.loads(raw)
    with (BASE/"transfer-outcome.json").open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="receipt"}))
    assert source_rc==dest_rc==0 and result["receipt"]["state"]=="VERIFIED_ORIGINAL_MODEL_FILES"
if __name__=="__main__":main()
