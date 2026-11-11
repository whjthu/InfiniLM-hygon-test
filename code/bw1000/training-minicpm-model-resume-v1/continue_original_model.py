"""Continue only the verified original MiniCPM prefix with one new finite budget."""
import ast,base64,datetime,hashlib,json,math,shlex,subprocess,sys,threading,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
SENDER=r"""
import datetime,hashlib,json,math,os,signal,sys,time
from pathlib import Path
remaining=DEADLINE-time.time()
assert math.isfinite(remaining) and remaining>0
def expired(*args):raise TimeoutError("registered continuation deadline reached")
signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(remaining)))
root=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-model-resume-v1")
assert not root.exists()
root.mkdir(mode=0o700)
startticks=int(Path("/proc/self/stat").read_text().split(") ",1)[1].split()[19])
intent={"claim_commit":CLAIM,"epoch":EPOCH,"deadline":DEADLINE,"PID":os.getpid(),"PGID":os.getpgrp(),
        "startticks":startticks,"prefix_bytes":OFFSET,"prefix_SHA256":PREFIX,"state":"SENDING"}
with (root/"intent.json").open("x") as f:json.dump(intent,f)
try:
    path=Path("/mnt/data/nvidia-baseline-20261010/inputs/models/MiniCPM-V-4.6/model.safetensors")
    before=path.stat();assert before.st_size==TOTAL and not path.is_symlink()
    h=prefix_hash(path,OFFSET,PREFIX)
    header={"offset":OFFSET,"prefix_SHA256":PREFIX,"total_bytes":TOTAL,"full_SHA256":FULL}
    sys.stdout.buffer.write((json.dumps(header)+"\n").encode());sys.stdout.buffer.flush()
    with path.open("rb") as f:
        f.seek(OFFSET);left=TOTAL-OFFSET
        while left:
            raw=f.read(min(left,1024*1024));assert raw
            h.update(raw);sys.stdout.buffer.write(raw);left-=len(raw)
        assert not f.read(1)
    sys.stdout.buffer.flush();after=path.stat()
    assert h.hexdigest()==FULL and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    with (root/"receipt.json").open("x") as f:json.dump({"claim_commit":CLAIM,"state":"SOURCE_WHOLE_FILE_HASH_VERIFIED","bytes":TOTAL,"SHA256":h.hexdigest()},f)
except BaseException as e:
    with (root/"failure.json").open("x") as f:json.dump({"claim_commit":CLAIM,"error_type":type(e).__name__,"state":"SOURCE_FAILED"},f)
    raise
"""
RECEIVER=r"""
import datetime,hashlib,json,math,os,signal,sys,time
from pathlib import Path
remaining=DEADLINE-time.time()
assert math.isfinite(remaining) and remaining>0
def expired(*args):raise TimeoutError("registered continuation deadline reached")
signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(remaining)))
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-original-model-v1")
assert hashlib.sha256((root/"failure.json").read_bytes()).hexdigest()=="bc3442cd990d713dfd62951eb41cba50509c480e2f082ce80d647795cbf3d3c3"
assert not (root/"receipt.json").exists() and not (root/"resume-v1-intent.json").exists()
model=root/"MiniCPM-V-4.6";partial=model/"model.safetensors.partial"
assert partial.stat().st_size==OFFSET and not (model/"model.safetensors").exists()
prefix_hash(partial,OFFSET,PREFIX)
startticks=int(Path("/proc/self/stat").read_text().split(") ",1)[1].split()[19])
intent={"claim_commit":CLAIM,"epoch":EPOCH,"deadline":DEADLINE,"PID":os.getpid(),"PGID":os.getpgrp(),
        "startticks":startticks,"state":"CONTINUING","prefix_bytes":OFFSET,"prefix_SHA256":PREFIX}
with (root/"resume-v1-intent.json").open("x") as f:json.dump(intent,f)
started=time.monotonic()
try:
    metadata=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-hygon-processor-v1/model")
    repaired=[]
    for name,item in PINS.items():
        if name=="model.safetensors":continue
        raw=(metadata/name).read_bytes()
        assert len(raw)==item["bytes"] and hashlib.sha256(raw).hexdigest()==item["SHA256"]
        target=model/name
        if target.exists():
            assert target.is_file() and not target.is_symlink() and target.read_bytes()==raw
        else:
            with target.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
            repaired.append(name)
    header=sys.stdin.buffer.readline(1024)
    assert header.endswith(b"\n") and json.loads(header)=={"offset":OFFSET,"prefix_SHA256":PREFIX,"total_bytes":TOTAL,"full_SHA256":FULL}
    checkpoint=OFFSET
    def progress(written,digest):
        global checkpoint
        if written-checkpoint<64*1024*1024:return
        checkpoint=written
        value={"claim_commit":CLAIM,"bytes_written":written,"continued_bytes":written-OFFSET,
               "prefix_hash_through_written":digest,"UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "PID":os.getpid(),"startticks":startticks,"wall_seconds":time.monotonic()-started}
        temp=root/"resume-v1-progress.tmp"
        temp.write_bytes(json.dumps(value).encode());temp.replace(root/"resume-v1-progress.json")
    copied=append_verified(partial,sys.stdin.buffer,OFFSET,TOTAL,PREFIX,FULL,progress)
    final=model/"model.safetensors"
    os.link(partial,final);partial.unlink()
    assert sorted(p.name for p in model.iterdir())==sorted(PINS)
    result={"claim_commit":CLAIM,"state":"CONTINUED_ORIGINAL_MODEL_FILES_VERIFIED","epoch":EPOCH,"deadline":DEADLINE,
            "files":PINS,"copy":copied,"metadata_files_repaired":repaired,"wall_seconds":time.monotonic()-started,
            "model_directory":str(model),"GPU_execution":False,"model_loaded":False}
    with (root/"resume-v1-receipt.json").open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
except BaseException as e:
    with (root/"resume-v1-failure.json").open("x") as f:json.dump({"claim_commit":CLAIM,"state":"CONTINUATION_FAILED","error_type":type(e).__name__,"partial_bytes":partial.stat().st_size if partial.exists() else None},f)
    raise
"""
def main():
    G=Path(sys.argv[1]);claim=sys.argv[2]
    def blob(path):return subprocess.check_output(["git","-c","safe.directory="+G.as_posix(),"-C",str(G),"show",claim+":"+path])
    task=json.loads(blob("coordination/tasks/training-minicpm-model-resume-v1.json"))
    source=blob(task["source"]+"/continue_original_model.py")
    assert source==Path(__file__).read_bytes() and hashlib.sha256(source).hexdigest()==task["source_SHA256"]["continue_original_model.py"]
    helper=blob(task["source"]+"/resume_stream.py")
    manifest=blob(task["source"]+"/MODEL_FILES.json")
    assert hashlib.sha256(helper).hexdigest()==task["source_SHA256"]["resume_stream.py"]
    assert hashlib.sha256(manifest).hexdigest()==task["source_SHA256"]["MODEL_FILES.json"]
    pins=json.loads(manifest)["files"]
    epoch=time.time();deadline=epoch+10800
    shared=f"CLAIM={claim!r}\nEPOCH={epoch!r}\nDEADLINE={deadline!r}\nOFFSET={task['prefix_bytes']!r}\nPREFIX={task['prefix_SHA256']!r}\nTOTAL={pins['model.safetensors']['bytes']!r}\nFULL={pins['model.safetensors']['SHA256']!r}\nPINS={pins!r}\n"+helper.decode()+"\n"
    sender_code=shared+SENDER;receiver_code=shared+RECEIVER
    with (BASE/"continuation-intent.json").open("x") as f:json.dump({"claim_commit":claim,"epoch":epoch,"deadline":deadline,
        "prefix_bytes":task["prefix_bytes"],"prefix_SHA256":task["prefix_SHA256"],
        "source_code_SHA256":hashlib.sha256(sender_code.encode()).hexdigest(),"receiver_code_SHA256":hashlib.sha256(receiver_code.encode()).hexdigest()},f,indent=2)
    ns={"__file__":str(Path(__file__).resolve())}
    prelude=(BASE.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf8")
    exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),ns)
    receiver=ns["ssh"].copy();receiver[-1]="/usr/local/python3.12/bin/python3 -B -u -c "+shlex.quote(receiver_code)
    config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
    sender=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),"a100-26","python3 -B -u -c "+shlex.quote(sender_code)]
    started=time.monotonic();copied=0;next_progress=64*1024*1024;error=None
    with (BASE/"sender.stderr").open("xb") as se,(BASE/"receiver.stderr").open("xb") as re,(BASE/"receiver.stdout").open("xb") as ro:
        dest=subprocess.Popen(receiver,stdin=subprocess.PIPE,stdout=ro,stderr=re)
        orig=subprocess.Popen(sender,stdout=subprocess.PIPE,stderr=se)
        with (BASE/"owned-local-transports.json").open("x") as f:json.dump({"sender_PID":orig.pid,"receiver_PID":dest.pid},f)
        def stop_owned():
            for p in (orig,dest):
                if p.poll() is None:p.terminate()
        timer=threading.Timer(max(1,10800-(time.monotonic()-started)),stop_owned);timer.daemon=True;timer.start()
        try:
            while True:
                raw=orig.stdout.read(1024*1024)
                if not raw:break
                dest.stdin.write(raw);copied+=len(raw)
                if copied>=next_progress:
                    row={"claim_commit":claim,"continued_pipe_bytes":copied,"retained_prefix_bytes":task["prefix_bytes"],
                         "wall_seconds":time.monotonic()-started,"sender_PID":orig.pid,"receiver_PID":dest.pid,
                         "sender_live":orig.poll() is None,"receiver_live":dest.poll() is None}
                    temp=BASE/"continuation-progress.tmp";temp.write_bytes(json.dumps(row).encode());temp.replace(BASE/"continuation-progress.json")
                    print(json.dumps(row),flush=True);next_progress+=64*1024*1024
            dest.stdin.close();orig.wait(timeout=20);dest.wait(timeout=20)
        except BaseException as e:
            error=type(e).__name__;stop_owned()
            for p in (orig,dest):
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        finally:timer.cancel()
    raw=(BASE/"receiver.stdout").read_bytes()
    result={"claim_commit":claim,"source_rc":orig.returncode,"destination_rc":dest.returncode,"local_error":error,
            "continued_pipe_bytes":copied,"wall_seconds":time.monotonic()-started,"stdout_SHA256":hashlib.sha256(raw).hexdigest()}
    if dest.returncode==0:result["receipt"]=json.loads(raw)
    with (BASE/"continuation-outcome.json").open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="receipt"}))
    assert orig.returncode==dest.returncode==0 and error is None
if __name__=="__main__":main()
