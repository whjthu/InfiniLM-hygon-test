"""Install the exact original fast SDK via the existing internal A100 SSH route."""
import hashlib,io,json,os,shlex,signal,subprocess,sys,tarfile,time
from pathlib import Path,PurePosixPath
CONTROL=Path(__file__).resolve().parent.parent
ROOT=CONTROL;SRC=CONTROL/"src"
installed=json.loads((CONTROL/"installed.json").read_bytes());CLAIM=installed["claim_commit"]
for name,pin in installed["files"].items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==pin,name
assert sorted(p.name for p in ROOT.iterdir())==["installed.json","src"]
assert hashlib.sha256((ROOT.parent/"direct-v2/receipt.json").read_bytes()).hexdigest()=="f15515aed67a4ab27879f7b32225c9aa59742f589b749c4360a17b8c0dca0ed6"

epoch=time.time();deadline=epoch+600;work_deadline=epoch+540
startticks=int(Path("/proc/self/stat").read_text().split(") ",1)[1].split()[19])
with (CONTROL/"intent.json").open("x") as f:json.dump({"claim_commit":CLAIM,"epoch":epoch,"deadline":deadline,
    "work_deadline":work_deadline,"PID":os.getpid(),"PGID":os.getpgrp(),"startticks":startticks},f)
def expired(*args):raise TimeoutError("registered software-copy work deadline reached")
signal.signal(signal.SIGALRM,expired);signal.alarm(540)
raw=(SRC/"OVERLAY_FILES.json").read_bytes();plan_sha=hashlib.sha256(raw).hexdigest();plan=json.loads(raw)
sender=f"DEADLINE={work_deadline!r}\nPLAN_SHA={plan_sha!r}\n"+(SRC/"export_overlay.py").read_text()
peer=["ssh","-T","-o","BatchMode=yes","-o","StrictHostKeyChecking=yes","-o","ConnectTimeout=8",
      "whj@10.119.16.27","python3 -B -u -c "+shlex.quote(sender)]
proc=None;seen=set()
try:
    with (CONTROL/"source.stderr").open("xb") as error:
        proc=subprocess.Popen(peer,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error,start_new_session=True)
        pstat=Path("/proc")/str(proc.pid)/"stat"
        values=pstat.read_text().split(") ",1)[1].split()
        with (CONTROL/"source-owner.json").open("x") as f:json.dump({"PID":proc.pid,"PGID":int(values[2]),"startticks":int(values[19]),"claim_commit":CLAIM},f)
        proc.stdin.write(raw);proc.stdin.close()
        with tarfile.open(fileobj=proc.stdout,mode="r|gz") as archive:
            entries=iter(archive);first=next(entries)
            assert first.isfile() and first.name=="PLAN.json" and first.size==len(raw)
            manifest=archive.extractfile(first).read()
            assert manifest==raw
            with (ROOT/"PLAN.json").open("xb") as f:f.write(manifest)
            for member in entries:
                p=PurePosixPath(member.name)
                assert member.isfile() and not p.is_absolute() and ".." not in p.parts
                assert member.name in plan["files"] and member.name not in seen
                item=plan["files"][member.name];assert member.size==item["bytes"]
                data=archive.extractfile(member).read()
                assert hashlib.sha256(data).hexdigest()==item["SHA256"]
                dest=ROOT/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open("xb") as f:f.write(data)
                dest.chmod(item["mode"]&0o777);seen.add(member.name)
            assert seen==set(plan["files"])
        assert proc.wait(timeout=max(1,work_deadline-time.time()-5))==0
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1",HIP_VISIBLE_DEVICES="-1",ROCR_VISIBLE_DEVICES="-1",
        PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",FAST_OVERLAY_ROOT=str(ROOT))
    env["PYTHONPATH"]=str(ROOT/"site")+os.pathsep+str(ROOT.parent/"site")+os.pathsep+env.get("PYTHONPATH","")
    env["PATH"]=str(ROOT.parent/"bin")+os.pathsep+"/home/whj/nvidia-baseline/env-vlm/bin"+os.pathsep+env["PATH"]
    with (CONTROL/"verification.log").open("xb") as log:
        verifier=subprocess.run(["/home/whj/nvidia-baseline/env-vlm/bin/python","-B","-u",str(SRC/"verify_overlay.py")],
            env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,min(120,work_deadline-time.time()-5)))
    result={"claim_commit":CLAIM,"state":"ORIGINAL_SOFTWARE_FILES_VERIFIED","epoch":epoch,"deadline":deadline,
        "work_deadline":work_deadline,"files":len(seen),"bytes":sum(x["bytes"] for x in plan["files"].values()),
        "PLAN_SHA256":plan_sha,"source_rc":proc.returncode,"verification_rc":verifier.returncode,"GPU_execution":False,
        "direct_internal_SSH":True,"original_attempt_preserved":True}
    p=CONTROL/"verification.json"
    if p.is_file():
        data=p.read_bytes();result["verification"]={"SHA256":hashlib.sha256(data).hexdigest(),"value":json.loads(data)}
    with (CONTROL/"receipt.json").open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
except BaseException as error:
    with (CONTROL/"failure.json").open("x") as f:json.dump({"claim_commit":CLAIM,"error_type":type(error).__name__,
        "completed_files":len(seen),"expected_files":len(plan["files"]),"GPU_execution":False},f)
    raise
finally:
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
