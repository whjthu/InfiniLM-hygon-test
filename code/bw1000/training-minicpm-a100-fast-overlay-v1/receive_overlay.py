"""Receive exact non-model software inputs into a new private A100 runtime prefix."""
import base64,hashlib,io,json,os,signal,subprocess,sys,tarfile,time
from pathlib import Path,PurePosixPath
remaining=DEADLINE-time.time()
assert 0<remaining<=600
signal.alarm(max(1,int(remaining)))
ROOT=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-fast-overlay-v1")
assert not ROOT.exists()
ROOT.mkdir(mode=0o700)
with (ROOT/"intent.json").open("x") as f:json.dump({"claim_commit":CLAIM,"epoch":EPOCH,"deadline":DEADLINE,"GPU_work":False},f)
raw=sys.stdin.buffer.read(256*1024**2+1);assert len(raw)<=256*1024**2 and hashlib.sha256(raw).hexdigest()==ARCHIVE_SHA
with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as archive:
    members=iter(archive);first=next(members)
    assert first.isfile() and first.name=="PLAN.json" and first.size<2000000
    data=archive.extractfile(first).read()
    assert hashlib.sha256(data).hexdigest()==PLAN_SHA
    plan=json.loads(data)
    with (ROOT/"PLAN.json").open("xb") as f:f.write(data)
    seen=set()
    for member in members:
        name=PurePosixPath(member.name)
        assert member.isfile() and not name.is_absolute() and ".." not in name.parts
        assert member.name in plan["files"] and member.name not in seen
        expected=plan["files"][member.name]
        assert member.size==expected["bytes"]
        data=archive.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==expected["SHA256"]
        path=ROOT/member.name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as f:f.write(data)
        path.chmod(expected["mode"]&0o777)
        seen.add(member.name)
    assert seen==set(plan["files"])
verifier=base64.b64decode(VERIFIER_B64);assert hashlib.sha256(verifier).hexdigest()==VERIFIER_SHA
with (ROOT/"verify_overlay.py").open("xb") as f:f.write(verifier)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="-1",HIP_VISIBLE_DEVICES="-1",ROCR_VISIBLE_DEVICES="-1",
    PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
env["PYTHONPATH"]=str(ROOT/"site")+os.pathsep+env.get("PYTHONPATH","")
env["PATH"]=str(ROOT/"bin")+os.pathsep+"/home/whj/nvidia-baseline/env-vlm/bin"+os.pathsep+env["PATH"]
with (ROOT/"verification.log").open("xb") as log:
    p=subprocess.run(["/home/whj/nvidia-baseline/env-vlm/bin/python","-B","-u",str(ROOT/"verify_overlay.py")],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,min(120,DEADLINE-time.time()-5)))
result={"claim_commit":CLAIM,"epoch":EPOCH,"deadline":DEADLINE,"state":"SOFTWARE_FILES_VERIFIED",
        "files":len(seen),"bytes":sum(x["bytes"] for x in plan["files"].values()),"PLAN_SHA256":PLAN_SHA,
        "archive_SHA256":ARCHIVE_SHA,"verification_rc":p.returncode,"GPU_execution":False,
        "private_prefix":str(ROOT)}
if (ROOT/"verification.json").is_file():
    data=(ROOT/"verification.json").read_bytes();result["verification"]={"SHA256":hashlib.sha256(data).hexdigest(),"value":json.loads(data)}
with (ROOT/"receipt.json").open("x") as f:json.dump(result,f,indent=2)
print(json.dumps(result))
