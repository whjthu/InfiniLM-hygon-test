"""Verify processor-only inputs, then execute one CPU step on its owned node."""
import base64,datetime,hashlib,io,json,shlex,subprocess,sys,tarfile,time
from pathlib import Path
R=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
ROOT=R/"results/training-minicpm-hygon-processor-r2"
ASSETS=R/"results/training-minicpm-hygon-processor-v1"
label="training-minicpm-hygon-processor-r2"
node="f11r4n10"
def query(*args):return subprocess.check_output(args,text=True,timeout=12).strip()
def accounting():
    text=query("sacct","-n","-P","-j","1087656","--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw")
    return [x.split("|") for x in text.splitlines() if "|"+label+"|" in x]
assert not ROOT.exists() and not accounting()
parent=query("squeue","-h","-j","1087656","-o","%T|%D|%N").split("|")
assert parent[:2]==["RUNNING","40"] and node in query("scontrol","show","hostnames",parent[-1]).splitlines()
for line in query("squeue","--steps","-h","-j","1087656","-o","%i|%N").splitlines():
    handle,names=line.split("|",1)
    if handle not in ("1087656.batch","1087656.extern") and names not in ("","(null)"):
        assert node not in query("scontrol","show","hostnames",names).splitlines()
for item in PRIOR["files"]:
    p=ASSETS/item["name"]; data=p.read_bytes()
    assert len(data)==item["bytes"] and hashlib.sha256(data).hexdigest()==item["SHA256"]
ROOT.mkdir(mode=0o700)
with (ROOT/"stage-intent.json").open("x") as f:json.dump({"claim_commit":CLAIM,"UTC":datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
raw=sys.stdin.buffer.read(25000001);assert len(raw)<=25000000 and hashlib.sha256(raw).hexdigest()==ARCHIVE_SHA
expected={x["name"]:x for x in PLAN["files"]};seen=set()
with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as archive:
    for item in archive:
        assert item.isfile() and item.name in expected and item.name not in seen
        row=expected[item.name];assert item.size==row["bytes"]
        data=archive.extractfile(item).read();assert hashlib.sha256(data).hexdigest()==row["SHA256"]
        p=ROOT/item.name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("xb") as f:f.write(data)
        seen.add(item.name)
assert seen==set(expected)
with (ROOT/"stage-receipt.json").open("x") as f:json.dump({"claim_commit":CLAIM,"archive_SHA256":ARCHIVE_SHA,"files":expected},f)
work=ROOT/"cpu";work.mkdir(mode=0o700)
for name,item in SOURCES.items():
    assert name in ("capture_hygon_processor.py","compare_encoded_inputs.py","verify_overlay.py")
    data=base64.b64decode(item["b64"]);assert hashlib.sha256(data).hexdigest()==item["SHA256"]
    with (work/name).open("xb") as f:f.write(data)
epoch=time.time();deadline=epoch+600
with (work/"intent.json").open("x") as f:json.dump({"claim_commit":CLAIM,"epoch":epoch,"deadline":deadline,"node":node,"GPU_allocation":0},f)
dataset=R/"results/training-vlm-original-assets-v1"
shell=("set -euo pipefail; source /opt/dtk/env.sh; "
       "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 "
       "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; "
       "export PYTHONPATH=/replay/runtime-site:/swift-runtime/site:$"+"{PYTHONPATH:-}; "
       "export PROCESSOR_CLAIM="+shlex.quote(CLAIM)+" PROCESSOR_DEADLINE="+str(deadline)+"; "
        "python3 -B -m pip install --no-deps --no-index --no-compile --target /replay/runtime-site /packages/transformers-5.12.1-py3-none-any.whl; "
       "python3 -B -u /replay/verify_overlay.py; "
       "python3 -B -u /replay/capture_hygon_processor.py; exec python3 -B -u /replay/compare_encoded_inputs.py")
args=["srun","--jobid=1087656","--job-name="+label,"--exact","--nodes=1","--ntasks=1","--nodelist="+node,
      "--cpus-per-task=4","--mem=16384M","--gres=none","--immediate=5","--time=00:08:00",
      "/public/software/apps/apptainer/1.3.4/bin/apptainer","exec","--no-home","--pwd","/",
      "--bind",str(work)+":/replay:rw","--bind",str(ASSETS/"model")+":/processor-model:ro",
      "--bind",str(ASSETS)+":/processor-reference:ro","--bind",str(ROOT/"packages")+":/packages:ro","--bind",str(dataset)+":"+str(dataset)+":ro",
      "--bind",str(R/"results/training-vlm-swift-offline-runtime-v1")+":/swift-runtime:ro",
      "--bind","/opt/hyhal:/opt/hyhal:ro",str(R/"containers/dtk26-sandbox-r1"),
      "/bin/bash","--noprofile","--norc","-c",shell]
with (work/"CPU.log").open("xb") as f:
    result=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()-5))
rows=[]
for _ in range(20):
    rows=accounting()
    if len(rows)==1 and rows[0][2] in ("COMPLETED","FAILED","TIMEOUT","CANCELLED","OUT_OF_MEMORY","NODE_FAIL"):break
    time.sleep(.5)
log=(work/"CPU.log").read_bytes();assert len(log)<131072
out={"claim_commit":CLAIM,"returncode":result.returncode,"accounting":rows,"epoch":epoch,"deadline":deadline,
     "wall_seconds":time.time()-epoch,"log_SHA256":hashlib.sha256(log).hexdigest(),
     "error_tail":[x[-220:] for x in log.decode(errors="replace").splitlines() if x.startswith(("srun:","AssertionError","TypeError","RuntimeError","ModuleNotFoundError","ImportError","ValueError"))][-6:]}
for name in ("overlay.json","encoded/failure.json","encoded/receipt.json","comparison.json"):
    p=work/name
    if p.is_file():
        data=p.read_bytes();assert len(data)<65536
        out[name]={"SHA256":hashlib.sha256(data).hexdigest(),"value":json.loads(data)}
with (work/"outcome.json").open("x") as f:json.dump(out,f,indent=2)
print(json.dumps(out))
