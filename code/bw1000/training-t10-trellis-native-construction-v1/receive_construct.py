import base64,hashlib,io,json,subprocess,tarfile,time
from pathlib import Path,PurePosixPath

root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
results=root/"results"
work=results/"training-t10-trellis-native-construction-v1"
label="t10-trellis-native-construction-v1"
node="a14r1n20"
def query(*cmd):
    return subprocess.check_output(cmd,text=True,timeout=12)
def accounting():
    return [l.split("|") for l in query("sacct","-n","-P","-j","1087656","--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw").splitlines() if "|"+label+"|" in l]
p=query("squeue","-h","-j","1087656","-o","%T|%D|%N").strip().split("|")
assert p[:2]==["RUNNING","40"] and node in query("scontrol","show","hostnames",p[2]).splitlines()
assert not accounting() and not work.exists()
for line in query("squeue","--steps","-h","-j","1087656","-o","%i|%N").splitlines():
    handle,hosts=line.split("|")
    if handle not in ("1087656.extern","1087656.batch") and hosts not in ("","(null)"):
        assert node not in query("scontrol","show","hostnames",hosts).splitlines()
weights=results/"training-t10-trellis-shape-flow-cluster-v1/model"
cache=results/"training-t10-trellis-reference-cache-preflight-v1"
reference=results/"training-t10-trellis-first-reference-r1"
assert weights.is_dir() and (cache/"train-cache.npz").is_file() and (reference/"receipt.json").is_file()
work.mkdir(mode=0o700)
for name, spec in FILES.items():
    raw=base64.b64decode(spec["b64"],validate=True)
    assert hashlib.sha256(raw).hexdigest()==spec["sha256"]
    path=work/name
    path.parent.mkdir(exist_ok=True,parents=True)
    with path.open("xb") as f:
        f.write(raw)
archive=(work/"shape-flow-source.tar.gz").read_bytes()
baseline=json.loads((work/"A100_SOURCE_INVENTORY.json").read_bytes())
excluded="trellis2/pipelines/rembg/BiRefNet.py"
with tarfile.open(fileobj=io.BytesIO(archive),mode="r:gz") as tar:
    members=tar.getmembers()
    assert len(members)==92 and {m.name for m in members}==set(baseline)-{excluded}
    for m in members:
        assert m.isfile() and not PurePosixPath(m.name).is_absolute() and ".." not in PurePosixPath(m.name).parts
        data=tar.extractfile(m).read()
        assert hashlib.sha256(data).hexdigest()==baseline[m.name]
        path=work/"upstream"/m.name
        path.parent.mkdir(exist_ok=True,parents=True)
        with path.open("xb") as f:
            f.write(data)
(work/"intent.json").write_text(json.dumps({"claim_commit":CLAIM,"label":label,"node":node})+"\n")
args=["srun","--jobid=1087656","--job-name="+label,"--exact","--nodes=1","--ntasks=1",
      "--nodelist="+node,"--cpus-per-task=4","--mem=24000M","--gres=none","--immediate=5","--time=00:04:00",
      "/public/software/apps/apptainer/1.3.4/bin/apptainer","exec","--no-home","--pwd","/",
      "--bind",f"{work}:/replay:ro","--bind",f"{reference}:/reference:ro","--bind",f"{weights}:/weights:ro",
      "--bind",f"{cache}:/cache:ro",str(root/"containers/dtk26-sandbox-r1"),
      "/bin/bash","--noprofile","--norc","-c",
      "set -euo pipefail; source /opt/dtk/env.sh; export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=-1 "
      "HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; "
      "exec python3 -B -u /replay/tools/hygon_trellis_first_step.py --reference /reference --weights /weights "
      "--cache /cache/train-cache.npz --source /replay/upstream --source-inventory /replay/A100_SOURCE_INVENTORY.json --construct-only"]
with (work/"run.log").open("xb") as log:
    proc=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=250)
rows=[]
for _ in range(15):
    rows=accounting()
    if len(rows)==1 and rows[0][2] in ("COMPLETED","FAILED","TIMEOUT","CANCELLED","OUT_OF_MEMORY"):
        break
    time.sleep(.6)
raw=(work/"run.log").read_bytes()
answer={"accounting":rows,"rc":proc.returncode,"claim_commit":CLAIM,"node":node,
        "source":{n:v["sha256"] for n,v in FILES.items()},
        "output":[json.loads(l) for l in raw.splitlines() if l.startswith(b"{")],
        "log_sha256":hashlib.sha256(raw).hexdigest(),
        "error_tail":[s[:220] for s in raw.decode(errors="replace").splitlines() if "Error:" in s or s.startswith("srun:")][-10:]}
(work/"outcome.json").write_text(json.dumps(answer,indent=2)+"\n",encoding="utf-8")
print(json.dumps(answer))
