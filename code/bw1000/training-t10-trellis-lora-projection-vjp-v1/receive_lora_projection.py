import base64,hashlib,json,subprocess,time
from pathlib import Path

root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
results=root/"results"
work=results/"training-t10-trellis-lora-projection-vjp-v1"
prior=results/"training-t10-trellis-native-construction-v1"
reference=results/"training-t10-trellis-first-reference-r1"
original_hip=results/"training-t10-trellis-hip-first-update-v1"
weights=results/"training-t10-trellis-shape-flow-cluster-v1/model"
cache=results/"training-t10-trellis-reference-cache-preflight-v1"
node="a14r1n20"
label="t10-trellis-lora-projection-vjp-v1"
def query(*cmd):
    return subprocess.check_output(cmd,text=True,timeout=12)
def accounting(name):
    return [l.split("|") for l in query("sacct","-n","-P","-j","1087656","--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw").splitlines() if "|"+name+"|" in l]
assert accounting("t10-trellis-native-construction-v1")==[["1087656.95","t10-trellis-native-construction-v1","COMPLETED","0:0","41"]]
assert accounting("t10-trellis-hip-first-update-v1")==[["1087656.98","t10-trellis-hip-first-update-v1","COMPLETED","0:0","62"]]
assert accounting("t10-trellis-block0-sdpa-vjp-v1")==[["1087656.106","t10-trellis-block0-sdpa-vjp-v1","COMPLETED","0:0","60"]]
assert (original_hip/"replay/observed.pt").is_file()
p=query("squeue","-h","-j","1087656","-o","%T|%D|%N").strip().split("|")
assert p[:2]==["RUNNING","40"] and node in query("scontrol","show","hostnames",p[2]).splitlines()
for line in query("squeue","--steps","-h","-j","1087656","-o","%i|%N").splitlines():
    handle,hosts=line.split("|")
    if handle not in ("1087656.extern","1087656.batch") and hosts not in ("","(null)"):
        assert node not in query("scontrol","show","hostnames",hosts).splitlines()
assert not work.exists() and not any(accounting(label+s) for s in ("","-pre","-post"))
work.mkdir(mode=0o700)
for name,spec in FILES.items():
    data=base64.b64decode(spec["b64"],validate=True)
    assert hashlib.sha256(data).hexdigest()==spec["sha256"]
    p=work/name
    p.parent.mkdir(exist_ok=True,parents=True)
    with p.open("xb") as f:
        f.write(data)
(work/"intent.json").write_text(json.dumps({"claim_commit":CLAIM,"node":node,"label":label})+"\n")
def run(suffix,gres,cores,mem,limit,timeout,command):
    name=label+suffix
    args=["srun","--jobid=1087656","--job-name="+name,"--exact","--nodes=1","--ntasks=1",
          "--nodelist="+node,f"--cpus-per-task={cores}",f"--mem={mem}M","--gres="+gres,"--immediate=5","--time="+limit,*command]
    path=work/(name+".log")
    with path.open("xb") as log:
        p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    rows=[]
    for _ in range(20):
        rows=accounting(name)
        if len(rows)==1 and rows[0][2] in ("COMPLETED","FAILED","TIMEOUT","CANCELLED","OUT_OF_MEMORY","NODE_FAIL"):
            break
        time.sleep(.6)
    raw=path.read_bytes()
    objects=[json.loads(l) for l in raw.splitlines() if l.startswith(b"{")]
    return {"rc":p.returncode,"accounting":rows,"log_sha256":hashlib.sha256(raw).hexdigest(),"objects":objects,
            "errors":[s[:230] for s in raw.decode(errors="replace").splitlines() if "Error:" in s or s.startswith("srun:")][-10:]}
def clean(suffix):
    result=run(suffix,"none",2,4096,"00:02:00",135,
               ["/usr/local/python3.12/bin/python3","-B",str(work/"snapshot.py")])
    r,=result["objects"]
    assert result["rc"]==0 and result["accounting"][0][2:4]==["COMPLETED","0:0"]
    assert r["step"]==result["accounting"][0][0] and r["kfd_holder_count"]==0
    return result,r
def container(command,gpu):
    flags="" if gpu else "export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; "
    shell=("set -euo pipefail; source /opt/dtk/env.sh; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "
           "OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; "+flags+command)
    return ["/public/software/apps/apptainer/1.3.4/bin/apptainer","exec",*(["--rocm"] if gpu else []),
            "--no-home","--pwd","/","--bind",f"{work}:/replay:rw","--bind",f"{prior}:/construct:ro",
            "--bind",f"{reference}:/reference:ro","--bind",f"{weights}:/weights:ro","--bind",f"{cache}:/cache:ro",
            "--bind",f"{original_hip}:/original:ro","--bind","/opt/hyhal:/opt/hyhal:ro",str(root/"containers/dtk26-sandbox-r1"),
            "/bin/bash","--noprofile","--norc","-c",shell]
pre,before=clean("-pre")
gpu=run("","dcu:1",8,64000,"00:07:00",445,container(
    "exec python3 -B -u /replay/lora_projection_gpu_entry.py --reference /reference --weights /weights --cache /cache/train-cache.npz "
    "--source /construct/upstream --source-inventory /construct/A100_SOURCE_INVENTORY.json --original-hip /original/replay/observed.pt --output /replay/diagnostic",True))
post,after=clean("-post")
assert after["boot_id"]==before["boot_id"] and after["pci"]==before["pci"]
result={"claim_commit":CLAIM,"pre":pre,"gpu":gpu,"post":post}
if (work/"assigned.json").is_file():
    result["assigned"]=json.loads((work/"assigned.json").read_bytes())
    assert result["assigned"]["pci"] in before["pci"] and result["assigned"]["step"]==gpu["accounting"][0][0]
if gpu["rc"]==0 and gpu["accounting"][0][2:4]==["COMPLETED","0:0"]:
    assert result["assigned"]["visible_cards"]==1 and len(gpu["objects"])==1
(work/"outcome.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
