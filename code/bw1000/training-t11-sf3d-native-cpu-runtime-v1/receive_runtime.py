import hashlib,json,shlex,shutil,subprocess,sys,time
from pathlib import Path
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
work=root/"results/training-t11-sf3d-native-cpu-runtime-v1"
staged=root/"results/training-t11-sf3d-source-stage-v1"
node="f11r3n02"
label="t11-sf3d-native-cpu-runtime-v1"
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
def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()
source_manifest=staged/"SOURCE_FILES.json"
assert sha(source_manifest)=="ba362e90a407d761a5b6fba279b5e4972684b89c33dd352400c3e7f758837acd"
files=json.loads(source_manifest.read_bytes())
assert len(files)==70
for name,row in files.items():
    path=staged/"source"/name
    assert path.stat().st_size==row["bytes"] and sha(path)==row["SHA256"],name
work.mkdir(mode=0o700)
inputs=work/"inputs"
inputs.mkdir()
assert len(PINS)==16
for row in PINS:
    name=row["filename"]
    assert Path(name).name==name and name.endswith((".whl",".tar.gz"))
    assert 0<row["bytes"]<=25*1024*1024
    data=sys.stdin.buffer.read(row["bytes"])
    assert len(data)==row["bytes"] and hashlib.sha256(data).hexdigest()==row["SHA256"]
    (inputs/name).write_bytes(data)
assert sys.stdin.buffer.read(1)==b""
for name in ("texture_baker","uv_unwrapper"):
    shutil.copytree(staged/"source"/name,work/name)
(work/"smoke.py").write_text(PROBE,encoding="utf-8")
(work/"install.sh").write_text(INSTALL,encoding="utf-8")
(work/"intent.json").write_text(json.dumps({"node":node,"label":label,"claim_commit":CLAIM,"manifest_sha":sha(source_manifest)})+"\n",encoding="utf-8")
cmd=["srun","--jobid=1087656","--job-name="+label,"--exact","--nodes=1","--ntasks=1","--cpus-per-task=4",
     "--nodelist="+node,"--gres=none","--mem=16384M","--immediate=5","--time=00:08:00",
     "/public/software/apps/apptainer/1.3.4/bin/apptainer","exec","--no-home","--pwd","/",
     "--bind",str(work)+":/runtime:rw","--bind",str(staged/"source")+":/sf3d-source:ro",
     "--bind","/opt/hyhal:/opt/hyhal:ro",str(root/"containers/dtk26-sandbox-r1"),
     "/bin/bash","--noprofile","--norc","/runtime/install.sh"]
with (work/"run.log").open("xb") as log:
    proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=500)
rows=[]
for _ in range(15):
    rows=accounting()
    if len(rows)==1 and rows[0][2] in ("COMPLETED","FAILED","TIMEOUT","CANCELLED","OUT_OF_MEMORY"):
        break
    time.sleep(.6)
raw=(work/"run.log").read_bytes()
results=[json.loads(l) for l in raw.splitlines() if l.startswith(b"{")]
summary={"rc":proc.returncode,"accounting":rows,"claim_commit":CLAIM,"node":node,"smoke":results,
         "log_sha256":hashlib.sha256(raw).hexdigest(),"log_tail":[s.decode(errors="replace")[:300] for s in raw.splitlines()[-18:]]}
(work/"receipt.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary),flush=True)
