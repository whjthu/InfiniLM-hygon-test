import base64,hashlib,json,os,pathlib,subprocess,time,signal
R=pathlib.Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
OLD=pathlib.Path("/public/home/aclsylqidf/hygon-inference-20261013")
LABEL="GLM474-paired-host-preflight-v1"
W=R/"results"/LABEL
PARENT="1103606"
NODES={"f13r1n05","f13r1n07"}
C="JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList"
def query(*argv):
    return subprocess.check_output(argv,text=True,timeout=8).splitlines()
def account():
    return [s.split("|") for s in query("sacct","-n","-P","-j",PARENT,"--format="+C) if "|"+LABEL+"|" in s]
def save(name,v):
    with (W/name).open("x") as f:
        json.dump(v,f,indent=2)
        f.flush()
        os.fsync(f.fileno())
def expand(hosts):
    return set(query("scontrol","show","hostnames",hosts))
assert not W.exists() and not account()
state,count,hosts=query("squeue","-h","-j",PARENT,"-o","%T|%D|%N")[0].split("|")
assert (state,count)==("RUNNING","40") and NODES<=expand(hosts)
for item in query("squeue","--steps","-h","-j",PARENT,"-o","%i|%N"):
    step,hosts=item.split("|")
    if step not in (PARENT+".batch",PARENT+".extern"):
        assert hosts not in ("","(null)","None") and not (expand(hosts)&NODES)
W.mkdir(mode=0o700)
S=W/"src"
S.mkdir()
save("intent.json",dict(request_id=REQUEST,commit=COMMIT,source_pins={name:v[0] for name,v in FILES.items()},GPU_cards=0))
for name,(sha,b64) in FILES.items():
    assert pathlib.Path(name).name==name
    data=base64.b64decode(b64,validate=True)
    assert hashlib.sha256(data).hexdigest()==sha
    with (S/name).open("xb") as f:f.write(data)
scope="/refactor/results/"+LABEL
shell='set -euo pipefail; src="$1"; root="$2"; s="$(mktemp -d /dev/shm/glm474-pre-XXXXXXXX)"; trap \'rmdir "$s"\' EXIT; mkdir -m700 "$root/tmp-$SLURM_PROCID"; export APPTAINERENV_CUDA_VISIBLE_DEVICES=-1 APPTAINERENV_HIP_VISIBLE_DEVICES=-1 APPTAINERENV_ROCR_VISIBLE_DEVICES=-1 APPTAINERENV_TMPDIR=/tmp; /public/software/apps/apptainer/1.3.4/bin/apptainer exec --rocm --home "$3/build-deps/root:/root" --bind "$3:/old:ro" --bind "$4:/refactor:ro" --bind "$root:$5:rw" --bind "$src:/paired-worker:ro" --bind "$s:/local" --bind "$root/tmp-$SLURM_PROCID:/tmp" --bind /opt/hyhal:/opt/hyhal:ro --bind /public/home/aclsylqidf/shared:/shared:ro "$4/containers/dtk26-sandbox-r1" /bin/bash --noprofile --norc -c \'source /paired-worker/ENV.sh; exec /usr/bin/python3 -B -u /paired-worker/preflight.py "$1"\' preflight "$5"; rm -f -- "$s/preflight"'
cmd=["srun","--jobid="+PARENT,"--job-name="+LABEL,"--exact","--nodes=2","--ntasks=2","--ntasks-per-node=1","--distribution=block:block","--nodelist="+",".join(sorted(NODES)),"--cpus-per-task=4","--mem=8192M","--gres=none","--immediate=5","--time=3","--kill-on-bad-exit=1","/bin/bash","-c",shell,"preflight",str(S),str(W),str(OLD),str(R),scope]
with (W/"CPU.log").open("xb") as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save("srun.json",dict(pid=child.pid))
    try: rc=child.wait(timeout=155)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGTERM)
        try: rc=child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=5)
import sys
sys.path.insert(0,str(S))
from slurm_accounting import wait_terminal
def row():
    rows=account()
    assert len(rows)<=1
    return rows[0] if rows else None
terminal=wait_terminal(row,time.monotonic()+22,interval=.5,max_observations=28)
ranks=[]
for rank in range(2):
    f=W/f"rank-{rank}.json"
    if f.is_file():ranks.append(json.loads(f.read_bytes()))
step=terminal["row"]
passed=bool(rc==0 and step and step[2:4]==["COMPLETED","0:0"] and {r["node"] for r in ranks}==NODES and all(r["step"]==step[0] for r in ranks))
raw=(W/"CPU.log").read_bytes()
out=dict(request_id=REQUEST,commit=COMMIT,passed=passed,accounting_columns=C.split(","),terminal=terminal,
         ranks=ranks,log_SHA256=hashlib.sha256(raw).hexdigest(),log_error_classes=[x for x in ("AssertionError","ValueError","Permission denied","No such file","ModuleNotFoundError","FATAL","Error") if x.encode() in raw],GPU_cards=0,rc=rc)
save("outcome.json",out)
print(json.dumps(out),flush=True)
