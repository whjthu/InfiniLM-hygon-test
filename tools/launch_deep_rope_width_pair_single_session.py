"""One finite admission and launch exchange for the frozen DeepSeek device comparison."""

import base64,hashlib,json,os,re,select,subprocess,sys,time,uuid
from pathlib import Path
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
results=root/"results"
label="deep-rope-width-device-pair-single-session-v2"
scope=results/label
source_root=results/"Deep-inverse-output-rope-prefix-width-pair-gfx936-CPU-v1-survivor-n13-source"
sys.path.insert(0,str(source_root/"tools"))
from control_deep_rope_prefix_width_survivor import verify_device,check_membership
from training_dispatch_allocation import allocation,SURVIVOR_LEDGER
_,groups,_=allocation(SURVIVOR_LEDGER.read_bytes())
claim_pin=sys.argv[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,b):
 with p.open("xb") as f:
  f.write(b)
  f.flush()
  os.fsync(f.fileno())
def obj(p):
 b=p.read_bytes()
 return {"SHA256":sha(b),"bytes":len(b)}
def save_json(path,d):
 b=json.dumps(d,sort_keys=True,allow_nan=False).encode()
 save(path,b)
 return sha(b)
def parse_claim(b):
 assert sha(b)==claim_pin
 c=json.loads(b)
 assert c["label"]==label and re.fullmatch(r"[0-9a-f]{32}",c["request"])
 assert c["gpus"]==1 and c["GPU_GO"] is True and c["replay_allowed"] is False
 return c
def accounting(parent):
 q=subprocess.check_output(["sacct","-n","-P","-j",parent,"-S","2026-11-22T00:00:00","-o","JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"],text=True,timeout=7)
 return [line.split("|") for line in q.splitlines() if line.strip()]
def require_sources(c, inspector):
 assert obj(inspector)=={k:c["source"][k] for k in ("bytes","SHA256")}
 link=results/"deep-rope-width-device-link-cpu-v1"
 assert obj(link/"artifacts/compare")==c["executable"]
 assert obj(link/"artifacts/original-result.json")["SHA256"]=="defbc0ab45f5b29d77ba7d365593dfbb0e5e586b45d14d930c1ff9c29cd884c9"
 rows=accounting(c["parent"])
 assert not any(row[1]==label for row in rows)
 assert any(row[0]=="1103606.1363" and row[2:4]==["COMPLETED","0:0"] and row[5]==c["node"] and "gres" not in row[6] for row in rows)
 return link
def receive_line(limit):
 raw=sys.stdin.buffer.readline(limit+1)
 assert 0<len(raw)<=limit and raw.endswith(b"\n")
 return json.loads(raw)

body=receive_line(20000)
claim_bytes=base64.b64decode(body["claim"],validate=True)
c=parse_claim(claim_bytes)
parent,node=(c[k] for k in ("parent","node"))
source=base64.b64decode(body["source"],validate=True)
assert sha(source)==c["source"]["SHA256"] and len(source)==c["source"]["bytes"]
prior=results/"deep-rope-width-device-pair-v1"
assert not (prior/"intent.json").exists()
assert not any(row[1]=="deep-rope-width-device-pair-v1" for row in accounting(parent))
scope.mkdir(mode=0o700)
(scope/"artifacts").mkdir(mode=0o700)
save(scope/"CLAIM.json",claim_bytes)
save(scope/"run_gpu.py",source)
link=require_sources(c,scope/"run_gpu.py")
device=verify_device(groups,uuid.uuid4().hex)
assert device["pci_SHA256"]==sha(json.dumps(sorted(c["observed_node_PCIs"]),separators=(",",":")).encode())
observed=time.monotonic()
adm_sha=save_json(scope/"admission.json",dict(device=device,request=c["request"],monotonic=observed))
print(json.dumps(dict(request=c["request"],admission_SHA256=adm_sha,device=device)),flush=True)
assert select.select([sys.stdin.buffer],[],[],50)[0], "local dispatch acknowledgement expired"
ack=receive_line(2048)
assert set(ack)=={"request","admission_SHA256","handoff_SHA256"}
assert (ack["request"],ack["admission_SHA256"])==(c["request"],adm_sha)
assert re.fullmatch(r"[0-9a-f]{64}",ack["handoff_SHA256"])
assert 0<=time.monotonic()-observed<55
check_membership(groups)
epoch=time.time()
save_json(scope/"intent.json",dict(request=c["request"],label=label,node=node,parent=parent,epoch=epoch,gpus=1,
                                 deadline=epoch+c["slurm_seconds"],admission_SHA256=adm_sha,handoff_SHA256=ack["handoff_SHA256"]))
shell=("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; unset LD_PRELOAD LD_AUDIT; "
       "export LC_ALL=C; exec /usr/bin/python3 -I -S -B /audit/run_gpu.py")
args=["srun","--jobid="+parent,"--job-name="+label,"--exact","--immediate=5","--nodes=1","--ntasks=1","--nodelist="+node,
 "--cpus-per-task=4","--mem=8192M","--gres=dcu:1","--time=3","/public/software/apps/apptainer/1.3.4/bin/apptainer",
 "exec","--rocm","--contain","--no-home","--pwd","/","--bind",str(scope)+":/audit:ro",
 "--bind",str(scope/"artifacts")+":/audit-out:rw","--bind",str(link)+":/original-link:ro","--bind","/opt/hyhal:/opt/hyhal:ro",
 str(root/"containers/dtk26-sandbox-r1"),"/bin/bash","--noprofile","--norc","-c",shell]
with (scope/"step.stdout").open("xb") as out,(scope/"step.stderr").open("xb") as err:
 r=subprocess.run(args,stdout=out,stderr=err,timeout=c["client_seconds"])
selected=[]
for _ in range(6):
 selected=[row for row in accounting(parent) if row[1]==label]
 if len(selected)==1 and selected[0][2] in ("COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL"):
  break
 time.sleep(0.8)
valid_terminal=len(selected)==1 and selected[0][2] in ("COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL")
identity=scope/"artifacts/device-identity.json"
samples=scope/"artifacts/samples.jsonl"
identity_summary=json.loads(identity.read_bytes()) if identity.is_file() else None
sample_rows=[json.loads(line) for line in samples.read_bytes().splitlines()] if samples.is_file() else []
err=(scope/"step.stderr").read_bytes()
known_errors=[s[:200] for s in err.decode("utf-8","replace").splitlines() if re.fullmatch(r"(?:ValueError: |srun: error: |Expected |required |device |output |original RoPE |strict |visible |select |CUDA |measurement |elapsed |create |upload |download |allocate |[Nn]onpositive )[A-Za-z0-9 ._:'+()/-]{1,170}",s)]
resp=dict(request=c["request"],valid_terminal=valid_terminal,accounting=selected,client_rc=r.returncode,device=identity_summary,
          samples=sample_rows,sample_pin=obj(samples) if samples.is_file() else None,stderr=obj(scope/"step.stderr"),safe_errors=known_errors[-4:])
save_json(scope/"host-result.json",resp)
print(json.dumps(resp,allow_nan=False),flush=True)
raise SystemExit(int(r.returncode!=0 or not valid_terminal))
