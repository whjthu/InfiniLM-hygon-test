"""Read-only independent audit of original failed GPU-v3 and exact stderr frames."""
import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess

ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
LABEL="training-qwen-treasury-explicit-dev3-gpu-v3"
R=ROOT/LABEL
REQUEST="3012fc14dd404fe39ffd12ce838a1154"
CLAIM_SHA="b582f440db32dc04886cf3c687ea627a35a15251247841519173a7567bcbf149"
MF_SHA="8c6177dda718c214edbb4f509b5dde77d443861972960311648decc9d19eae78"
SPEC={"base":("1087656","a01r4n14"),"old_domain":("1087656","a01r4n16"),"domain":("1103606","f13r1n10")}
EXPECTED={"base":{"pre":"1087656.343","gpu":"1087656.346","post":"1087656.347"},"old_domain":{"pre":"1087656.344","gpu":"1087656.345","post":"1087656.348"},"domain":{"pre":"1103606.171","gpu":"1103606.172","post":"1103606.173"}}
COLS="JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
PINS={}

def require(ok,why):
 if not ok:raise ValueError(why)

def pairs(items):
 out={}
 for k,v in items:
  require(k not in out,"duplicate original JSON key");out[k]=v
 return out

def decode(raw):
 return json.loads(raw.decode("utf8"),object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError("nonfinite JSON")))

def raw(path,maximum=1024**2):
 before=path.lstat();require(stat.S_ISREG(before.st_mode) and 0<=before.st_size<=maximum,"unsafe/big original")
 b=path.read_bytes();after=path.lstat();require(all(getattr(before,k)==getattr(after,k) for k in ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns")),"original changed")
 PINS[str(path)]=hashlib.sha256(b).hexdigest();return b

def read(path):return decode(raw(path))

def host_snapshot(b,node,step):
 x=decode(b);require(set(x)=={"kind","mode","node","step","pci","pci_to_kfd","kfd_holder_count","full_fd_inventory_performed","holder_inventory_basis","boot_id","monotonic_ns"},"snapshot exact11 differs")
 require(x["kind"]=="training_qwen_device_snapshot" and x["mode"]=="host_complete" and x["node"]==node and x["step"]==step,"snapshot identity differs")
 require(x["kfd_holder_count"]==0 and type(x["kfd_holder_count"]) is int and x["full_fd_inventory_performed"] is True and x["holder_inventory_basis"]=="kernel_KFD_plus_sameUID_FD_complete" and x["pci_to_kfd"] is None,"snapshot FD/KFD differs")
 require(len(x["pci"])==len(set(x["pci"]))==8 and x["pci"]==sorted(x["pci"]) and all(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",p) for p in x["pci"]),"snapshot PCI differs")
 require(re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",x["boot_id"]) and type(x["monotonic_ns"]) is int and x["monotonic_ns"]>0,"snapshot clock differs")
 return x

def stderr_diagnosis(b):
 text=b.decode("utf8",errors="replace");lines=text.splitlines();frames=[]
 for line in lines:
  match=re.fullmatch(r'  File "([^"\r\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>.]+)',line)
  if match:
   path=match.group(1)
   # Paths/frame names only, never the potentially sensitive echoed source line.
   if path.startswith(("/public/home/","/swift-runtime/","/minicpm-transformers/","/vision/","/usr/","/opt/","<")):
    frames.append(dict(file=path,line=int(match.group(2)),function=match.group(3)))
 exceptions=[]
 for line in lines:
  match=re.match(r'^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))(?::(.*))?$',line)
  if match:
   message=(match.group(2) or "").strip()
   # Restrict exposed messages to known API/CLI exception shapes, otherwise hash only.
   safe=bool(re.fullmatch(r"(?:module '[A-Za-z0-9_.]+' has no attribute '[A-Za-z0-9_]+'|cannot import name '[A-Za-z0-9_]+' from '[A-Za-z0-9_.]+'(?: \([^\r\n]*\))?|[A-Za-z0-9_.]+\(\) got an unexpected keyword argument '[A-Za-z0-9_]+'|[A-Za-z0-9_.]+\(\) missing [0-9]+ required positional arguments?: [A-Za-z0-9_', ]+|[A-Za-z0-9_.]+\(\) takes [0-9]+ positional arguments? but [0-9]+ (?:was|were) given)",message))
   exceptions.append(dict(type=match.group(1),safe_message=message if safe else None,message_SHA256=hashlib.sha256(message.encode()).hexdigest()))
 cli_errors=[hashlib.sha256(line.encode()).hexdigest() for line in lines if "error:" in line and ("usage:" in text or "unrecognized arguments" in line)]
 return dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),traceback_frames=frames,exceptions=exceptions,
             stderr_line_count=len(lines),srun_rate_warning_count=sum(line.startswith("srun: request too frequently, retrying") for line in lines),CLI_error_line_SHA256=cli_errors,
             raw_original_remains_R=True)

def audit():
 intent,outcome=read(R/"intent.json"),read(R/"outcome.json")
 claim,mf=read(R/"source/CLAIM.json"),read(R/"source/SOURCE_FILES.json")
 require(PINS[str(R/"source/CLAIM.json")]==CLAIM_SHA and PINS[str(R/"source/SOURCE_FILES.json")]==MF_SHA,"v3 source differs")
 require(len(mf["files"])+1==24,"source24 denominator differs")
 for name,info in mf["files"].items():
  require(Path(name).name==name,"unsafe source member");b=raw(R/"source"/name);require(len(b)==info["bytes"] and hashlib.sha256(b).hexdigest()==info["SHA256"],"source member differs")
 require(intent["request"]==outcome["request"]==REQUEST and intent["publication"].startswith("99c1e7fd") and intent["claim_SHA256"]==CLAIM_SHA and outcome["original_epoch"]==intent["epoch"] and outcome["deadline"]==intent["deadline"]==intent["epoch"]+1800,"original request/epoch differs")
 require(outcome["passed"] is False and outcome["all_arm_denominator"]==324 and set(outcome["arms"])==set(SPEC),"failed complete3 original differs")
 assets=read(R/"source/MODEL_ASSETS.json")["model"];require(len(assets)==14 and sum(x["bytes"] for x in assets.values())==17545907231,"model14 identity differs")
 source_lines=raw(R/"source/native_generation.py").decode().splitlines()
 materialize_line=next(n for n,line in enumerate(source_lines,1) if "model, processor = get_model_processor(" in line)
 raw_bindings=read(R/"source/INPUT_BINDING.json");inventory=read(ROOT/"training-qwen-treasury-explicit-dev-CPU-v1/inventory.json")
 require(PINS[str(ROOT/"training-qwen-treasury-explicit-dev-CPU-v1/inventory.json")]=="07d906e3ed840d828ee41509cae21b45a754ddef0825d411ca3d8216f606b0d8" and len(inventory)==39,"inventory39 differs")
 allrows=[]
 for parent in ("1087656","1103606"):
  text=subprocess.check_output(["sacct","-n","-P","-j",parent,"-o",COLS],text=True,timeout=9)
  allrows += [row for line in text.splitlines() if len(row:=line.split("|"))==7 and row[1] in {LABEL+"-"+a+"-"+p for a in SPEC for p in ("pre","gpu","post")}]
 require(len(allrows)==9,"actual9 accounting denominator differs")
 arms={}
 for arm,(parent,node) in SPEC.items():
  folder=R/"arms"/arm;ai=read(folder/"intent.json");controller=read(folder/"controller-outcome.json");assigned=read(folder/"assigned.json");provider=read(folder/"provider.json");entry=read(folder/"input-entry-gate.json");failure=read(folder/"worker-failure.json") if (folder/"worker-failure.json").is_file() else None;work=read(folder/"work-intent.json")
  require(ai["epoch"]==intent["epoch"] and ai["deadline"]==intent["deadline"] and ai["request"]==REQUEST and controller["passed"] is False,"arm identity/result differs")
  accounting={};diagnostics={}
  for phase in ("pre","gpu","post"):
   rows=[x for x in allrows if x[1]==LABEL+"-"+arm+"-"+phase];require(len(rows)==1,"step row missing/duplicate");row=rows[0];original=read(folder/(phase+"-accounting.json"))
   require(row==original["row"]==controller["stages"][phase]["row"] and row[0] in {EXPECTED[a][phase] for a in SPEC} and (phase!="gpu" or row[0]==EXPECTED[arm][phase]) and row[5]==node,"step identity differs")
   tres=dict(x.split("=",1) for x in row[6].split(","))
   require(tres==({"cpu":"8","gres/dcu":"1","mem":"64G","node":"1"} if phase=="gpu" else {"cpu":"2","mem":"4G","node":"1"}),"step resource differs")
   require(row[2:4]==(["FAILED","1:0"] if phase=="gpu" else ["COMPLETED","0:0"]),"original actual terminal differs")
   accounting[phase]=row;diagnostics[phase]=stderr_diagnosis(raw(folder/(phase+".stderr"),65536))
  before=host_snapshot(raw(folder/"pre.stdout"),node,accounting["pre"][0]);after=host_snapshot(raw(folder/"post.stdout"),node,accounting["post"][0])
  require(read(folder/"before.json")==before and read(folder/"after.json")==after and before["pci"]==after["pci"]==assigned["pci"] and before["boot_id"]==after["boot_id"]==assigned["boot_id"],"independent post PCI/boot differs")
  require(assigned["node"]==node and assigned["step"]==accounting["gpu"][0] and assigned["visible_pci"] in before["pci"] and assigned["visible_kfd_id"]>0 and assigned["worker_pid"]>0 and assigned["worker_startticks"]>0,"assigned process/device differs")
  require(entry["actual_files"]==39 and entry["visual_rows"]==entry["text_only_rows"]==108 and entry["torch_imported"] is False and entry["member_SHA256"]==inventory,"actual bound input entry differs")
  require(provider["model_manifest_SHA256"]==PINS[str(R/"source/MODEL_ASSETS.json")] and provider["model_validation_seconds"]>0,"actual model-hash provider differs")
  require(work["original_epoch"]==intent["epoch"] and 0<work["load_and_generation_budget_seconds"]<=900 and work["actual_Slurm_cap_seconds"]<=960 and work["deadline"]<=intent["deadline"]-180,"work exceeds original epoch")
  if failure is not None:
   require(failure["arm"]==arm and failure["planned_rows"]==108 and failure["actual_training_updates"]==0,"original worker failure differs")
  raw(folder/"gpu.stdout",65536)
  originals={}
  for name in ("progress.jsonl",arm+".jsonl",arm+"-raw.jsonl","generation-outcome.json"):
   p=folder/"outputs"/name;entry_state=dict(exists=p.exists())
   if p.exists():
    b=raw(p,4*1024**2);entry_state.update(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),lines=len(b.splitlines()))
   originals[name]=entry_state
  frames=diagnostics["gpu"]["traceback_frames"]
  generate_frames=[x for x in frames if x["file"]==str(R/"source/native_generation.py") and x["function"]=="generate"]
  model_evidence="UNKNOWN_FROM_AVAILABLE_TRACE"
  if generate_frames and generate_frames[-1]["line"]<materialize_line and not originals["progress.jsonl"]["exists"]:
   model_evidence="NOT_REACHED_BY_EXACT_SOURCE_TRACE_BEFORE_MATERIALIZE"
  arms[arm]=dict(actual_accounting=accounting,independent_post_verified=True,all_PCI_boot_match=True,post_KFD_holders=0,
    owned_worker_PID=assigned["worker_pid"],owned_worker_startticks=assigned["worker_startticks"],post_source_checks_samePID_startticks=True,
    input_entry39_verified=True,model_manifest_files=14,model_manifest_bytes=17545907231,actual_model_hash_wall_seconds=provider["model_validation_seconds"],
    model_materialize_source_line=materialize_line,model_construction_evidence=model_evidence,
    worker_failure_marker_present=failure is not None,worker_error_class=failure["error_class"] if failure else None,diagnostics=diagnostics,original_outputs=originals,planned_rows=108,
    verified_generated_rows=0 if not originals[arm+"-raw.jsonl"]["exists"] else None,
    missing_rows=108 if not originals[arm+"-raw.jsonl"]["exists"] else None)
 return dict(state="ORIGINAL_GPU_V3_FAILURE_AND_ALL_POSTS_INDEPENDENTLY_VERIFIED",request=REQUEST,source_publication=intent["publication"],epoch=intent["epoch"],deadline=intent["deadline"],arms=arms,
   complete_arm_denominator=3,all_arm_denominator=324,source_files_verified=24,original_artifact_pins=PINS,
   prior_v1_324_missing_retained=True,prior_v2_324_missing_retained=True,root_corrected_reader_UNKNOWN_untouched=True,
   GPU_GO=False,promotion=False,formal_effect=False,formal_quality=False,formal_performance=False,
   current_fleet_busy_qualified=False,next="Stop GPU retries; Astra review and complete HOST-main import/API preflight with materialize interception required.")

if __name__=="__main__":
 print(json.dumps(audit(),allow_nan=False))
