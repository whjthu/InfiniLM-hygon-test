"""Exact R27 HOST admission, single launch intent and independent terminal read."""

from base64 import b64decode
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time


UID=154459
PARENT,NODE="1103606","f13r1n12"
REQUEST="537844da9e9b470993209b74ad45c7f7"
PUBLICATION="c6e1b4af4de26902e51a1f7549e53543cf773c8e"
LABEL="training-gemma-usgs-r26-native-image-CPU-v1"
JOB=LABEL+"-"+REQUEST[:12]
STANDBY_STEP="1103606.3274"
STANDBY_LABEL="training80-dcu-standby-3beb6b8b5ea4-f13r1n12"
STANDBY_ACCOUNT_COLUMNS="JobIDRaw,JobName%100,State,ExitCode,NodeList,AllocTRES"
RESULTS=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
RUN=RESULTS/LABEL/REQUEST
CLAIM="fa3a7ba4b34a4047cc39d7aead3465e24c7225da52e21367135731da6f203997"
MANIFEST="0dad2a737e821ae08992351c6c4aaedd2cbf46c8a1838c340d587abea835f7ce"
DEPLOY="de61163a6876c4e409c250ee28db522e72db3842d64bf751905ba7bdee7c1e84"
ADMISSION="cbdb51098763e9691c0a21fe57163b08fce05b1d95c2a76c73e906edf97368ef"
SOURCE_ARCHIVE="e6e8c1e90c6a6520f773d1055fe0ed6b0017798e2b103e310421198128f7e51b"
DATA_ARCHIVE="bedefd9f8f4b2bf3383ea9714647b68c6cca005c83848a8a69a0bf4348f6cd7a"
STAGES=(
    (482,"3fa32d19083af547cc868cc7e9e8939f172de95d07e5c46b99340905b78a868b","c278f82c239a7842e7b3368f31254319"),
    (615,"8c76da3b5a1e1bc7ce46f4c35c3e51fca8f597e9dbecc06b4ab08da9468e68c6","6fcc0840b6f43a66d179e5718ad9e303"),
)
WORK=120
INTENT="HOST_LAUNCH_INTENT.json"
HOST_RESULT="HOST_SRUN_RETURN.json"
LOCAL_ENV={"TZ":"UTC","SLURM_TIME_FORMAT":"%Y-%m-%dT%H:%M:%S"}
TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","NODE_FAIL","OUT_OF_MEMORY","PREEMPTED","BOOT_FAIL","DEADLINE","REVOKED"}


class Rejected(ValueError):
    """A fixed diagnostic with no raw target contents."""


def need(ok,code):
    if not ok:
        raise Rejected(code)


def doc(obj):
    return (json.dumps(obj,sort_keys=True,allow_nan=False)+"\n").encode()


def pin(raw):
    return {"SHA256":sha256(raw).hexdigest(),"bytes":len(raw)}


def hash_read(path,size=None,digest=None,cap=16384):
    a=path.lstat()
    need(path.resolve(strict=True)==path and stat.S_ISREG(a.st_mode) and 0<a.st_size<=cap
         and (size is None or size==a.st_size),"BOUNDED_CANONICAL_METADATA")
    with path.open("rb") as stream:
        raw=stream.read(cap+1)
    b=path.lstat()
    features=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(len(raw)==a.st_size and features(a)==features(b)
         and (digest is None or pin(raw)["SHA256"]==digest),"EXACT_METADATA_PIN")
    return raw


def save(root,name,obj):
    path=root/name
    raw=doc(obj)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return pin(raw)


def claim(raw):
    need(pin(raw)["SHA256"]==CLAIM,"EXACT_PUBLISHED_CLAIM_BYTES")
    value=json.loads(raw)
    need((value["request"],value["parent"],value["candidate_node"],value["Slurm_label"],
          value["relative_result_namespace"])==(REQUEST,PARENT,NODE,JOB,LABEL+"/"+REQUEST)
         and value["max_target_steps"]==1 and value["replay_allowed"] is value["training_ready"] is False
         and value["target_GPU"]==0 and value["placement"]["CPU"]==4 and value["placement"]["mem_MiB"]==8192
         and value["placement"]["GRES"]=="none" and value["placement"]["original_compute_budget_s"]==WORK
         and value["placement"]["runtime_cpu_shell_explicitly_unsets_inherited_python_cache_prefix"] is True
         and value["input"]["source_manifest_sha"]==MANIFEST and value["input"]["source_archive_sha"]==SOURCE_ARCHIVE
         and value["input"]["private_data_archive_sha"]==DATA_ARCHIVE
         and value["input"]["train_only"] is True,"EXACT_PUBLISHED_CPU_SCOPE")
    return value


def stage(root):
    for i,(size,digest,request) in enumerate(STAGES):
        label=f"stage{i}.json"
        row=json.loads(hash_read(root/label,size,digest))
        need((row["request"],row["owner"],row["publication"],row["part"],row["state"])==
             (request,REQUEST,PUBLICATION,i,"STAGED") and row["slurm_steps"]==row["GPU_steps"]==0,
             "AUTHENTICATED_STAGE_SCOPE")
        if i==0:
            need(row["source_archive_SHA256"]==SOURCE_ARCHIVE,"EXACT_STAGE0_SOURCE")
        else:
            need(row["source_members"]==6 and row["data_members"]==15
                 and row["source_manifest_SHA256"]==MANIFEST and row["deployment_SHA256"]==DEPLOY
                 and row["data_archive_SHA256"]==DATA_ARCHIVE,"EXACT_STAGE1_SEALED_SOURCE_AND_DATA")
    need(all((root/name).is_dir() and (root/name).resolve(strict=True)==root/name for name in ("src","data","out")),
         "EXACT_STAGED_DIRECTORIES")
    hash_read(root/"src/SOURCE_FILES.json",643,MANIFEST)
    raw=hash_read(root/"src/admission.py",6426,ADMISSION)
    code=hash_read(root/"deployment.py",4584,DEPLOY)
    import types
    original=sys.modules.get("admission")
    module=types.ModuleType("admission")
    exec(compile(raw,"published_R27_admission","exec"),module.__dict__)
    sys.modules["admission"]=module
    try:
        deployed={}
        exec(compile(code,"published_R27_deployment","exec"),deployed)
    finally:
        if original is None:
            del sys.modules["admission"]
        else:
            sys.modules["admission"]=original
    return deployed


def utc(text):
    return datetime.strptime(text,"%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def only(rows):
    need(len(rows)==1,"EXACT_ONE_CONTROLLER_RECORD")
    return rows[0]


def fields(line):
    return dict(item.split("=",1) for item in line.split() if "=" in item)


def accounting(query,deployed):
    rows=[line.split("|") for line in query(*deployed["accounting_read_argv"](PARENT)[3:])]
    need(all(len(row)==9 for row in rows),"EXACT_NINE_ACCOUNTING_COLUMNS")
    relevant=[row for row in rows if row[1]==JOB]
    need(len(relevant)<=1 and all(re.fullmatch(PARENT+r"\.[0-9]+",row[0]) and row[5]==NODE for row in relevant),
         "EXACT_OWNER_LABEL_ACCOUNTING_IDENTITY")
    return relevant


def standby_resources(value):
    entries=value.split(",")
    tres=dict(item.split("=",1) for item in entries)
    need(len(entries)==len(tres)==4 and tres.get("cpu")=="8" and tres.get("mem") in ("16G","16384M")
         and tres.get("gres/dcu")=="8" and tres.get("node")=="1","EXACT_3274_INCUMBENT_RESOURCES")
    return 8,16384


def standby_accounting(query):
    rows=[line.split("|") for line in query("sacct","-j",STANDBY_STEP,"-nP","--format",STANDBY_ACCOUNT_COLUMNS)]
    need(all(len(row)==6 for row in rows),"EXACT_3274_ACCOUNTING_COLUMNS")
    row=only([row for row in rows if row[0]==STANDBY_STEP])
    need(row[1]==STANDBY_LABEL and row[4]==NODE,"EXACT_3274_ACCOUNTING_IDENTITY")
    resources=standby_resources(row[5])
    return row[2].split()[0].rstrip("+"),row[3],resources


def fresh_slots(query,now):
    aliases={}
    def hosts(placed):
        need(placed not in ("","(null)","None","Unknown","N/A"),"EXACT_LIVE_NODELIST")
        if re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",placed):
            return {placed}
        if placed not in aliases:
            aliases[placed]=set(query("scontrol","show","hostnames",placed))
        need(aliases[placed],"EXACT_LIVE_NODELIST")
        return aliases[placed]
    job=fields(only(query("scontrol","show","job",PARENT,"-o")))
    parent,state,queued=only(query("squeue","-h","-j",PARENT,"-o","%i|%T|%N")).split("|")
    nodes=hosts(job["NodeList"])
    end=utc(job["EndTime"])
    need(job["JobId"]==parent==PARENT and job["JobState"]==state=="RUNNING"
         and job["UserId"].endswith(f"({UID})") and job["NumNodes"]=="40" and len(nodes)==40
         and NODE in nodes and nodes==hosts(queued) and "gres/dcu=320" in job["AllocTRES"].split(",")
         and now+WORK<=end,"EXACT_ACTIVE_OWNED_PARENT_WINDOW")
    machine=fields(only(query("scontrol","show","node",NODE,"-o")))
    need(machine["NodeName"]==NODE and int(machine["CPUTot"])>=4 and int(machine["RealMemory"])>=8192
         and not re.search("DOWN|DRAIN|FAIL|MAINT|NO_RESPOND",machine["State"]),"LIVE_NODE_CPU_MEMORY")
    limits=[(cpu,mem) for line in query("scontrol","-d","show","job",PARENT)
            for names,cpu,mem in re.findall(r"\bNodes=(\S+)\s+CPU_IDs=(\S+)\s+Mem=(\S+)",line) if NODE in hosts(names)]
    slots,memory=only(limits)
    cpu_ids=set()
    for item in slots.split(","):
        m=re.fullmatch(r"([0-9]+)(?:-([0-9]+))?",item)
        need(m is not None,"PARENT_CPU_MAP")
        lo,hi=int(m[1]),int(m[2] or m[1])
        need(0<=lo<=hi<=16384,"PARENT_CPU_MAP")
        cpu_ids.update(range(lo,hi+1))
    m=re.fullmatch("([0-9]+)([MGT]?)",memory)
    need(m is not None,"PARENT_MEMORY_UNITS")
    mem=int(m[1])*{"":1,"M":1,"G":1024,"T":1048576}[m[2]]
    need(len(cpu_ids)>=4 and mem>=8192,"PARENT_NODE_FOUR_CPU_EIGHT_GIB")
    controlled=[fields(line) for line in query("scontrol","show","step",PARENT,"-o")]
    def scope(rows):
        occupied=[]
        for step,name,where in rows:
            suffix=step.removeprefix(PARENT+".")
            need(step==PARENT+"."+suffix and name!=JOB,"LIVE_REQUEST_LABEL_ALREADY_EXISTS")
            if suffix in ("batch","extern"):
                continue
            need(suffix.isdecimal(),"UNCLASSIFIED_PARENT_STEP")
            members=hosts(where)
            if step==STANDBY_STEP or name==STANDBY_LABEL or NODE in members:
                need((step,name)==(STANDBY_STEP,STANDBY_LABEL) and members=={NODE},
                     "UNKNOWN_NUMERIC_STEP_ON_TARGET_NODE")
                occupied.append(step)
        need(len(occupied)<=1,"DUPLICATE_STANDBY_STEP_IN_SCHEDULER_VIEW")
        return occupied
    active=[row for row in controlled if row.get("State","").split("+")[0] not in TERMINAL]
    controller_scope=scope((row["StepId"],row.get("Name",""),row["NodeList"]) for row in active)
    queue_scope=scope(tuple(line.split("|")) for line in query("squeue","--steps","-h","-j",PARENT,"-o","%i|%j|%N"))
    need(not queue_scope or queue_scope==controller_scope,"STANDBY_QUEUE_CONTRADICTS_CONTROLLER")
    accounted_state,accounted_exit,accounted_resources=standby_accounting(query)
    used_cpu=used_mem=0
    if controller_scope:
        own=only([row for row in active if row["StepId"]==STANDBY_STEP])
        need(own["State"]=="RUNNING" and accounted_state=="RUNNING" and accounted_exit=="0:0"
             and standby_resources(own["TRES"])==accounted_resources,"STANDBY_CONTROLLER_ACCOUNTING_DISAGREEMENT")
        used_cpu,used_mem=accounted_resources
    else:
        need(accounted_state in TERMINAL,"STANDBY_ACCOUNTING_ACTIVE_WITHOUT_CONTROLLER")
    available_cpu,available_mem=len(cpu_ids)-used_cpu,mem-used_mem
    need(available_cpu>=4 and available_mem>=8192,"AVAILABLE_FOUR_CPU_EIGHT_GIB_AFTER_3274")
    return {"controller_and_queue_confirmed":True,"parent_end_epoch":end,"parent_node_CPU_slots":len(cpu_ids),
            "parent_node_memory_MiB":mem,"available_CPU_slots_after_standby":available_cpu,
            "available_memory_MiB_after_standby":available_mem,"owned_standby_step":STANDBY_STEP if controller_scope else None,
            "standby_controller_accounting_confirmed":bool(controller_scope),"standby_squeue_visible":bool(queue_scope),
            "other_target_numbered_steps":0}


def protect(deployed,epoch):
    command=deployed["argv"](PARENT,NODE,REQUEST,PUBLICATION,epoch,MANIFEST)
    need(command[0]=="srun" and [arg for arg in command if arg.startswith("--gres")]==["--gres=none"]
         and "--overlap" not in command and "--cpus-per-task=4" in command and "--mem=8192M" in command
         and command[-1].count("source /opt/dtk/env.sh; export ")==1,"FROZEN_CPU_ARGV")
    command[-1]=command[-1].replace("source /opt/dtk/env.sh; export ",
                                  "source /opt/dtk/env.sh; unset PYTHONPYCACHEPREFIX; export ",1)
    return command


def host_environment():
    env=dict(os.environ,**LOCAL_ENV)
    for name in tuple(env):
        if name=="PYTHONPYCACHEPREFIX" or name.endswith("_PYTHONPYCACHEPREFIX"):
            env.pop(name)
    return env


def runtime_paths(command):
    locations=[]
    for i,arg in enumerate(command):
        if arg=="--bind":
            location=Path(command[i+1].split(":",1)[0])
            need(location.exists() and location.resolve(strict=True)==location
                 and (location.is_dir() or location.is_file()),"EXACT_RENDERED_RUNTIME_BIND")
            locations.append(location)
    app=Path(command[command.index("--time=00:03:00")+1])
    sandbox=Path(command[-6])
    need(len(locations)==9 and app.is_file() and os.access(app,os.X_OK)
         and sandbox.is_dir() and sandbox.resolve(strict=True)==sandbox,"ACTUAL_APPTAINER_AND_SANDBOX")


def check(root,query,deployed,clock):
    need(not any(os.path.lexists(root/name) for name in (INTENT,HOST_RESULT,"HOST.stdout","HOST.stderr"))
         and not any((root/"out").iterdir()),"ORIGINAL_INTENT_EXISTS_RECONCILE_ONLY")
    need(not accounting(query,deployed),"ORIGINAL_LABEL_EXISTS_RECONCILE_ONLY")
    info=fresh_slots(query,clock())
    need(not accounting(query,deployed),"ORIGINAL_LABEL_VISIBLE_AFTER_PLACEMENT")
    return info


def launch(root,query,deployed,clock,invoke,paths=runtime_paths):
    before=check(root,query,deployed,clock)
    epoch=clock()
    command=protect(deployed,epoch)
    paths(command)
    need(0<=clock()-epoch<=5 and epoch+WORK<=before["parent_end_epoch"],"FRESH_FULL_120_SECOND_HOST_EPOCH")
    intent={"request":REQUEST,"publication":PUBLICATION,"parent":PARENT,"node":NODE,"label":JOB,"epoch":epoch,
            "deadline":epoch+WORK,"command_SHA256":pin(doc(command))["SHA256"],"stage1_SHA256":STAGES[1][1],
            "source_manifest_SHA256":MANIFEST,"one_srun_maximum":1,"replay_allowed":False,"placement":before}
    # Durable exclusive intent precedes every possible srun entry. O_EXCL fences competing repeats.
    intent_pin=save(root,INTENT,intent)
    rc=None
    failure=None
    started=False
    try:
        with (root/"HOST.stdout").open("xb") as out,(root/"HOST.stderr").open("xb") as err:
            remaining=epoch+WORK-clock()
            need(remaining>0,"ORIGINAL_HOST_DEADLINE")
            started=True
            rc=invoke(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=remaining,
                      env=host_environment(),check=False).returncode
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        failure=type(error).__name__
    result={"state":"SRUN_RETURNED_REQUIRES_INDEPENDENT_READ" if rc is not None else "UNCHARTED_SRUN_OUTCOME",
            "request":REQUEST,"intent":intent_pin,"srun_call_entered":started,"srun_return_code":rc,"error_type":failure,
            "raw_logs_private":True,"replay_allowed":False,"training_ready":False,"actual_optimizer_updates":0}
    save(root,HOST_RESULT,result)
    return result


def terminal(root,query,deployed):
    observed=accounting(query,deployed)
    if not observed:
        return {"state":"NO_AUTHORITATIVE_NUMERIC_STEP_OBSERVED","replay_allowed":False}
    row=observed[0]
    base={"step":row[0],"slurm_state":row[2],"slurm_exit":row[3],"elapsed_seconds":row[4],
          "allocated_TRES":row[6],"replay_allowed":False,"training_ready":False}
    if row[2].split()[0] not in TERMINAL:
        return {**base,"state":"NUMERIC_STEP_IN_FLIGHT"}
    if row[2:4]!=["COMPLETED","0:0"]:
        return {**base,"state":"EXACT_STEP_TERMINAL_FAILURE"}
    intent=json.loads(hash_read(root/INTENT))
    need((intent["request"],intent["publication"],intent["parent"],intent["node"],intent["label"],
          intent["stage1_SHA256"],intent["source_manifest_SHA256"])==
         (REQUEST,PUBLICATION,PARENT,NODE,JOB,STAGES[1][1],MANIFEST),"EXACT_ORIGINAL_HARD_INTENT")
    result_path=root/"out/RESULT.json"
    if not result_path.exists():
        return {**base,"state":"STEP_C0_WORKER_RESULT_NOT_VISIBLE"}
    raw=hash_read(result_path)
    verified=deployed["terminal"](row,json.loads(raw),PARENT,NODE,REQUEST,PUBLICATION,MANIFEST,intent["epoch"])
    return {**base,**verified,"state":"INDEPENDENT_NATIVE_IMAGE_CPU_TERMINAL_PASS","worker_result":pin(raw)}


def main():
    need(re.fullmatch(r"[0-9a-f]{32}",globals().get("TRANSPORT_REQUEST",""))
         and TRANSPORT_REQUEST!=REQUEST,"FRESH_BOUND_TRANSPORT_REQUEST")
    request=json.loads(sys.stdin.buffer.read(24577))
    need(set(request)=={"mode","claim_b64"} and request["mode"] in ("check","launch","terminal"),
         "EXACT_BOUND_HOST_MODE")
    mode=request["mode"]
    claim(b64decode(request["claim_b64"],validate=True))
    need(sys.platform=="linux" and os.getuid()==UID and RUN.resolve(strict=True)==RUN,"EXACT_AUTHENTICATED_HOST_OWNER")
    deployed=stage(RUN)
    os.umask(0o077)
    query_end=time.monotonic()+30
    def query(*argv):
        need(argv[0] in ("sacct","scontrol","squeue"),"READ_ONLY_SCHEDULER_ALLOWLIST")
        left=min(6,query_end-time.monotonic())
        need(left>0,"FRESH_HOST_QUERY_LIMIT")
        raw=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,check=True,timeout=left,
                           env=host_environment()).stdout
        need(len(raw)<=1048576,"BOUNDED_SCHEDULER_METADATA")
        return raw.decode("ascii").splitlines()
    if mode=="check":
        placement=check(RUN,query,deployed,time.time)
        runtime_paths(protect(deployed,time.time()))
        report={"state":"HOST_READ_ONLY_CHECK_PASS","placement":placement,"actual_launch_calls":0}
    elif mode=="launch":
        report=launch(RUN,query,deployed,time.time,subprocess.run)
    else:
        report=terminal(RUN,query,deployed)
    reply(report)


def reply(report):
    output={**report,"request":globals().get("TRANSPORT_REQUEST"),"scope_request":REQUEST}
    print(json.dumps(output,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    try:
        main()
    except (OSError,ValueError,KeyError,TypeError,AttributeError,IndexError,subprocess.SubprocessError) as error:
        report={"state":"HOST_CHECK_BLOCKED_OR_RECONCILE","reason":str(error) if isinstance(error,Rejected) else type(error).__name__,
                "request":REQUEST,"replay_allowed":False,"no_raw_output":True}
        reply(report)
