"""Independent failure-aware reader. First actual terminal accounting, then immutable CPU artifacts."""
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess

ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
R=ROOT/"training-qwen-treasury-cache-host-main-candidate-v1"
LABEL="training-qwen-treasury-cache-host-main-CPU-v1"
REQUEST="83c1b1221bb54d3bbf3f2fde8f9dc708"
PUBLICATION="acde72e548d88acd539bb1b688494e4124291df2"
CLAIM_SHA="7fe521e30549334d60981d9ff3bc9d40f89057f7825df9be86904dabee0dac94"
MF_SHA="ea63886aee44354e12e6335626f5462f8d81a08b6de6b73354bb977753845b18"
CANDIDATE_MF="c4916078b7a96367f774109bb502d96946907824dd1105ac26faa980b86c70ec"
SPEC={"base":("1087656","a01r4n14"),"old_domain":("1087656","a01r4n16"),"domain":("1103606","f13r1n10")}
COLS="JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL","DEADLINE","REVOKED"}
PINS={}
MISSING=[]
OPTIONAL_ERRORS=[]


def require(ok,why):
    if not ok:raise ValueError(why)


def unique_pairs(items):
    result={}
    for k,v in items:
        require(k not in result,"duplicate original JSON key");result[k]=v
    return result


def decode(b):
    return json.loads(b.decode("utf8"),object_pairs_hook=unique_pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError("nonfinite original")))


def raw(path,limit=2*1024**2):
    before=path.lstat();require(stat.S_ISREG(before.st_mode) and 0<=before.st_size<=limit,"unsafe/big original file")
    b=path.read_bytes();after=path.lstat()
    require(all(getattr(before,k)==getattr(after,k) for k in ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns")),"original changed while reading")
    PINS[str(path)]=hashlib.sha256(b).hexdigest();return b


def read(path):return decode(raw(path))


def optional(path):
    if not path.exists():MISSING.append(str(path));return None
    try:return read(path)
    except (OSError,ValueError,TypeError) as error:
        OPTIONAL_ERRORS.append(dict(path=str(path),error_class=type(error).__name__,original_SHA256=PINS.get(str(path))))
        return None


def ownership_summary(value,parent,node):
    if value is None:return dict(present=False,qualified=False,reason="MISSING")
    reasons=[]
    def check(ok,why):
        if not ok:reasons.append(why)
    check(value.get("complete_inventory") is True and value.get("same_UID_FD_inventory_complete") is True and value.get("errors")==[],"INCOMPLETE_OR_ERROR")
    check(value.get("node")==node and bool(re.fullmatch(re.escape(parent)+r"\.[0-9]+",value.get("step",""))),"STEP_NODE")
    pci=value.get("pci")
    check(isinstance(pci,list) and len(pci)==len(set(pci))==8 and pci==sorted(pci)
          and all(isinstance(p,str) and re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",p) for p in pci),"PCI8")
    proc=value.get("stage_process") or {}
    check(type(value.get("UID")) is int and proc.get("UID")==value.get("UID") and proc.get("PID",0)>0 and proc.get("startticks",0)>0,"UID_PID_START")
    check(value.get("kernel_KFD_holders")==[] and value.get("same_UID_device_FD_holders")==[] and value.get("KFD_holder_count")==0,"KFD_DEVICE_HOLDER_OR_UNKNOWN")
    check(value.get("proc_cmdline_or_environ_read") is False and value.get("busy_qualification") is False,"READONLY_OR_BUSY_BOUNDARY")
    boot=value.get("boot_id")
    check(isinstance(boot,str) and bool(re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",boot)),"BOOT")
    return dict(present=True,qualified=not reasons,reasons=reasons,source_errors=value.get("errors"),node=value.get("node"),step=value.get("step"),UID=value.get("UID"),
        stage_process=proc,PCI_count=len(pci) if isinstance(pci,list) else None,PCI=pci,boot_id=boot,monotonic_ns=value.get("monotonic_ns"),
        KFD_holder_count=value.get("KFD_holder_count"),kernel_KFD_holders=value.get("kernel_KFD_holders"),same_UID_device_FD_holders=value.get("same_UID_device_FD_holders"),
        same_UID_processes_checked=value.get("same_UID_processes_checked"),FD_entries_checked=value.get("FD_entries_checked"),freshness_is_not_busy=True)


def post_verdict(before,after,comparison,owned,parent,node):
    pre=ownership_summary(before,parent,node);post=ownership_summary(after,parent,node)
    same=False
    if before and after:
        same=bool(before.get("step")==after.get("step") and before.get("pci")==after.get("pci") and before.get("boot_id")==after.get("boot_id")
            and before.get("UID")==after.get("UID") and (before.get("stage_process") or {}).get("PID")== (after.get("stage_process") or {}).get("PID")
            and (before.get("stage_process") or {}).get("startticks")== (after.get("stage_process") or {}).get("startticks")
            and type(after.get("monotonic_ns")) is int and type(before.get("monotonic_ns")) is int and after["monotonic_ns"]>=before["monotonic_ns"])
    exited=(comparison or {}).get("container_exit") or {}
    known_exit=False
    if owned:
        if "startticks" in owned:
            known_exit=bool(exited.get("qualified") is True and (exited.get("original_owned_PID_absent") is True
                or (exited.get("observed_PID")==owned.get("PID") and (exited.get("observed_startticks")!=owned.get("startticks") or exited.get("observed_state")=="Z"))))
    else:
        known_exit=exited.get("qualified") is True and exited.get("container_not_started") is True
    return dict(independent_post_data_qualified=bool(pre["qualified"] and post["qualified"] and same and known_exit),pre=pre,post=post,
        same_CPUstep_PCI_boot_UID_stage_PID_startticks=same,known_owned_container_identity=owned,recorded_container_exit=exited,
        owned_container_exit_identity_qualified=known_exit,producer_comparison_passed=(comparison or {}).get("passed"),
        limitation="Recorded same-step post only; no new node/proc/SDK probe performed.")


def redact(text):
    text=re.sub(r"\x1b\[[0-9;]*[A-Za-z]","",text).replace(str(ROOT.parent),"<PROJECT_ROOT>")
    text=re.sub(r"(?i)(?:[a-z][a-z0-9+.-]*://|www\.)[^\s'\"]+","<URL_REDACTED>",text)
    text=re.sub(r"(?<![A-Za-z0-9_])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]+)?","<ENDPOINT_REDACTED>",text)
    text=re.sub(r"(?i)\b(?:password|token|secret|key)\s*[:=]\s*(?:'[^']*'|\"[^\"]*\"|[^\s,;]+)","<CREDENTIAL_ASSIGNMENT_REDACTED>",text)
    text=re.sub(r"\b[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+(?::[0-9]+)?","<ENDPOINT_REDACTED>",text)
    return text


def log_diagnostic(path):
    if not path.exists():return dict(present=False)
    info=path.lstat();require(stat.S_ISREG(info.st_mode),"unsafe log")
    with path.open("rb") as stream:stream.seek(max(0,info.st_size-4096));b=stream.read(4096)
    lines=redact(b.decode("utf8",errors="replace")).splitlines()
    selected=[line[:240] for line in lines if re.search(r"Traceback|File .*line [0-9]+|Error|ERROR|Exception|warning|WARNING|srun:",line)]
    return dict(present=True,bytes=info.st_size,tail_bytes=len(b),tail_SHA256=hashlib.sha256(b).hexdigest(),redacted_whitelist_tail_lines=selected[-8:],
                raw_logs_retained_R=True,whole_log_not_transferred=True)


def cache_counts(cache):
    if cache is None:return dict(present=False,verified_tested_bytes=None,actual_partial_writes="UNKNOWN")
    records=cache.get("records",{})
    good=bool(cache.get("all_cache_canaries_verified") is True and cache.get("total_canary_bytes")==40960 and len(records)==10
        and all(v.get("bytes_written")==4096 and v.get("own_canary_removed") is True for v in records.values())
        and cache.get("tested_bytes_only") is True and cache.get("user_quota_and_arbitrary_JIT_capacity_qualified") is False and cache.get("GPU_GO") is False)
    return dict(present=True,qualified=good,verified_tested_bytes=40960 if good else None,reported_total=cache.get("total_canary_bytes"),
        actual_cache_directory_count=len(records),canary_only_tested_bytes=True,quota_or_arbitrary_JIT_capacity=False)


def main_counts(main,arm,cache_path):
    if main is None:return dict(present=False,qualified=False,actual_InferRequest_objects=None,actual_AdapterRequest_objects=None,reason="MISSING_RECEIPT_OBJECT_COUNTS_UNKNOWN_NOT_ZERO")
    native=R/"source/native_generation.py"
    good=bool(main.get("state")=="TRUE_ARGPARSE_MAIN_REAL_ENGINE_IMPORTS_AND_CPU_PROCESSOR_RETURNED_BEFORE_LOAD_MODEL_TRUE"
      and main.get("arm")==arm and main.get("actual_InferRequest_objects")==108 and main.get("actual_AdapterRequest_objects")==2
      and main.get("actual_RequestConfig")=={"max_tokens":64,"temperature":0,"num_beams":1,"n":1,"stream":False}
      and main.get("load_model_True_calls")==main.get("Engine_instances_created")==main.get("generation_calls")==0
      and main.get("CUDA_success_mocked") is False and main.get("GPU_GO") is False
      and main.get("native_main_file")==str(native) and main.get("native_main_SHA256")==PINS.get(str(native))
      and main.get("input_inventory_SHA256")=="07d906e3ed840d828ee41509cae21b45a754ddef0825d411ca3d8216f606b0d8"
      and main.get("cache_readiness_SHA256")==PINS.get(str(cache_path)))
    return dict(present=True,qualified=good,actual_InferRequest_objects=main.get("actual_InferRequest_objects"),actual_AdapterRequest_objects=main.get("actual_AdapterRequest_objects"),
        actual_RequestConfig=main.get("actual_RequestConfig"),CPU_processor_type=main.get("real_CPU_processor_type"),template_type=main.get("real_template_type"),
        recorded_provider_files=main.get("providers"),recorded_torch_provider=main.get("torch_provider"),ENGINE_PINS_verified=main.get("ENGINE_PINS_verified"),
        loadModelTrue_calls=main.get("load_model_True_calls"),Engine_instances=main.get("Engine_instances_created"),generation_calls=main.get("generation_calls"),
        signature_only=True,CPU_metadata_not_GPU_numerics=True,library_implicit_device_queries="UNKNOWN",current_provider_files_rehashed=False)


def audit():
    # This ordering is deliberate: never start by consuming an in-flight producer receipt.
    observed=[]
    for parent in ("1087656","1103606"):
        text=subprocess.check_output(["sacct","-n","-P","-j",parent,"-o",COLS],text=True,timeout=9)
        observed += [row for line in text.splitlines() if len(row:=line.split("|"))==7 and row[1] in {LABEL+"-"+a for a in SPEC}]
    rows={}
    for arm in SPEC:
        found=[row for row in observed if row[1]==LABEL+"-"+arm];require(len(found)<=1,"duplicate CPU label")
        rows[arm]=found[0] if found else None
    if not observed or any(row[2].split()[0] not in TERMINAL for row in observed):
        return dict(state="ACTUAL_ACCOUNTING_MISSING_OR_NONTERMINAL_UNKNOWN",request=REQUEST,accounting_columns=COLS,actual_accounting=rows,
              planned_arm_denominator=3,artifact_consumption_started=False,GPU_GO=False,replay_allowed=False)
    intent=read(R/"intent.json");collector=optional(R/"outcome.json");claim=read(R/"CPU_CLAIM.json");mf=read(R/"CPU_PACKAGE_FILES.json")
    require(PINS[str(R/"CPU_CLAIM.json")]==CLAIM_SHA and PINS[str(R/"CPU_PACKAGE_FILES.json")]==MF_SHA,"CPU claim/package source differs")
    require(len(mf["files"])+1==22,"source22 denominator differs")
    for name,info in mf["files"].items():
        p=Path(name);require(not p.is_absolute() and ".." not in p.parts and "\\" not in name,"unsafe package path")
        b=raw(R/name);require(len(b)==info["bytes"] and hashlib.sha256(b).hexdigest()==info["SHA256"],"original source member differs")
    require(PINS[str(R/"source/SOURCE_FILES.json")]==CANDIDATE_MF,"exact candidate11 differs")
    require(intent["request"]==REQUEST and intent["publication"]==PUBLICATION and intent["claim_SHA256"]==CLAIM_SHA
          and intent["deadline"]==intent["epoch"]+900 and intent["actual_expected_GPU_cards"]==0 and intent["GPU_GO"] is False,"original900epoch/request differs")
    if collector is not None:require(collector["request"]==REQUEST and collector["epoch"]==intent["epoch"] and collector["deadline"]==intent["deadline"] and set(collector["arms"])==set(SPEC),"collector identity/3denominator differs")
    arms={}
    for arm,(parent,node) in SPEC.items():
        folder=R/"arms"/arm;item=optional(folder/"outcome.json");row=rows[arm]
        if item is not None and item.get("dispatched") is False:
            require(row is None and item.get("passed") is False,"NOT_DISPATCHED conflicts with actualstep")
            arms[arm]=dict(dispatch_state="NOT_DISPATCHED",producer_reason=item.get("reason"),prior=item.get("prior"),actual_accounting=None,
               planned_InferRequest_objects=108,planned_AdapterRequest_objects=2,planned_canary_bytes=40960,actual_InferRequest_objects=0,actual_AdapterRequest_objects=0,verified_canary_bytes=0,qualification=False)
            continue
        if row is None:
            arms[arm]=dict(dispatch_state="UNKNOWN_NO_ACTUAL_ACCOUNTING",producer_outcome_present=item is not None,planned_InferRequest_objects=108,planned_canary_bytes=40960,qualification=False)
            continue
        tres=dict(v.split("=",1) for v in row[6].split(","))
        resources=(row[5]==node and row[0].startswith(parent+".") and tres=={"cpu":"8","mem":"12G","node":"1"})
        require(resources,"actual CPU8/mem12G/0GRES resource differs")
        if item is not None:require(item.get("accounting",{}).get("row")==row,"producer and actualaccounting differ")
        inside=optional(folder/"inside-outcome.json");step=optional(folder/"step-intent.json");container=optional(folder/"container-intent.json")
        pre=optional(folder/"ownership-pre.json");post=optional(folder/"ownership-post.json");comparison=optional(folder/"ownership-comparison.json");owned=optional(folder/"container-owned-process.json")
        cache_path=folder/"owned-cache-readiness.json";cache=optional(cache_path);main=optional(folder/"host-main/host-main-receipt.json")
        epoch_ok=bool(step and step.get("original_epoch")==intent["epoch"] and step.get("deadline")==intent["deadline"]
           and step.get("work_deadline",float("inf"))<=intent["deadline"]-40 and step.get("limits",{}).get("Slurm_seconds") in (120,180)
           and step.get("limits",{}).get("work_seconds",999)<=150)
        if inside is not None:epoch_ok=bool(epoch_ok and inside.get("original_epoch")==intent["epoch"] and inside.get("deadline")==intent["deadline"] and inside.get("work_deadline")==step.get("work_deadline"))
        if container is not None:epoch_ok=bool(epoch_ok and container.get("original_epoch")==intent["epoch"] and container.get("deadline")==intent["deadline"] and container.get("work_deadline")==step.get("work_deadline"))
        post_result=post_verdict(pre,post,comparison,owned,parent,node)
        cc=cache_counts(cache);mc=main_counts(main,arm,cache_path)
        qualified=bool(row[2:4]==["COMPLETED","0:0"] and epoch_ok and post_result["independent_post_data_qualified"] and cc.get("qualified") and mc.get("qualified")
                       and item and item.get("passed") is True and inside and inside.get("passed") is True)
        diagnostics={name:log_diagnostic(folder/name) for name in ("Slurm.stderr","stage.stdout","container.stderr","canary-hostmain.stdout")}
        arms[arm]=dict(dispatch_state="ACTUAL_TERMINAL",actual_accounting=row,actual_terminal_state=row[2],actual_resources_0GPU_verified=resources,
            original_epoch_not_reset_verified=epoch_ok,producer_passed=(item or {}).get("passed"),inside_passed=(inside or {}).get("passed"),inside_error_class=(inside or {}).get("error_class"),
            post=post_result,cache=cc,HOST_main=mc,diagnostics=diagnostics,qualification=qualified,planned_InferRequest_objects=108,planned_AdapterRequest_objects=2,planned_canary_bytes=40960)
    complete_pass=bool(collector and collector.get("passed") is True and len(arms)==3 and all(v.get("qualification") for v in arms.values()))
    return dict(state="ORIGINAL_CPU_CACHE_MAIN_TERMINAL_CONSUMED_FAILURES_RETAINED",request=REQUEST,source_publication=PUBLICATION,epoch=intent["epoch"],deadline=intent["deadline"],
        accounting_columns=COLS,source_files_verified=22,candidate_exact_files=11,planned_arm_denominator=3,arms=arms,
        original_collector_passed=(collector or {}).get("passed"),independent_complete_CPU_scope_pass=complete_pass,
        canary_verified_bytes_from_receipts=sum(v.get("cache",{}).get("verified_tested_bytes") or 0 for v in arms.values()),
        InferRequest_objects_verified_from_main_receipts=sum(v.get("HOST_main",{}).get("actual_InferRequest_objects") or 0 for v in arms.values() if v.get("HOST_main",{}).get("qualified")),
        AdapterRequest_objects_verified_from_main_receipts=sum(v.get("HOST_main",{}).get("actual_AdapterRequest_objects") or 0 for v in arms.values() if v.get("HOST_main",{}).get("qualified")),
        missing_or_partial_main_object_counts_are_unknown=True,objects_are_not_generated_tokens_or_requests=True,generation_observed_or_executed_by_reader=0,
        original_artifact_pins=PINS,missing_originals=MISSING,optional_original_errors=OPTIONAL_ERRORS,GPU_GO=False,no_GPUv4=True,
        capacity_producer_is_not_independent_PASS=True,old_capacity_reader_UNKNOWN_not_replayed=True,data53_rejected_domain_untouched=True,
        actual_new_probe_SDK_weight_reads_or_Slurm=0,current_fleet_busy_qualified=False,promotion=False)


if __name__=="__main__":print(json.dumps(audit(),allow_nan=False))
