"""Materialize an existing legal dataset locally; never dispatch or connect remotely."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

S=Path(__file__).resolve().parent
G=S.parents[2]
RUN=G/"local-only/training-gemma-render72-local-view-v1-original"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name,value):
    with (RUN/name).open("x",encoding="utf8") as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")


def module(name,pin):
    path=G/pin["path"]
    raw=path.read_bytes()
    if len(raw)!=pin["bytes"] or sha(raw)!=pin["SHA256"]:
        raise ValueError("Frozen reusable helper changed")
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def verify():
    manifest=json.loads((S/"SOURCE_FILES.json").read_bytes())
    for name,pin in manifest["files"].items():
        path=S/name
        raw=path.read_bytes()
        if path.is_symlink() or path.parent!=S or len(raw)!=pin["bytes"] or sha(raw)!=pin["SHA256"]:
            raise ValueError("Local materializer source identity changed")
    return json.loads((S/"PLAN.json").read_bytes())


def worker(epoch):
    plan=verify()
    if not math.isfinite(epoch) or not 0<epoch<=time.time()<epoch+60:
        raise ValueError("Original local preparation epoch exhausted")
    helper=module("gemma_local_original_qualified_job",plan["job_guard"])
    gate=RUN/"job-gate.json"
    while not gate.exists():
        if time.time()>=epoch+50:
            raise TimeoutError("Local resource gate missing")
        time.sleep(.02)
    resource=json.loads(gate.read_bytes())
    if (resource["deadline"]!=epoch+60 or resource["memory_limit_bytes"]!=2*1024**3
            or resource["CPU_count"]!=2 or resource["job_affinity_verified"] is not True
            or resource["worker_affinity_mask"]!=helper.current_affinity()["process_mask"]):
        raise ValueError("Actual local Windows Job limits differ")
    def isolated(event,args):
        if event in ("socket.connect","socket.getaddrinfo","subprocess.Popen","os.system"):
            raise RuntimeError("Local-only data task cannot contact or dispatch externally")
        if event=="open" and args and isinstance(args[0],(str,bytes)):
            name=str(args[0]).replace("\\","/").lower()
            if "/blind/" in name or "/dev/labels-private.jsonl" in name:
                raise RuntimeError("Development gold and blind bodies are closed")
    sys.addaudithook(isolated)
    result=dict(passed=False,planned_train_rows=72,planned_dev_rows=36,planned_media=18,
                actual_native=0,actual_target_copy=0,actual_optimizer_updates=0,
                dev_gold_read=False,blind_inputs_labels_media_read=False)
    started=time.monotonic()
    try:
        adapter=module("gemma_existing_five_VLM_data_adapter",plan["adapter"])
        args=SimpleNamespace(case="T02.sft",dataset=G/plan["dataset"],output=RUN/"adapter",
            manifest_bytes=plan["dataset_manifest"]["bytes"],manifest_sha=plan["dataset_manifest"]["SHA256"],
            target_data=plan["declared_target_data"])
        generated=adapter.prepare(args)
        from audit import check
        audit=check(args.dataset,args.output,plan)
        save("audit-private.json",audit)
        binding=RUN/"adapter/data-binding-private.json"
        result.update(passed=True,state="LOCAL_DATA_VIEW_PREPARED_AND_AUDITED",adapter=generated,
             audit_SHA256=sha((RUN/"audit-private.json").read_bytes()),
             binding=dict(bytes=binding.stat().st_size,SHA256=sha(binding.read_bytes())),
             training_READY=False,target_ready=False,target_presence_verified=False)
    except BaseException as exc:
        result.update(state="KNOWN_LOCAL_DATA_FAILURE_NO_BACKFILL",error_type=type(exc).__name__)
        save("failure-private.json",dict(error_type=type(exc).__name__,message=str(exc)[:1024],
              planned_train_rows=72,planned_dev_rows=36,planned_media=18))
    result["wall_seconds"]=time.monotonic()-started
    save("worker-result.json",result)
    return 0 if result["passed"] else 1


def prepare_local():
    epoch=time.time()
    end=time.monotonic()+60
    plan=verify()
    # A repeated automatic refill call consumes the exact local terminal record,
    # never regenerates data or retries an unknown/failed attempt.
    if RUN.exists():
        path=RUN/"outcome.json"
        if not path.is_file():
            raise RuntimeError("Original local attempt requires reconciliation; no replay")
        old=json.loads(path.read_bytes())
        if old["passed"]:
            worker=json.loads((RUN/"worker-result.json").read_bytes())
            binding_path=RUN/"adapter/data-binding-private.json"
            raw=binding_path.read_bytes()
            if len(raw)!=worker["binding"]["bytes"] or sha(raw)!=worker["binding"]["SHA256"]:
                raise ValueError("Prepared local input binding changed; do not auto-dispatch")
            binding=json.loads(raw)
            for name,pin in binding["files"].items():
                raw=(RUN/"adapter"/name).read_bytes()
                if len(raw)!=pin["bytes"] or sha(raw)!=pin["SHA256"]:
                    raise ValueError("Prepared local JSONL disappeared or changed")
        return dict(state="EXISTING_LOCAL_TERMINAL_REUSED_NO_NEW_WORK",passed=old["passed"],
                    original_outcome_SHA256=sha(path.read_bytes()),training_READY=False,actual_new_target_calls=0)
    RUN.mkdir(mode=0o700)
    save("intent.json",dict(request=plan["request"],epoch=epoch,deadline=epoch+60,
         source_manifest_SHA256=sha((S/"SOURCE_FILES.json").read_bytes()),dataset_manifest=plan["dataset_manifest"],
         planned_train=72,planned_dev=36,planned_media=18,local_only=True,actual_remote=0))
    directory=RUN/"job"
    directory.mkdir()
    helper=module("gemma_local_existing_job_guard",plan["job_guard"])
    argv=[sys.executable,"-B","-X","utf8",str(S/"once.py"),"--worker","--epoch",str(epoch)]
    job=helper.guarded_run(argv,RUN/"job-gate.json",directory,epoch+60,end)
    save("job-result.json",job)
    result_path=RUN/"worker-result.json"
    result=json.loads(result_path.read_bytes()) if result_path.is_file() else None
    passed=job["returncode"]==0 and not job["timed_out"] and result and result["passed"] is True
    outcome=dict(request=plan["request"],passed=bool(passed),epoch=epoch,deadline=epoch+60,job=job,
         worker_result_SHA256=sha(result_path.read_bytes()) if result_path.is_file() else None,
         local_only=True,target_ready=False,training_READY=False,actual_optimizer_updates=0,
         cluster_authentication_verified=False,actual_remote=0)
    save("outcome.json",outcome)
    if passed:
        binding=RUN/"adapter/data-binding-private.json"
        save("refill-candidate-private.json",dict(id="gemma-render72-local-prepared-next-native512",
            kind="local_data_prerequisite",state="LOCAL_PREPARED_TARGET_BLOCKED",case="T02.sft",
            input_binding=dict(bytes=binding.stat().st_size,SHA256=sha(binding.read_bytes()),path=str(binding)),
            automatically_repeat_safe_entry="once.prepare_local()",automatic_remote_dispatch=False,
            next_required_stage="PUBLISH_NATIVE_GEMMA512_CPU_CLAIM_AFTER_AUTH_RECOVERY",
            requires=["actual cluster authentication","published native CPU scope","actual original model/initial identities",
                      "all108 native input/mask/EOS/full512 qualification","fresh resource admission before training"],
            proposed_future_training=dict(continuous_updates=25,independent_resume20_updates=5,
                original_frozen_recipe_preserved=True,new_domain_recipe=True),
            target_presence_verified=False,training_READY=False,GPU_GO=False,Model_GO=False))
    return dict(passed=bool(passed),local_only=True,training_READY=False,target_ready=False,
                actual_native=0,actual_remote=0,actual_optimizer_updates=0)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--prepare-local",action="store_true")
    p.add_argument("--worker",action="store_true")
    p.add_argument("--epoch",type=float)
    a=p.parse_args()
    if a.worker:
        raise SystemExit(worker(a.epoch))
    if not a.prepare_local:
        p.error("explicit --prepare-local is required; no remote mode exists")
    print(json.dumps(prepare_local(),sort_keys=True),flush=True)
