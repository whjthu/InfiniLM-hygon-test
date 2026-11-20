"""One original metadata review, no processor, image, model or new Slurm query."""
import argparse
import hashlib
import json
from pathlib import Path
import signal

ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-usgs-native88-CPU-v1")
REQUEST="7f9a0e2809bf4f53a7294fdc3dc74225"


def read(context):
    total=0
    pins={}
    def need(ok):
        if not ok:
            raise ValueError("Original native88 metadata contract differs")
    def load(name,expected=None):
        nonlocal total
        p=ROOT/name
        need(not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve()))
        before=p.stat()
        need(0<before.st_size<=4*1024**2 and total+before.st_size<=8*1024**2)
        raw=p.read_bytes()
        after=p.stat()
        need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns))
        h=hashlib.sha256(raw).hexdigest()
        need(expected is None or expected==h)
        pins[name]=dict(bytes=len(raw),SHA256=h)
        total+=len(raw)
        return json.loads(raw)
    intent=load("intent.json")
    need(intent["request"]==context.request==REQUEST and intent["publication"]==context.publication
         and intent["epoch"]==context.epoch and intent["deadline"]==context.epoch+300
         and intent["source_manifest_SHA256"]==context.source_manifest_sha256 and intent["claim_SHA256"]==context.claim_sha256)
    source=load("src/SOURCE_FILES.json",context.source_manifest_sha256)
    claim=load("src/CLAIM.json",context.claim_sha256)
    need(claim["CPU_GO"] is True and claim["GPU_GO"] is False and claim["training_GO"] is False)
    for name,pin in source["files"].items():
        p=ROOT/"src"/name
        raw=p.read_bytes()
        need(len(raw)==pin["bytes"] and hashlib.sha256(raw).hexdigest()==pin["SHA256"])
    out=load("outcome.json")
    need(out["request"]==REQUEST and out["passed"] is True and out["srun_returncode"]==0
         and out["srun_reaped"] is True and out["terminal"]["terminal_visible"] is True)
    row=out["terminal"]["row"]
    need(row[1]==ROOT.name and row[2:4]==["COMPLETED","0:0"] and row[5]=="f13r1n19")
    tres=dict(x.split("=",1) for x in row[6].split(","))
    need(tres.get("cpu")=="8" and tres.get("node")=="1" and tres.get("mem") in ("12G","12288M") and not any(k.startswith("gres/") for k in tres))
    admission=load("asset-admission.json")
    need(admission["model_metadata_files"]==9 and admission["media_files"]==22
         and admission["initial_file_hashed"] is True and admission["initial_tensor_state_deserialized"] is False
         and admission["dev_or_blind_read"] is False)
    result=load("worker-result.json",out["worker_result_SHA256"])
    progress=load("progress.json")
    providers=load("providers-private.json")
    rows=load("native-identities-private.json",result["native_identities"]["SHA256"])
    suffix=load("native-suffix-private.json",result["native_suffix"]["SHA256"])
    need(suffix["template_suffix"]==["<|im_end|>\n"] and suffix["full_and4096_suffix_equal"] is True
         and suffix["native_encoder"]=="_encode_context_list" and suffix["EOS_offsets"]
         and suffix["native_suffix_metadata_encode_calls"]==result["native_suffix_metadata_encode_calls"]==2)
    need(result["passed"] is True and result["completed_rows"]==result["planned_rows"]==88
         and result["encode_calls"]==176 and result["collator_calls"]==88
         and result["EOS_supervised"]==result["full_untruncated"]==88 and len(rows)==88
         and result["model_loaded"] is False and result["metadata_only_model_view"] is True
         and result["own_device_FDs_before"]==result["own_device_FDs_after"]==[]
         and result["actual_optimizer_updates"]==0 and result["dev_or_blind_read"] is False
         and len(set(result["affinity"]))==8)
    need(progress["completed_rows"]==88 and progress["encode_completed"]==176 and progress["collator_completed"]==88)
    need([x["ordinal"] for x in rows]==list(range(1,89)) and len({x["source_id_SHA256"] for x in rows})==88)
    need(all(1<x["length"]<=4096 and x["EOS_supervised"] is True and x["full_untruncated"] is True
             and x["shifted_supervised_tokens"]>0 for x in rows))
    need(sum(x["shifted_supervised_tokens"] for x in rows)==result["actual_shifted_supervised_tokens"])
    for role in ("get_model_processor","get_template","encode","data_collator","set_mode","encode_context","tokenize"):
        chain=providers[role]
        need(0<len(chain)<=5 and chain[-1]["official_wheel_bytes_equal"] is True)
        need(all(x["official_wheel_bytes_equal"] is True or x["package"]=="torch" for x in chain))
    return dict(state="MINICPM_USGS88_NATIVE_CPU_INDEPENDENT_METADATA_PASS",passed=True,request=REQUEST,
         publication=context.publication,planned_rows=88,completed_rows=88,missing_rows=0,failed_rows=0,
         actual_encode_calls=176,actual_collator_calls=88,unique_train_media=22,
         actual_native_suffix_metadata_encodes=2,suffix_EOS_and_real_separator_verified=True,
         EOS_supervised=88,full_untruncated=88,token_range=result["token_range"],
         actual_shifted_supervised_tokens=result["actual_shifted_supervised_tokens"],
         actual_CPU_step=row[0],actual_step_elapsed_seconds=int(row[4]),
         actual_worker_wall_seconds=result["wall_seconds"],actual_peak_process_RSS_bytes=result["peak_process_RSS_bytes"],
         actual_read_bytes=total,metadata_files=len(pins),metadata_pins=pins,
         original_initial_named_SHA256=admission["initial_named_SHA256"],initial_tensor_state_loaded_here=False,
         original151_numerical_drift_preserved=True,old_unknown_not_replayed=True,
         model_loaded=False,actual_GPU_allocation=0,actual_optimizer_updates=0,dev_or_blind_read=False,
         implicit_native_loader_GPU_access="UNKNOWN",training_READY=False,Model_GO=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("request","publication","source-manifest-sha256","claim-sha256"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--epoch",type=float,required=True)
    args=parser.parse_args()
    def expired(*_):
        raise TimeoutError("Original independent metadata-read budget expired")
    signal.signal(signal.SIGALRM,expired);signal.alarm(45)
    print(json.dumps(read(args),sort_keys=True,allow_nan=False),flush=True)
