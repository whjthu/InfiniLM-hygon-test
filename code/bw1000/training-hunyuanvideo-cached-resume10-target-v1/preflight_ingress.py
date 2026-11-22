"""Prepare fresh HOST preflight bytes from the sealed original .828 asset proof."""

import argparse
import json
from pathlib import Path

from contract import (Budget, NODE, PARENT, REQUEST, RESUME_AFTER, ROOT, RUN, WHOLE, load, need,
                      save, validate_assets, validate_preflight, verify_package, verify_resume_input)
from launch import prepare


RUN = Path(RUN)
OLD_LABEL = "training-hunyuanvideo-cached-resume20-target-v1"
OLD_REQUEST = "80e60c5dab1cd4eda2b3d99be39c96d4"
OLD_PREFLIGHT = "14370e8ae9b7583c43bc9ff1b13cec152dc94853abcf2e4872045df407386912"
PROOF_SHA = "98011dffef6501ab4dacdfbc67c6d090cf386ba4e34f80ea7d2b00509da9834d"
NEW_RECEIPT = "preflight.json"


def inherit(old, proof, origin, source):
    need(proof["request"]==OLD_REQUEST and proof["original_GPU_step"]=="1087656.828"
         and proof["original_GPU_slurm_state"]=="FAILED" and proof["original_post_CPU_step"]=="1087656.829"
         and proof["comparison_available"] is True and proof["same_platform_recovery_qualified"] is True
         and proof["comparison"]["passed"] is True and proof["comparison"]["resumed_updates"]==5
         and proof["accounting"][0][0:4]==["1087656.830","T09-original828-recovery-test-CPU-v1","COMPLETED","0:0"],
         "corrected original .830 recovery precedent differs")
    need(old["request_id"]==OLD_REQUEST and old["run"]==str(Path(ROOT)/"results"/OLD_LABEL)
         and old["node"]==NODE and old["parent"]==PARENT,"prior asset evidence belongs to a different scope")
    return dict(old,request_id=REQUEST,run=str(source.parent),epoch=origin,deadline=origin+WHOLE,
                inherited_original_preflight_SHA256=OLD_PREFLIGHT,
                original_corrected_comparison_precedent_SHA256=PROOF_SHA,
                new_resume_after=RESUME_AFTER,native_source_manifest=dict(
                    old["native_source_manifest"],path=str(source/"NATIVE_BINDING.json")))


def ingress(origin, manifest_sha):
    budget=Budget(origin)
    source=RUN/"src"
    need(RUN.resolve(strict=True)==RUN and not (RUN/NEW_RECEIPT).exists()
         and not any((RUN/name).exists() for name in ("intent.json","outcome.json","outputs")),
         "original new request already prepared or not in canonical target namespace")
    verify_package(source,manifest_sha)
    proof,pin=load(source/"REFERENCE-830.json",16384)
    need(pin["SHA256"]==PROOF_SHA,"corrected public .830 receipt bytes changed")
    old,pin=load(Path(ROOT)/"results"/OLD_LABEL/"preflight.json",16384)
    need((pin["SHA256"],pin["bytes"])==(OLD_PREFLIGHT,2251),"sealed .828 preflight bytes differ")
    preflight=inherit(old,proof,origin,source)
    validate_preflight(preflight,origin)
    assets=validate_assets(preflight,execution_mount=False)
    resume=verify_resume_input()
    need(budget.left()>180,"original preflight depleted the terminal/accounting reserve")
    save(RUN/NEW_RECEIPT,preflight)
    _,pin=load(RUN/NEW_RECEIPT)
    return dict(state="FRESH_HOST_CP10_PREFLIGHT_PASS",request=REQUEST,scope_replay_allowed=False,
                preflight=pin,assets=assets,resume_input=resume,
                dispatch=prepare(origin,manifest_sha,pin["SHA256"]),actual_GPU_steps=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch",type=float,required=True)
    parser.add_argument("--source-manifest-sha256",required=True)
    args=parser.parse_args()
    print(json.dumps(ingress(args.epoch,args.source_manifest_sha256),sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
