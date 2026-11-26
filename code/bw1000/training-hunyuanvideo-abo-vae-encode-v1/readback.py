"""Independently read back one worker artifact without importing any model runtime."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import contract as c


def account(doc,request,worker,worker_sha):
    row=doc["accounting"]
    c.require(doc["independent"] is True and doc["TZ"]=="UTC" and doc["request"]==request
              and doc["worker_SHA256"]==worker_sha and type(row) is list and len(row)==9
              and row[0]==worker["actual_step"] and re.fullmatch(r"[0-9]+\.[0-9]+",row[0])
              and row[2:4]==["COMPLETED","0:0"] and row[4].isdecimal(),"EXACT_INDEPENDENT_NUMERIC_TERMINAL")
    start,end=[datetime.strptime(x,"%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() for x in row[7:9]]
    c.require(worker["node"]==row[5] and 0<=int(row[4])<=(120 if worker["mode"]=="inspect" else 600)
              and end>=start and abs(end-start-int(row[4]))<=1
              and worker["epoch"]-1<=start<=end<=worker["deadline"]+1,"EXACT_ONE_EPOCH_TERMINAL")
    items=row[6].split(",")
    resources=dict(x.split("=",1) for x in items)
    c.require(len(resources)==len(items) and resources.get("node")=="1"
              and resources["cpu"].isdecimal() and int(resources["cpu"])>0 and resources.get("mem"),
              "EXACT_CPU_MEMORY_NODE_TRES")
    gres={k:v for k,v in resources.items() if k.startswith("gres/")}
    c.require((not gres if worker["mode"]=="inspect" else
              gres=={"gres/dcu":"1"} or gres=={"gres/dcu:gfx936":"1"} or
              gres=={"gres/dcu":"1","gres/dcu:gfx936":"1"}),"EXACT_WORKER_TRES")
    return row


def review(folder,request,worker_sha,terminal,terminal_sha):
    raw=c.read(folder/"worker.json",16384)
    terminal_raw=c.read(terminal,16384)
    c.require(c.pin(raw)["SHA256"]==worker_sha and c.pin(terminal_raw)["SHA256"]==terminal_sha,"BOUND_ORIGINAL_WORKER_AND_TERMINAL")
    event,accounted=map(json.loads,(raw,terminal_raw))
    c.require(event["schema"]=="HUNYUAN_ABO_VAE_CANDIDATE_WORKER_V1" and event["request"]==request
              and event["passed"] is True and event["phase"]=="sealed" and event["recipe"]==c.RECIPE
              and event["mode"] in {"inspect","encode"}
              and event["planned_frames"]==event["validated_frames"]==len(c.RGB)==9
              and event["media"]["decoded_RGB_SHA256"]==list(c.RGB)
              and event["source"]["active_class_source"]==c.SOURCE_PIN
              and event["old_cache_equivalence_verified"] is False
              and event["training_rows"]==event["optimizer_updates"]==event["text_embeddings"]==0,
              "SEALED_SOURCE_AND_NO_TRAINING_NATIVE_WORKER")
    row=account(accounted,request,event,worker_sha)
    asset=event["asset"]
    c.require(asset["config"]==c.CONFIG_PIN and asset["header"]==c.HEADER_PIN and asset["weight"]==c.WEIGHTS_PIN
              and asset["current_weight_bytes"]==c.WEIGHTS_PIN["bytes"],"BOUND_NATIVE_VAE_CONFIG_HEADER_WEIGHT")
    result=dict(schema="HUNYUAN_ABO_VAE_INDEPENDENT_READBACK_V1",request=request,
                worker=c.pin(raw), terminal=c.pin(terminal_raw), accounting=row, independent=True,
                mode=event["mode"],native_encode_calls=0,validated_frames=9, candidate_latents=0,
                old_cache_equivalence_verified=False,training_rows=0,text_embeddings=0,optimizer_updates=0)
    if event["mode"]=="inspect":
        c.require(event["actual_native_encode_calls"]==event["actual_candidate_latents"]==asset["weight_payload_bytes_read"]==0
                and event["actual_GPU"]==event["native_encode_attempts"]==0 and asset["actual_whole_weight_SHA256"] is None
                and not (folder/c.LATENT).exists(),"METADATA_ONLY_ZERO_GRES_CUSTODY")
    else:
        runtime=event["runtime"]
        c.require(event["actual_candidate_latents"]==runtime["actual_native_encode_calls"]==
                  event["actual_native_encode_calls"]==event["native_encode_attempts"]==event["actual_GPU"]==1
                  and runtime["device"]=="cuda:0" and runtime["architecture"]=="gfx936"
                  and runtime["config_scale"]==1.03682 and runtime["config_shift"]==0.0
                  and runtime["posterior"]=="mode" and asset["actual_whole_weight_SHA256"]==c.WEIGHTS_PIN["SHA256"],
                  "ACTUAL_NATIVE_ENCODER_EXECUTION_REQUIRED")
        latent=c.payload_summary(c.read(folder/c.LATENT,c.BYTES,event["latent"]["payload"]),c.SHAPE)
        c.require(latent==event["latent"],"INDEPENDENT_LATENT_NUMERIC_SUMMARY")
        result.update(native_encode_calls=1,candidate_latents=1,verified_latent=latent)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request",required=True)
    p.add_argument("--worker-dir",type=Path,required=True)
    p.add_argument("--worker-sha256",required=True)
    p.add_argument("--terminal",type=Path,required=True)
    p.add_argument("--terminal-sha256",required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=review(args.worker_dir,args.request,args.worker_sha256,args.terminal,args.terminal_sha256)
    receipt=c.write(args.out,c.encoded(result))
    print(json.dumps(dict(request=args.request,mode=result["mode"],readback=receipt,
                          native_encode_calls=result["native_encode_calls"],training_rows=0),sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
