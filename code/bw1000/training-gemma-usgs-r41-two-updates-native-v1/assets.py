"""Require exact target model bytes/header continuity without copying weights."""

import json
import os
from pathlib import Path
import stat

from prepare import MODEL_BINDING_SHA, MODEL_INVENTORY_SHA, NEW, PROJECT_SHA, ROOT, digest, need, read


BASE=ROOT/"training-gemma-original-model-v1/gemma-3-4b-it"
HISTORIC_FULL_SHA="e7004699fd4864dca94491300c8cd77604eb414c5bb6164a66d904408239ce82"
FILE_PLAN_SHA="5c6875cfe190980cad4315703e8d73dbd1fb803d620ea1f707ba35fe5de30c63"
FRESH_AUTH_SHA="3ebdcf197800c92c970233883e377ea86f10541bb7e20c44344dc6d03451334c"


def identity(info):
    return [info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns]


def truth(fresh):
    seen=fresh["response"]
    need(fresh["authority_verified"] is fresh["remote_received"] is True and fresh["rc"]==0
         and type(seen["asset_pass"]) is int and seen["asset_pass"]==seen["asset_checked"]==16 and seen["tensor_index_keys"]==883
         and seen["payload_bytes_rehashed"]==0 and seen["actual_ranks"]==seen["optimizer_steps"]==seen["slurm_steps_created"]==0
         and seen["private_projection_SHA256"]==PROJECT_SHA and seen["projection_slots"]==800 and seen["first_slots"]==64,
         "R39_EXACT_FRESH_MODEL16_AND_CPU_PROJECTION_RECEIPT")
    return seen["asset_bindings"]


def revalidate(model,manifest,observed,check):
    files=manifest["files"]
    need(model.resolve(strict=True)==model and set(files)==set(observed)=={p.name for p in model.iterdir()}
         and len(files)==16 and sum(value["bytes"] for value in files.values())==8639630561,
         "R39_EXACT_FULL_BASE16_CLOSURE")
    bindings={}
    for name,expected in files.items():
        check()
        path=model/name
        before=path.lstat()
        observed_file=observed[name]
        need(Path(name).name==name and stat.S_ISREG(before.st_mode) and path.resolve(strict=True)==path
             and before.st_size==expected["bytes"]==observed_file["bytes"] and stat.S_IMODE(before.st_mode)==observed_file["mode"],
             "R39_EXACT_MODEL16_FILE_IDENTITY")
        if name.endswith(".safetensors"):
            need(identity(before)[1:]==observed_file["bound_file_identity"][1:],"R39_SHARD_AUTHORITY_IDENTITY_CHANGED_NEEDS_REBIND")
            with path.open("rb") as stream:
                extent=int.from_bytes(stream.read(8),"little")
                need(extent==observed_file["header_bytes"] and 0<extent<=1<<20,"R39_EXACT_PINNED_SHARD_HEADER_EXTENT")
                header=stream.read(extent)
                after=os.fstat(stream.fileno())
            need(len(header)==extent and digest(header)==observed_file["header_SHA256"]
                 and identity(before)==identity(after)==identity(path.lstat()),"R39_EXACT_PINNED_SHARD_HEADER_AND_IDENTITY")
        else:
            need(expected["SHA256"]==observed_file["SHA256"],"R39_BOUNDED_METADATA_RECEIPT_IDENTITY")
            read(path,expected["SHA256"],64<<20,expected["bytes"])
        bindings[name]=identity(before)
    return bindings


def historical_authority(original,manifest,observed):
    all_files=manifest["files"]
    old_files=original["target_independent_file_SHA"]
    headers=original["target_weight_headers"]
    shards={"model-00001-of-00002.safetensors","model-00002-of-00002.safetensors"}
    need(original["model_files_READY"] is True and original["serialized_tensor_count"]==883
         and original["total_model_bytes"]==8639630561 and len(all_files)==16 and set(old_files)==set(all_files)
         and sum(record["bytes"] for record in all_files.values())==original["total_model_bytes"],
         "R40_HISTORICAL_BASE16_AND_883_NAMED_TENSORS")
    need(all(old_files[name]["SHA256"]==expected["SHA256"] and old_files[name]["bytes"]==expected["bytes"]
             and old_files[name]["stable_during_read"] is True for name,expected in all_files.items()),
         "R40_INDEPENDENT_HISTORICAL_ALL16_FULL_FILE_SHAS")
    need(set(headers)==set(original["original_weight_SHAs"])==shards
         and {name for name in all_files if name.endswith(".safetensors")}==shards
         and sum(headers[name]["tensor_count"] for name in shards)==883,
         "R40_HISTORICAL_EXACT_TWO_BF16_SHARD_HEADERS")
    payload=sum(headers[name]["weight_payload_bytes"] for name in shards)
    file_bytes=sum(all_files[name]["bytes"] for name in shards)
    overhead=sum(8+headers[name]["header_bytes"] for name in shards)
    need(original["weight_payload_bytes"]==payload==8600158944,"R40_HISTORICAL_TENSOR_ONLY_PAYLOAD_BYTES")
    need(original["raw_weight_transfer_bytes"]==file_bytes==8600277880 and overhead==118936
         and payload+overhead==file_bytes,"R40_HISTORICAL_TWO_FULL_SHARD_AND_FRAMING_BYTES")
    need(all(headers[name]["dtype"]=="BF16"
             and headers[name]["weight_payload_bytes"]+8+headers[name]["header_bytes"]==all_files[name]["bytes"]
             and original["original_weight_SHAs"][name]==all_files[name]
             and headers[name]["header_SHA256"]==observed[name]["header_SHA256"]
             and headers[name]["header_bytes"]==observed[name]["header_bytes"] for name in shards),
         "R40_EACH_SHARD_FRAMING_FULL_SHA_AND_FRESH_HEADER_CHAIN")
    return {"historically_hashed_full_model_bytes":original["total_model_bytes"],"tensor_payload_bytes":payload,
            "whole_shard_file_bytes":file_bytes,"header_and_prefix_bytes":overhead,"full_model_members":len(all_files)}


def from_target(check):
    binding=NEW/"authority"
    read(binding/"SOURCE_ASSET_INVENTORY.json",MODEL_INVENTORY_SHA)
    read(binding/"ORIGINAL_TARGET_BINDING.json",MODEL_BINDING_SHA)
    original=json.loads(read(binding/"HISTORIC_FULL_MODEL16.json",HISTORIC_FULL_SHA))
    manifest=json.loads(read(binding/"MODEL_FILES.json",FILE_PLAN_SHA))
    actual=json.loads(read(binding/"FRESH_MODEL16_READBACK.json",FRESH_AUTH_SHA))
    observed=truth(actual)
    historical_authority(original,manifest,observed)
    return revalidate(BASE,manifest,observed,check)
