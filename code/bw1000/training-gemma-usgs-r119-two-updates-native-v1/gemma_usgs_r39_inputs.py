"""Bind ordered production sampler indices and actual forwards to R39 only."""

from copy import deepcopy
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import re


HEX=re.compile(r"[0-9a-f]{64}")
R38_PROJECTION="5c4662c96f5f05cf89d85525d0beb5396d862c940b9a7bc07d739479958489bf"
REQUEST="6d14e9df9832432b97f1f46f032b49f3"


def need(ok,gate):
    if not ok:
        raise ValueError(gate)


def batch_gate_metadata(vals, int64):
    ids = vals["input_ids"]
    labels = vals["labels"]
    pixels = vals["pixel_values"]
    input_shape = tuple(ids.shape)
    label_shape = tuple(labels.shape)
    pixel_shape = tuple(pixels.shape)
    checks = {
        "token_dtype_int64": ids.dtype == int64,
        "label_dtype_int64": labels.dtype == int64,
        "token_label_shape_match": input_shape == label_shape,
        "token_rank_two": ids.ndim == 2,
        "batch_size_one": ids.ndim == 2 and input_shape[0] == 1,
        "at_most_512_tokens": ids.ndim == 2 and input_shape[1] <= 512,
        "pixels_floating": pixels.is_floating_point(),
        "pixels_rank_at_least_three": pixels.ndim >= 3,
    }
    return {
        "checks": checks,
        "shapes": {"input_ids": input_shape, "labels": label_shape, "pixel_values": pixel_shape},
        "dtypes": {"input_ids": str(ids.dtype), "labels": str(labels.dtype), "pixel_values": str(pixels.dtype)},
    }


def input_signature(batch, gate_failure=None):
    # Exact VizWiz native forward key/dtype domain; actual full hook hash is
    # recorded separately by the frozen fresh-domain callback.
    import torch
    from vlm.runtime.first_update_audit import fingerprint
    names=("input_ids","labels","pixel_values")
    need(all(name in batch and isinstance(batch[name],torch.Tensor) for name in names),"R39_NATIVE_BATCH_TEXT_LABEL_IMAGE_TENSORS")
    vals={name:batch[name].detach().cpu().contiguous() for name in names}
    if not (vals["input_ids"].dtype==vals["labels"].dtype==torch.int64 and vals["input_ids"].shape==vals["labels"].shape
         and vals["input_ids"].ndim==2 and vals["input_ids"].shape[0]==1 and vals["input_ids"].shape[1]<=512
         and vals["pixel_values"].is_floating_point() and vals["pixel_values"].ndim>=3):
        if gate_failure is not None:
            gate_failure(batch_gate_metadata(vals, torch.int64))
        need(False, "R39_EXACT_BS1_CAPPED_NATIVE_BATCH")
    vals["pixel_values"]=vals["pixel_values"].to(torch.bfloat16)
    fp=fingerprint(vals)
    need(fp["finite"] is True and HEX.fullmatch(fp["SHA256"]),"R39_FINITE_FULL_NATIVE_INPUT_BINDING")
    return fp["SHA256"]


def build_index(view,lineage,template,signature=input_signature,check=lambda:None):
    lines=view.splitlines(keepends=True)
    need(len(lines)==800 and sha256(view).hexdigest()==lineage["view"]["SHA256"],"R39_STAGED_800_VIEW")
    result={}
    new=set()
    old=set()
    replacements={item["physical_ordinal"]:item for item in lineage["replacements"]}
    for rank,ordinals in lineage["rank_first64"].items():
        records=[]
        for index,ordinal in enumerate(ordinals,1):
            check()
            raw=lines[ordinal-1]
            original_sha=sha256(raw).hexdigest()
            need(original_sha==lineage["selected_line_SHA256"][str(ordinal)],"R39_EXPECTED_PHYSICAL_LINE_DIGEST")
            row=json.loads(raw)
            # The template sees the same text/image columns it receives from
            # the actual Swift preprocessor. No row ID is used as model input.
            example={key:row[key] for key in ("messages","images")}
            batch=template.data_collator([template.encode(example)])
            sig=signature(batch)
            replacement=replacements.get(ordinal)
            if replacement is None:
                old.add(sig)
            else:
                new.add(sig)
            records.append({"ordinal":ordinal,"forward":index,"line_SHA256":original_sha,"input_SHA256":sig,
                "pair":replacement["pair"] if replacement else None,"sign":replacement["sign"] if replacement else None})
        result[rank]=records
    need(len(new)==12 and new.isdisjoint(old) and len(replacements)==12 and sum(len(r) for r in result.values())==64,
         "R39_NEW_SIGNATURES_UNIQUE_ACROSS_64_PROBED_SLOTS")
    return {"schema":"R39_NATIVE_CPU_64_PRODUCTION_FORWARD_INDEX_V1","request":REQUEST,
            "projection_SHA256":R38_PROJECTION,"view":lineage["view"],"rank":result,"distinct_new_signature_count":12,
            "count":64,"GPU":0,"optimizer_updates":0}


def bind_index(config):
    from vlm.runtime.resume_cursor import read_pinned_json
    document=read_pinned_json(Path(config["native_index"]),config["native_index_SHA256"],maximum_bytes=49152)
    ancestry=read_pinned_json(Path(config["lineage"]),config["lineage_SHA256"],maximum_bytes=65536)
    rank_doc=document["rank"]
    need(document["schema"]=="R39_NATIVE_CPU_64_PRODUCTION_FORWARD_INDEX_V1"
         and ancestry["schema"]=="R39_TRAIN_ONLY_800_PHYSICAL_PATCH_AND_3393_SLOTS_V1"
         and (document["request"],ancestry["request"])==(REQUEST,REQUEST)
         and document["projection_SHA256"]==ancestry["projection_SHA256"]==R38_PROJECTION
         and document["view"]==ancestry["view"] and document["view"]["SHA256"]==config["dataset_SHA256"]
         and document["view"]==config["view_pin"]
         and set(rank_doc)==set(ancestry["rank_first64"])=={str(i) for i in range(8)}
         and document["GPU"]==document["optimizer_updates"]==0 and document["count"]==64
         and ancestry["physical_rows"]==800 and ancestry["unchanged_rows"]==788
         and ancestry["replaced_rows"]==12 and ancestry["pairs"]==6 and not ancestry["read_blind_or_dev_body"],
         "R39_EXACT_RUNTIME_INDEX_AND_LINEAGE")
    expected={item["physical_ordinal"]:(item["pair"],item["sign"]) for item in ancestry["replacements"]}
    invariant=[]
    for rank,records in rank_doc.items():
        need(len(records)==8 and [item["ordinal"] for item in records]==ancestry["rank_first64"][rank],
             "R39_ACTUAL_ORDERED_RANK_INDEX")
        for i,item in enumerate(records,1):
            ordinal=item["ordinal"]
            need(item["forward"]==i and item["line_SHA256"]==ancestry["selected_line_SHA256"][str(ordinal)]
                 and HEX.fullmatch(item["input_SHA256"]) and (item["pair"],item["sign"])==expected.get(ordinal,(None,None)),
                 "R39_ONE_EXPECTED_NATIVE_FORWARD_IDENTITY")
            if ordinal in expected:
                invariant.append((item["pair"],item["sign"],item["input_SHA256"]))
    need(len({r[2] for r in invariant})==len(invariant)==12 and
         {(p,s) for p,s,_ in invariant}=={(i,s) for i in range(1,7) for s in ("positive","negative")},
         "R39_EXACT_ALL_PAIRED_FORWARD_TARGETS")
    return document,ancestry


def verify_sampler(loader,rank,ancestry,check):
    import torch
    from accelerate.data_loader import BatchSamplerShard,SeedableRandomSampler
    # Swift may wrap the dataset for lazy encoding. Only inspect an actual
    # Arrow index, and fail before any forward if the production lineage is lost.
    current=loader.dataset
    while type(current).__module__.split(".",1)[0]=="swift" and hasattr(current,"dataset"):
        other=current.dataset
        need(other is not current,"R39_NONCYCLIC_NATIVE_DATASET_WRAPPER")
        current=other
    need(type(current).__module__.startswith("datasets.") and len(current)==800 and current._indices is not None
         and current._indices.table.column_names==["indices"],"R39_PRODUCTION_NATIVE_ARROW_INDEX_REQUIRED")
    order=current._indices.table.column("indices").to_pylist()
    need(all(type(x) is int for x in order) and [x+1 for x in order]==ancestry["raw_ordinals_by_dataset_index"],
         "R39_PRODUCTION_STATE_MUST_MATCH_ACTUAL_CPU_PHYSICAL_LINEAGE")
    shard=loader.batch_sampler
    need(type(shard) is BatchSamplerShard and shard.process_index==rank and shard.num_processes==8
         and shard.split_batches is False and shard.even_batches is True and type(shard.batch_sampler) is torch.utils.data.BatchSampler,
         "R39_EXACT_LIVE_PRODUCTION_BATCH_SHARD")
    sampler=shard.batch_sampler.sampler
    need(type(sampler) is SeedableRandomSampler and sampler.initial_seed==42 and sampler.epoch==0
         and sampler.replacement is False and sampler.num_samples==800 and not shard.batch_sampler.drop_last
         and shard.batch_sampler.batch_size==1 and len(loader)==100,"R39_EXACT_PRODUCTION_SEED_BATCH_AND_EPOCH")
    check()
    clone=deepcopy(shard,{id(sampler.data_source):sampler.data_source})
    need(clone is not shard and clone.batch_sampler.sampler is not sampler,"R39_NO_LIVE_SAMPLER_MUTATION")
    slots=list(islice(iter(clone),8))
    need(len(slots)==8 and all(len(slot)==1 and type(slot[0]) is int and 0<=slot[0]<800 for slot in slots),
         "R39_EXACT_LIVE_PRODUCTION_SAMPLER_SLOTS")
    expected=[order[slot[0]]+1 for slot in slots]
    need(expected==ancestry["rank_first64"][str(rank)],"R39_EXACT_PRODUCTION_FIRST_TWO_RANK_SLOTS")
    return expected


def identify(batch,index,rank,microbatch,expected,gate_failure=None):
    item=index["rank"][str(rank)][microbatch]
    need(item["ordinal"]==expected[microbatch] and input_signature(batch, gate_failure=gate_failure)==item["input_SHA256"],
         "R39_ACTUAL_FORWARD_DIFFERS_FROM_PROVEN_NATIVE_SLOT")
    return item
