"""Identity and accounting for the original 41-row street-scene input stream."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re


BASE = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
NATIVE_REQUEST = "04a2af4f4b5b4e44bf5411cf32606bb9"
STAGE = BASE / ("training-qwen-textvqa-native-input-CPU-" + NATIVE_REQUEST)
DATASET = STAGE / "inputs/TRAIN_SWIFT.jsonl"
NATIVE_OUT = STAGE / "worker.out"
DATA_SHA = "dd6c882665db32a47ff4736392b63f0d83e4ea0caa226534ec2f1093c6c04ecc"
MEDIA_SHA = "0b2ff0a69ccee6bde9d91445b66754de46b80691a1bf93e5d1fea74d3450947d"
ZIP_SHA = "ba880f5e08551dcdefb7a83e866f682781c8638aa7efada8fd5a0c744e2a8df6"
TRANSFER_SHA = "b5e2563b1437bec31b24f32ee546b1e8fa73b2bd64f9e52e7b9cf60a70558743"
NATIVE_REPORT_SHA = "770c6215ab2016a66cc9996a51dd7aaa64e64d1bbe37e6189d5c1ee042adc019"
NATIVE_PUBLIC_SHA = "d0b845d5b71656ab600afcf3b53038b1d9866568a2ed50500682acf1d11e0086"
ROWS, WORLD, GA, NOMINAL_STEPS, OBSERVED_UPDATES = 41, 8, 4, 25, 2
RANK_EPOCH_BATCHES = (ROWS + WORLD - 1) // WORLD
TASKS = {"VQA":26,"OCR":15}
SHA = re.compile(r"[0-9a-f]{64}")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(value):
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    return sha256(raw).hexdigest()


def read(path, maximum, expected=None, length=None):
    path=Path(path)
    need(path.is_file() and path.resolve(strict=True)==path.absolute() and not path.is_symlink(),"EXACT_PRIVATE_FILE")
    with path.open("rb") as stream:
        raw=stream.read(maximum+1)
    need(0<len(raw)<=maximum and (expected is None or sha256(raw).hexdigest()==expected)
         and (length is None or len(raw)==length),"EXACT_PRIVATE_BYTES")
    return raw


def dataset_rows(dataset, inventory):
    raw=read(dataset,17941,DATA_SHA,17941)
    media=json.loads(read(inventory,1936,MEDIA_SHA,1936))
    need(len(media)==16,"EXACT_MEDIA_SET")
    for name, item in media.items():
        need(re.fullmatch(r"[a-f0-9]{16}\.jpg",name),"EXACT_MEDIA_NAME")
        read(Path(dataset).parent/name,item["bytes"],item["SHA256"],item["bytes"])
    result=[]
    for line in raw.splitlines():
        row=json.loads(line)
        need(set(row)=={"images","messages"} and len(row["images"])==1 and row["images"][0] in media
             and [r["role"] for r in row["messages"]]==["user","assistant"]
             and row["messages"][0]["content"].startswith("<image>\n"),"EXACT_TRAINING_ROW")
        result.append(sha256(line).hexdigest())
    need(len(result)==len(set(result))==ROWS,"EXACT_ORIGINAL_41_UNIQUE_TRAINING_ROWS")
    return result


def native_report(log):
    matches=[]
    for line in read(log,262144).splitlines():
        if sha256(line).hexdigest()==NATIVE_REPORT_SHA:
            matches.append(json.loads(line))
    need(len(matches)==1,"EXACT_NATIVE_1251_REPORT_LINE")
    report=matches[0]
    need(report["state"]=="ACTUAL_NATIVE_TRAINING_INPUTS_CHECKED" and report["processed"]==report["planned"]==ROWS
         and report["dataset_SHA256"]==DATA_SHA and report["media_inventory_SHA256"]==MEDIA_SHA
         and report["GPU"]==report["optimizer_updates"]==0 and report["collator_passed"] is True
         and report["model_materialized"] is False and report["failures"]==[]
         and report["provider_versions"]=={"ms-swift":"4.4.2","transformers":"5.12.1","qwen-vl-utils":"0.0.14"},
         "EXACT_NATIVE_1251_TRAIN_ONLY_IDENTITY")
    return report["rows"]


def fingerprints(rows, raw_rows):
    need(len(rows)==len(raw_rows)==ROWS and Counter(row["task"] for row in rows)==TASKS,"EXACT_41_ROW_CASES")
    result={}
    for ordinal, row in enumerate(rows):
        key=(row["input_ids_SHA256"],row["labels_SHA256"])
        need(row["row"]==ordinal and all(SHA.fullmatch(value) for value in key)
             and key not in result and 0<row["supervised_tokens"]<row["tokens"]<=4096
             and len(row["grid"])==3 and all(type(v) is int and v>0 for v in row["grid"]),
             "NATIVE_ROW_FINGERPRINT_MISSING_OR_AMBIGUOUS")
        result[key]={"row":ordinal,"record_SHA256":raw_rows[ordinal],"task":row["task"],
                     "tokens":row["tokens"],"supervised_tokens":row["supervised_tokens"],"grid":row["grid"]}
    return result


def identify(index, ids, labels, grid):
    row=index.get((pin(ids),pin(labels)))
    need(row is not None and len(ids)==len(labels)==row["tokens"] and grid==row["grid"]
         and sum(label!=-100 for label in labels)==row["supervised_tokens"],"ACTUAL_BATCH_HAS_NO_EXACT_ORIGINAL_ROW")
    return row


def schedule_position(rank, seen):
    need(type(rank) is int and 0<=rank<WORLD and type(seen) is int and seen>=0,"EXACT_RANK_BATCH_ORDINAL")
    epoch, batch=divmod(seen,RANK_EPOCH_BATCHES)
    slot=batch*WORLD+rank
    return {"epoch":epoch,"batch_in_rank_epoch":batch,"global_slot_in_epoch":slot,
            "sampler_padding":slot>=ROWS}


def summarize(ranks):
    need(len(ranks)<=WORLD and all(type(r) is int and 0<=r<WORLD for r in ranks),"EXACT_RANK_INVENTORY")
    updates=[]
    consumed=[]
    for step in (1,2):
        current=[(rank,records[step]) for rank,records in sorted(ranks.items()) if step in records]
        if not current:
            continue
        observed=set()
        micro=[]
        for rank, record in current:
            need(record["rank"]==rank and record["global_step"]==step and 0<len(record["batches"])<=GA,
                 "EXACT_OBSERVED_UPDATE_BOUNDARY")
            observed.add(len(record["batches"]))
            micro.extend(record["batches"])
        need(len(observed)==1,"RANKS_DISAGREE_ON_ACTUAL_UPDATE_BATCH")
        complete=len(current)==WORLD
        need(not complete or len({row["parameters_after_SHA256"] for _,row in current})==1,
             "RANKS_DISAGREE_ON_TRAINABLE_UPDATE_HASH")
        updates.append({"global_step":step,"ranks":len(current),"actual_examples":len(micro),
                        "sampler_padding_examples":sum(row["sampler_padding"] for row in micro),
                        "observed_nonzero_update":complete and all(record["parameter_bytes_changed"] for _,record in current),
                        "ranks_confirmed_optimizer_advance":sum(record["counter_advance"] for _,record in current)})
        consumed.extend(micro)
    groups={}
    for row in consumed:
        index=row["global_slot_in_epoch"]
        epoch=row["epoch"]
        need(0<=row["row"]<ROWS and type(epoch) is int and epoch>=0 and 0<=index<RANK_EPOCH_BATCHES*WORLD
             and row["sampler_padding"]==(index>=ROWS),"EXACT_ACTUAL_SAMPLER_POSITION")
        slots=groups.setdefault(epoch,{})
        need(index not in slots,"DUPLICATE_GLOBAL_SAMPLER_SLOT")
        slots[index]=row["row"]
    for slots in groups.values():
        unpadded=[row for k,row in slots.items() if k<ROWS]
        need(len(unpadded)==len(set(unpadded)) and all(slots.get(k-ROWS,row)==row for k,row in slots.items() if k>=ROWS),
             "NATIVE_SAMPLER_PADDING_OR_PERMUTATION_DIVERGED")
    qualified=[row["global_step"] for row in updates if row["observed_nonzero_update"]
               and row["ranks_confirmed_optimizer_advance"]==WORLD]
    rows=[row["row"] for row in consumed]
    return {"planned_optimizer_updates":OBSERVED_UPDATES,"nominal_schedule_steps":NOMINAL_STEPS,
            "configured_effective_batch":WORLD*GA,"original_training_rows":ROWS,"rank_receipts":len(ranks),
            "qualified_nonzero_update":min(qualified,default=None),"observed_slots":len(rows),
            "distinct_original_rows":len(set(rows)),"repeated_slots":len(rows)-len(set(rows)),"updates":updates,
            "native_sampler_verified_all_original_rows":any(set(range(ROWS))<=set(slots) for slots in groups.values())}
