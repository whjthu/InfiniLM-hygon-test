"""Independent real local views/media audit. No native encoder or model imports."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
import zlib


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned(path,pin,cap=2*1024**2):
    before=path.lstat()
    need(path.is_file() and not path.is_symlink() and 0<before.st_size==pin["bytes"]<=cap,"bounded original data member differs")
    raw=path.read_bytes()
    after=path.lstat()
    need(all(getattr(before,k)==getattr(after,k) for k in ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns"))
         and digest(raw)==pin["SHA256"],"original or derived data bytes changed")
    return raw


def png_structure(raw):
    need(raw[:8]==b"\x89PNG\r\n\x1a\n","PNG signature differs")
    offset=8
    dimensions=None
    software=None
    ended=False
    chunks=0
    while offset<len(raw):
        need(offset+12<=len(raw),"PNG chunk header missing")
        length=struct.unpack(">I",raw[offset:offset+4])[0]
        tag=raw[offset+4:offset+8]
        need(length<=1024**2 and offset+length+12<=len(raw),"PNG chunk boundary differs")
        body=raw[offset+8:offset+8+length]
        crc=struct.unpack(">I",raw[offset+8+length:offset+12+length])[0]
        need(zlib.crc32(tag+body)&0xffffffff==crc,"PNG chunk CRC differs")
        if tag==b"IHDR":
            need(chunks==0 and length==13,"PNG IHDR differs")
            dimensions=list(struct.unpack(">II",body[:8]))
        if tag==b"tEXt" and body.startswith(b"Software\0"):
            software=body.split(b"\0",1)[1].decode("latin1")
        offset+=length+12
        chunks+=1
        if tag==b"IEND":
            need(length==0 and offset==len(raw),"PNG trailing bytes differ")
            ended=True
            break
    need(ended and dimensions==[512,512] and software=="training-USGS-multievent-render144-v2",
         "PNG dimensions or frozen rendering identity differs")
    return dict(pixels=dimensions,Software=software,CRC_all_chunks=True)


def check(dataset,prepared,plan):
    dataset,prepared=Path(dataset),Path(prepared)
    raw=pinned(dataset/"manifest-private.json",plan["dataset_manifest"])
    m=json.loads(raw)
    need(m["Software"]=="training-USGS-multievent-render144-v2" and m["actual_split_QA"]==dict(train=72,dev=36,blind=36),
         "Frozen dataset version/counts differ")
    charts=m["charts"]
    need(len(charts)==24 and len({c["id"] for c in charts})==24,"Chart metadata denominator differs")
    by_split={s:[c for c in charts if c["split"]==s] for s in ("train","dev","blind")}
    groups={s:{c["group"] for c in cs} for s,cs in by_split.items()}
    events={s:{e for c in cs for e in c["event_ids"]} for s,cs in by_split.items()}
    need({s:len(cs) for s,cs in by_split.items()}==dict(train=12,dev=6,blind=6),"Chart split counts differ")
    need({s:len(g) for s,g in groups.items()}==dict(train=3,dev=2,blind=2),"Closure group counts differ")
    need(sum(len(e) for e in events.values())==144 and all(len(c["event_ids"])==6 for c in charts),"Unique event metadata differs")
    for a,b in (("train","dev"),("train","blind"),("dev","blind")):
        need(not groups[a]&groups[b] and not events[a]&events[b],"Frozen group/event metadata crosses split")
    image_pins=[m["outputs"][c["image"]]["SHA256"] for c in charts]
    need(len(set(image_pins))==24,"Identical rendered image identity crosses chart groups")
    binding=json.loads((prepared/"data-binding-private.json").read_bytes())
    need(binding["case"]=="T02.sft" and binding["max_sequence_length"]==512
         and binding["dataset_manifest"]==plan["dataset_manifest"] and binding["dev_gold_read"] is False
         and binding["blind_inputs_gold_or_media_read"] is False,"Gemma split/input contract differs")
    source_names=("train/inputs.jsonl","train/labels-private.jsonl","dev/inputs.jsonl")
    original={n:[json.loads(x) for x in pinned(dataset/n,m["outputs"][n]).splitlines()] for n in source_names}
    train_src,train_labels,dev_src=(original[n] for n in source_names)
    need([len(train_src),len(train_labels),len(dev_src)]==[72,72,36],"Actual source rows differ")
    targets={}
    for name,count in (("train-swift.jsonl",72),("dev-swift.jsonl",36)):
        body=pinned(prepared/name,binding["files"][name])
        values=[json.loads(x) for x in body.splitlines()]
        need(len(values)==count and body.count(b"\n")==count,"Actual JSONL physical rows differ")
        targets[name]=values
    gold={x["id"]:x for x in train_labels}
    need(len(gold)==72,"Training label IDs differ")
    target=PurePosixPath(plan["declared_target_data"])
    for split,source_rows,dest in (("train",train_src,targets["train-swift.jsonl"]),("dev",dev_src,targets["dev-swift.jsonl"])):
        for original_row,actual in zip(source_rows,dest,strict=True):
            expected=dict(original_row)
            need(expected.pop("labels") is None and len(expected["messages"])==1
                 and expected["messages"][0]["role"]=="user","Source is not full answer-free user input")
            if split=="train":
                label=gold[expected["id"]]
                need(all(label[k]==expected[k] for k in ("chart_id","group","kind")),"Actual label join changes source group")
                expected["messages"]=[dict(expected["messages"][0]),dict(role="assistant",content="FINAL: "+label["answer"])]
            expected["images"]=[str(target/name) for name in expected["images"]]
            need(expected==actual,"Actual training/dev view changes source input or supervision")
    media=binding["media"]
    need(len(media)==18,"Exact train+dev media denominator differs")
    for name,item in media.items():
        p=dataset/name
        need(p.resolve().is_relative_to(dataset.resolve()) and name.split("/")[0] in ("train","dev"),"Media escapes permitted split")
        png_structure(pinned(p,m["outputs"][name],1024**2))
        need(item["SHA256"]==m["outputs"][name]["SHA256"] and item["target_presence_verified"] is False,"Media staging identity differs")
    return dict(passed=True,state="GEMMA_RENDER72_LOCAL_VIEW_AND_MEDIA_AUDITED_TARGET_NOT_READY",
         case="T02.sft",train_rows=72,dev_user_only_rows=36,train_charts=12,dev_charts=6,
         inspected_PNG=18,PNG_dimension=[512,512],PNG_all_chunks_CRC=True,
         train_kinds=dict(Counter(x["kind"] for x in targets["train-swift.jsonl"])),
         dev_kinds=dict(Counter(x["kind"] for x in targets["dev-swift.jsonl"])),
         manifest_closure_groups={s:len(g) for s,g in groups.items()},
         manifest_distinct_events={s:len(e) for s,e in events.items()},
         group_event_cross_split_overlap=0,manifest_image_SHA_duplicates=0,
         frozen_manifest_metadata_only_for_blind=True,blind_inputs_labels_media_read=False,dev_gold_read=False,
         actual_native_encode_calls=0,actual_target_copy_calls=0,target_presence_verified=False,
         native_512_full_sequence_EOS_mask_qualification="PENDING",Gemma_original_initial_target_qualification="PENDING",
         actual_optimizer_updates=0,training_READY=False,GPU_GO=False,Model_GO=False)
