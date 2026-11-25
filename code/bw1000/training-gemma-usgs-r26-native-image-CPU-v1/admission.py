"""Frozen R26 data admission and R27 Gemma image CPU scope."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import stat


LABEL="training-gemma-usgs-r26-native-image-CPU-v1"
RESULTS=PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
FRAMEWORK_HOST=RESULTS/"training-gemma-usgs72-fresh-initial-domain25-v1/framework"
FRAMEWORK_MANIFEST_HOST=FRAMEWORK_HOST.parent/"fresh-framework-files.json"
FRAMEWORK_MANIFEST=(55787,"40ed7488227fffc4da0a34353fc711c8720c0ebc407040fc61e748b2a7d24329")
PROCESSOR_HOST=RESULTS/"training-gemma-processor-assets-v1/models/gemma-3-4b-it"
PROCESSOR_MANIFEST=(2107,"eaab7b89d89e5cbe27391c3601ab36acfa5aa35f8c0ea18f6f1730d5424d228b")
WORK_SECONDS=120
ROWS=12
PAIRS=6
INDEX=(8943,"da46986c6799414edff4a57de0112bc72e1fc64e21334141b83c14d26b2b93e1")
TRAIN_SHA="3187ae7f69f10ed58c4135d7d253a2675f81375b94f24d53343eb4abf1edf4ae"
PAIR_INDEX=(4586,"3edce8e0e5962aedba12e0cf02c6f8bdd0a5b889094204bf6ccf0e7c0cebd9f4")
INDEPENDENT_R26="0e350844d5db94c24f687407622aebd3444fe771bd9a4d0bc2eecc01855f4dc3"
LEGACY_AUDIT=(9368,"890b4ebcea30972f604ffaa1d4b7f5cca2b76e919722944123fa3368fd29b1ba")
LEGACY_EXPOSURE=(4747,"c0c380c51ed4af09779fa270b4419ce511571137d95f1676b3dd9a4eaad715a7")
VERSIONS={"torch":"2.10.0+das.opt1.dtk2604.2606161421.g91a580","transformers":"5.12.1",
          "ms-swift":"4.4.2","tokenizers":"0.22.2","Pillow":"11.3.0"}
HEX=re.compile("[0-9a-f]{64}")
PICTURE=re.compile(r"train/media/[0-9]{3}\.png")
DELTA=re.compile(r"FINAL: -?(?:0|[1-9][0-9]*)\.[0-9]{2}")


def need(ok,gate):
    if not ok:
        raise ValueError(gate)


def pin(raw):
    return {"bytes":len(raw),"SHA256":sha256(raw).hexdigest()}


def read(path,size,digest,cap=2<<20):
    a=path.lstat()
    need(stat.S_ISREG(a.st_mode) and path.resolve(strict=True)==path and 0<a.st_size<=cap
         and (size is None or size==a.st_size),"PINNED_REGULAR_FILE")
    with path.open("rb") as stream:
        raw=stream.read(cap+1)
    b=path.lstat()
    def identity(info):
        return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns
    need(len(raw)==a.st_size and identity(a)==identity(b) and pin(raw)["SHA256"]==digest,"PINNED_FILE_IDENTITY")
    return raw


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def admit(data, deadline=lambda:None, *, pair_path=None):
    """Verify exact sealed source bytes, images and pairing before any native work."""
    index_raw=read(data/"CANDIDATE_INDEX.json",*INDEX)
    index=json.loads(index_raw)
    pair_raw=read(data/"PAIR_INDEX.json" if pair_path is None else pair_path,*PAIR_INDEX)
    pair_doc=json.loads(pair_raw)
    need(index["schema"]=="USGS_R26_PUBLIC_JULY_2024_TRAIN_ONLY_DEPTH_CHANGE_CANDIDATE_V1"
         and index["training_ready"] is False and index["actual_optimizer_updates"]==0
         and pair_doc["source_candidate_SHA256"]==INDEX[1]
         and pair_doc["independent_verification_SHA256"]==INDEPENDENT_R26
         and pair_doc["train_only"] is True and pair_doc["old_reuse_rows_in_this_index"]==0
         and len(pair_doc["pairs"])==PAIRS,"EXACT_R26_NEW_TRAIN_ONLY_ORIGIN")
    files=index["files"]
    train="train-depth-change.jsonl"
    images={name for name in files if PICTURE.fullmatch(name)}
    need(set(files)==images|{train} and len(images)==ROWS and files[train]["SHA256"]==TRAIN_SHA,
         "EXACT_R26_FILE_INVENTORY")
    raw={name:read(data/name,value["bytes"],value["SHA256"]) for name,value in files.items()}
    train_raw=raw[train]
    lines=train_raw.splitlines(keepends=True)
    need(len(lines)==ROWS and train_raw.endswith(b"\n"),"EXACT_PHYSICAL_ROW_DENOMINATOR")
    entries={}
    charts={}
    for line in lines:
        row=json.loads(line)
        need(set(row)=={"chart_id","event_ids","group","id","images","kind","messages"}
             and row["kind"]=="depth_change" and HEX.fullmatch(row["chart_id"])
             and row["group"]==row["chart_id"] and row["id"]==row["chart_id"]+":q4"
             and row["id"] not in entries and len(row["images"])==1 and row["images"][0] in images
             and row["event_ids"]==[f"E{i:02d}" for i in range(1,7)],"NEW_IMAGE_TRAIN_ROW_IDENTITY")
        text=row["messages"]
        need(len(text)==2 and all(set(item)=={"content","role"} for item in text)
             and [item["role"] for item in text]==["user","assistant"]
             and text[0]["content"].startswith("<image>\n") and text[0]["content"].count("<image>")==1
             and DELTA.fullmatch(text[1]["content"]) and text[1]["content"] not in ("FINAL: 0.00","FINAL: -0.00"),
             "REAL_IMAGE_AND_NONZERO_SIGNED_DELTAS")
        name=row["images"][0]
        need(name not in charts,"TRAIN_IMAGE_REUSE")
        charts[name]=row["chart_id"]
        entries[row["id"]]=(row,pin(line)["SHA256"])
    chart_ids={chart["chart_id"] for chart in index["chart_identity"]}
    need(set(charts)==images and set(charts.values())==chart_ids
         and len(chart_ids)==len(index["chart_identity"])==ROWS,"EXACT_NEW_CHART_ROSTER")
    pair_ids={}
    used=set()
    for ordinal,pair in enumerate(pair_doc["pairs"],1):
        need(set(pair)=={"positive","negative"},"EXACT_PAIRED_ARMS")
        for sign,identity in pair.items():
            need(set(identity)=={"chart_id","image","image_SHA256","row_id"},"PAIR_IDENTITY_SCHEMA")
            row_id,name=identity["row_id"],identity["image"]
            row,_=entries[row_id]
            negative=row["messages"][1]["content"].startswith("FINAL: -")
            need(row_id not in used and row["images"]==[name] and row["chart_id"]==identity["chart_id"]
                 and files[name]["SHA256"]==identity["image_SHA256"] and negative==(sign=="negative"),
                 "EXACT_NEW_PAIR_BINDING")
            used.add(row_id)
            pair_ids[row_id]={"pair_ordinal":ordinal,"sign":sign}
    need(used==set(entries),"PAIR_INDEX_FULL_DENOMINATOR")
    need(len({files[name]["SHA256"] for name in images})==ROWS,"UNIQUE_PNG_BYTES")
    for name in images:
        deadline()
        need(raw[name][:24]==b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x02\x00\x00\x00\x02\x00",
             "ACTUAL_512_SQUARE_PNG")
    return {"raw":{**raw,"CANDIDATE_INDEX.json":index_raw,"PAIR_INDEX.json":pair_raw},
            "entries":entries,"pairs":pair_ids,"files":files,"chart_count":len(charts)}
