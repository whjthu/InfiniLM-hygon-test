"""Build one private R39 training view from target-resident train-only evidence."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import stat


REQUEST="6d14e9df9832432b97f1f46f032b49f3"
ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
LABEL="training-gemma-usgs-r39-two-updates-v1"
NEW=ROOT/LABEL/REQUEST
ORIGINAL=ROOT/"training-gemma-usgs72-fresh-initial-domain25-v1/view/train800-swift.jsonl"
PROJECT=ROOT/"training-gemma-usgs-r38-independent-message-media-CPU-v1/cad91f70c6414062bd9d0f5f363befbb/out/PRIVATE_ORDER.json"
R26=ROOT/"training-gemma-usgs-r26-native-image-CPU-v1/537844da9e9b470993209b74ad45c7f7/data"
MEDIA_OLD=ROOT/"training-gemma-usgs-render72-native-CPU-v1/data/train/media"
ORIGINAL_PIN=(602794,"74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc")
PROJECT_SHA="5c4662c96f5f05cf89d85525d0beb5396d862c940b9a7bc07d739479958489bf"
PROJECT_FIRST_SHA="4dd6a3d7e3fe1184f5aa0c1579e32433d9ae57faacefe67b6c10b314e9f3f390"
R38_SHA="4a993068b26bd0aefb31d83c2d2bfa6728d9e1db3ac7dee00491b30f38c28a9c"
R27_SHA="dd2a5c08fe34003ae0ca7822bbd95ae8d094c5437107a25112684cc6a41d15a9"
R26_ADMIT_SHA="cbdb51098763e9691c0a21fe57163b08fce05b1d95c2a76c73e906edf97368ef"
PLAN_SHA="12d67618d386b04ac973e59e4580f32e8cb938e77d7322fb2bd5e4c31bf64de7"
MODEL_INVENTORY_SHA="b9d97db598ac4255c3beab588c04ee06f1a3fa8ec047534a98ecd9967ccdbb0c"
MODEL_BINDING_SHA="46428e251757e386835e2832a32308a6a07178d9ef776a3ea1b70062c5cf77b3"
MASK=re.compile(r"[0-9a-f]{64}")


def need(ok,why):
    if not ok:
        raise ValueError(why)


def digest(raw):
    return sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def pin(raw):
    return {"bytes":len(raw),"SHA256":digest(raw)}


def read(path,expected,cap=2097152,size=None):
    info=path.lstat()
    need(stat.S_ISREG(info.st_mode) and path.resolve(strict=True)==path and 0<info.st_size<=cap
         and (size is None or info.st_size==size),"R39_PRIVATE_REGULAR_EXACT_INPUT")
    stable=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns) if os.name=="nt" else (
        s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    with path.open("rb") as stream:
        opened=os.fstat(stream.fileno())
        raw=stream.read(cap+1)
        after=os.fstat(stream.fileno())
    final=path.lstat()
    need(stable(info)==stable(opened)==stable(after)==stable(final) and info.st_ctime_ns==final.st_ctime_ns and len(raw)==info.st_size
         and digest(raw)==expected,"R39_EXACT_FROZEN_BYTES")
    return raw


def lines(raw,n):
    values=raw.splitlines(keepends=True)
    need(len(values)==n and all(line.endswith(b"\n") for line in values),"R39_EXACT_JSONL_PHYSICAL_DENOMINATOR")
    return values


def projection_contract(doc):
    require_slots=doc["two_optimizer_boundaries_projection_only"]
    ranks=doc["ranks"]
    logical=doc["raw_ordinals_by_dataset_index"]
    need(doc["schema"]=="R29_NATIVE_CPU_PHYSICAL_PROJECTION_V1"
         and doc["step"]=="1103606.3393" and doc["request"]=="cad91f70c6414062bd9d0f5f363befbb"
         and doc["baseline_view_SHA256"]==ORIGINAL_PIN[1] and doc["production_adoption"] is False
         and set(ranks)=={str(i) for i in range(8)} and all(type(x) is list and len(x)==100 for x in ranks.values())
         and sorted(x for rank in ranks.values() for x in rank)==sorted(logical)==list(range(1,801))
         and all(type(x) is int for x in logical+[x for rank in ranks.values() for x in rank])
         and logical!=list(range(1,801)) and require_slots=={k:[v[:4],v[4:8]] for k,v in ranks.items()},
         "R39_ACTUAL_3393_PHYSICAL_AND_NATIVE_SAMPLER_PROJECTION")
    slots={key:a+b for key,(a,b) in require_slots.items()}
    need(len({n for slots in slots.values() for n in slots})==64,"R39_UNIQUE_FIRST64_NATIVE_ORDINALS")
    return logical,slots


def original_row(line):
    row=json.loads(line)
    need(isinstance(row,dict) and type(row["images"]) is list and len(row["images"])==1
         and type(row["images"][0]) is str and Path(row["images"][0]).parent==MEDIA_OLD
         and re.fullmatch(r"[0-9]{3}\.png",Path(row["images"][0]).name)
         and [m["role"] for m in row["messages"]]==["user","assistant"]
         and all(type(m["content"]) is str for m in row["messages"]),"R39_EXACT_ORIGINAL_TRAIN_MEDIA")
    return row


def build_view(original,proof,data,media_host=R26):
    physical=lines(original,800)
    logical,slots=projection_contract(proof)
    original_rows=[original_row(line) for line in physical]
    need(len({row["images"][0] for row in original_rows})==12,"R39_ORIGINAL_TRAIN_IMAGES")
    paired={(value["pair_ordinal"],value["sign"]):key for key,value in data["pairs"].items()}
    need(len(data["entries"])==12 and set(paired)=={(n,sign) for n in range(1,7) for sign in ("positive","negative")},
         "R39_SIX_INDEPENDENT_NEW_TRAIN_PAIRS")
    replacements={}
    for pair in range(1,7):
        rank=str(pair-1)
        inner=(pair-1)%4
        signs=("positive","negative") if pair%2 else ("negative","positive")
        for update,sign in enumerate(signs):
            slot=4*update+inner
            ordinal=slots[rank][slot]
            row_id=paired[(pair,sign)]
            row,source_pin=data["entries"][row_id]
            image,=row["images"]
            need(re.fullmatch(r"train/media/[0-9]{3}\.png",image) and row["id"]==row_id and MASK.fullmatch(source_pin),
                 "R39_EXACT_NEW_TRAIN_ROW_PATH_AND_IDENTITY")
            inserted=encoded({**row,"images":[str(media_host/image)]})
            index=ordinal-1
            old=physical[index]
            need(index not in replacements and inserted!=old and Path(json.loads(inserted)["images"][0]).parent!=MEDIA_OLD,
                 "R39_EXACT_UNIQUE_REPLACEMENT_PHYSICAL_SLOT")
            replacements[index]={"physical_ordinal":ordinal,"rank":rank,"rank_forward":slot+1,"update":update+1,"pair":pair,"sign":sign,
                "source_id_SHA256":digest(row_id.encode()),"source_row_SHA256":source_pin,
                "source_image_SHA256":data["files"][image]["SHA256"],"old_line_SHA256":digest(old),"new_line_SHA256":digest(inserted)}
            physical[index]=inserted
    original_lines=lines(original,800)
    changed=[i for i,(old,new) in enumerate(zip(original_lines,physical,strict=True)) if old!=new]
    need(changed==sorted(replacements) and len(changed)==12 and len({replacements[i]["new_line_SHA256"] for i in changed})==12,
         "R39_PROVE_EXACT_12_REPLACED_AND_788_UNCHANGED")
    view=b"".join(physical)
    lineage={"schema":"R39_TRAIN_ONLY_800_PHYSICAL_PATCH_AND_3393_SLOTS_V1","request":REQUEST,
        "base_view":pin(original),"view":pin(view),"projection_SHA256":PROJECT_SHA,
        "physical_rows":800,"unchanged_rows":788,"replaced_rows":12,"pairs":6,
        "raw_ordinals_by_dataset_index":logical,"rank_first64":slots,"replacements":[replacements[i] for i in changed],
        "selected_line_SHA256":{str(ord_):digest(physical[ord_-1]) for row in slots.values() for ord_ in row},
        "read_blind_or_dev_body":False,"actual_optimizer_updates":0,"production_ready":False}
    return view,lineage


def archive(repo):
    r=repo/"results/bw1000/2026-11-25"
    a=json.loads(read(r/"training-r38-gemma-3393-800-rows-1600-messages-64-native-sampler-pass.json",R38_SHA))
    b=json.loads(read(r/"training-r27-gemma-usgs-native-image-CPU-3308-terminal.json",R27_SHA))
    need(a["accounting"][0]=="1103606.3393" and a["accounting"][2:4]==["COMPLETED","0:0"]
         and a["accepted"]["private_projection_SHA256"]==PROJECT_SHA and a["native_first_two_rank_slots"]["private_sha256"]==PROJECT_FIRST_SHA
         and a["accepted"]["optimizer_updates"]==0 and a["accepted"]["arrow_physical_rows"]==800
         and a["accepted"]["first_two_slots"]==64 and b["native_train_pairs"]==6 and b["native_train_only_rows"]==12
         and b["observations"]["independent_actual_TERMINAL"]["response"]["step"]=="1103606.3308","R39_EXACT_REAL_3393_AND_3308")


def emit_file(path,raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def stage(repo):
    archive(repo)
    admit_path=(repo/"code/bw1000/training-gemma-usgs-r26-native-image-CPU-v1/admission.py").resolve()
    read(admit_path,R26_ADMIT_SHA)
    admitted=runpy.run_path(str(admit_path))["admit"](R26)
    original=read(ORIGINAL,ORIGINAL_PIN[1],size=ORIGINAL_PIN[0])
    project=json.loads(read(PROJECT,PROJECT_SHA,65536))
    need(digest(encoded(project["two_optimizer_boundaries_projection_only"]))==PROJECT_FIRST_SHA,"R39_ACTUAL_FIRST64_PRIVATE_DIGEST")
    view,meta=build_view(original,project,admitted)
    os.umask(0o077)
    need(ROOT.resolve(strict=True)==ROOT,"R39_EXACT_TARGET_RESULTS_ROOT")
    NEW.parent.mkdir(mode=0o700,exist_ok=True)
    need(NEW.parent.resolve(strict=True)==NEW.parent,"R39_EXACT_TARGET_OWNER_PARENT")
    NEW.mkdir(mode=0o700)
    emit_file(NEW/"train800-swift.jsonl",view)
    raw=encoded(meta)
    emit_file(NEW/"lineage-private.json",raw)
    result={"request":REQUEST,"view":pin(view),"lineage":pin(raw),"old_bytes_unchanged":788,"new_train_rows":12,
            "new_pairs":6,"planned_rows":800,"target_GPU":0,"actual_optimizer_updates":0,"ready":False}
    emit_file(NEW/"STAGE_VIEW.json",encoded(result))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(stage(a.repo.resolve(strict=True)),sort_keys=True))
