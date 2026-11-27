"""Resolve the exact prior target source, split files and native reporter."""
import importlib.util
import os
from pathlib import Path
import stat
import sys

import lineage


ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
SPLIT=ROOT/"results/internvl-gqa64-train56-internal8-aa810b24efcc4a5ab04a3cdd1c7825a4/output"
EXPOSURES=SPLIT/"proposed-exposures64.jsonl"
ORIGINAL=ROOT/"results/training-internvl-gqa-native-64-cpu-cfea7ada75b741449ce786c20974b61b"
TRAIN64=ORIGINAL/"data/train.jsonl"
PROCESSOR=ORIGINAL/"InternVL3_5-8B-HF"
FINGERPRINT=ROOT/"results/training-internvl-gqa-logit64-cpu-caac5aebc8e04fef9fd88f12ab69d952/result.json"
SAMPLER=ROOT/"incoming/internvl-gqa56-swift-sampler-v6-9056f6af43864cf4934af5026451d063/output/result.json"
FILES={
    "slot_guard.py":"cc4d441c4e961a524211bed8deee162b08a10b035cd3fa431aa470a81be1e4fa",
    "internvl_gqa_training_roster.py":"3bd9a38cbd6c1aae99a6365cdb8899bfe1c73e436359ee14da13cc55323163f1",
    "sampler_v6.py":"2896f9f6a30c9855b812b4c788896d861acc5fce485ba87e46aa25d535b00486",
    "original-launch.json":"7fefdc16f5519a998a94fdef703c2dd326b3c15e2e31d5a3735f6c4558269c67",
}


def pin(path,size,wanted,*,exact=True):
    lineage.need(path.is_absolute() and path.resolve(strict=True)==path,"CANONICAL_PRIVATE_INPUT")
    listed=path.lstat()
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_BINARY",0))
    with os.fdopen(fd,"rb") as stream:
        before=os.fstat(stream.fileno())
        raw=stream.read(size+1)
        after=os.fstat(stream.fileno())
    identity=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
    # Windows exposes ctime differently for fd and path. Both observations must remain stable.
    comparable=4 if os.name=="nt" else 5
    lineage.need(stat.S_ISREG(before.st_mode) and identity(before)==identity(after) and
                 identity(listed)==identity(path.lstat()) and identity(before)[:comparable]==identity(listed)[:comparable] and
                 0<len(raw)==before.st_size<=size and (not exact or len(raw)==size) and lineage.digest(raw)==wanted,
                 "EXACT_INPUT_OR_SOURCE_PIN")
    return raw


def from_scope(path):
    name=path.stem
    raw=pin(path,65536,FILES[path.name],exact=False)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    exec(compile(raw,str(path),"exec"),module.__dict__)
    return module


def load_members(here):
    read=lambda path,expected=None,extent=None,cap=32768:pin(path,extent if extent is not None else cap,expected,
                                                           exact=extent is not None)
    roster=from_scope(here/"internvl_gqa_training_roster.py")
    guard=from_scope(here/"slot_guard.py")
    train=pin(TRAIN64,19607,"78de19307549affa877be85756e7077bbb70d8d9feea09e1a8214157ee2fa000")
    prior=pin(FINGERPRINT,53728,"3b24a2b1f9f22408cc779129e335197de4eb513561369b297341608675a4e599")
    full=roster.build_roster(lineage.strict_json(prior),train,"caac5aebc8e04fef9fd88f12ab69d952")
    subset,_=guard.load(SPLIT,train,full,read)
    ordinal_to_row={row["ordinal"]:row["row_SHA256"] for row in subset.values()}
    e=pin(EXPOSURES,*guard.FILES[EXPOSURES.name])
    name="proposed-slots.json"
    slots=lineage.strict_json(pin(SPLIT/name,*guard.FILES[name]))
    frozen=lineage.freeze_physical(e,slots,ordinal_to_row,set(ordinal_to_row))
    matrix=lineage.reference_matrix(pin(SAMPLER,65536,lineage.SAMPLER_RESULT_SHA,exact=False))
    command=lineage.strict_json(pin(here/"original-launch.json",65536,FILES["original-launch.json"],exact=False))["command"]
    return frozen,matrix,command
