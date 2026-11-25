"""Bounded evidence for the exact live Gemma sampler pre-optimizer gate."""

from copy import deepcopy
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
import json
import os
from pathlib import Path
import re
import stat
import sys


MISSING=object()
ORIGINAL_CALLBACK_SHA="5296dff4a099240438b094a6841a50d8950e18ef7f7dcd172c4e3cf7a6a4d452"
ORIGINAL_INPUTS_SHA="229e3e70edab723d7e5d2828a7176bc272e7badf15e9a9c1156f6595b65bc144"
ORIGINAL_3450_ARCHIVE_SHA="ac6714e6ce26c68e311867d11737c377f7a7f07ae046e98bf5a7f2df1aa8ee22"


def safe_name(value):
    return value if type(value) is str and re.fullmatch(r"[A-Za-z0-9_.+<>-]{1,160}",value) else None


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def implementation(module):
    value=sys.modules.get(module)
    path=getattr(value,"__file__",None)
    if not path or not module.startswith(("swift.","datasets.","accelerate.","torch.","transformers.")):
        return {"status":"UNAVAILABLE"}
    file=Path(path)
    try:
        before=file.stat()
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=4<<20:
            return {"status":"BOUNDED_FILE_UNAVAILABLE"}
        identify=lambda a:(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)
        with file.open("rb") as source:
            opened=os.fstat(source.fileno())
            raw=source.read((4<<20)+1)
            after=os.fstat(source.fileno())
        if identify(before)==identify(opened)==identify(after)==identify(file.stat()) and len(raw)==before.st_size:
            return {"status":"READ_ACTUAL_LOADED_SOURCE","SHA256":sha256(raw).hexdigest(),"bytes":len(raw)}
    except (OSError,ValueError):
        return {"status":"READ_UNAVAILABLE"}
    return {"status":"CHANGED_OR_TRUNCATED"}


class Observation:
    def __init__(self,rank):
        self.doc={"schema":"GEMMA_R43_LIVE_TRAIN_BEGIN_SAMPLER_PROJECTION_V1","rank":rank,
                  "actual_production_callback":True,"optimizer_updates":0,"live_loader_iterated":False,
                  "live_sampler_iterated":False,"RNG_unchanged":None,"clone_iterated":False,
                  "exact_original_callback_SHA256":ORIGINAL_CALLBACK_SHA,
                  "exact_original_inputs_SHA256":ORIGINAL_INPUTS_SHA,
                  "original_3450_archive_SHA256":ORIGINAL_3450_ARCHIVE_SHA,
                  "objects":{},"fields":{},"implementations":{},"checks":{},"first_failed_check":None}

    def versions(self):
        result={}
        for package in ("ms-swift","accelerate","torch","transformers","datasets"):
            try:
                result[package]=safe_name(version(package))
            except (PackageNotFoundError,OSError,ValueError):
                result[package]=None
        self.doc["installed_versions"]=result

    def object(self,name,value):
        if value is MISSING or value is None:
            self.doc["objects"][name]={"status":"MISSING" if value is MISSING else "NONE"}
            return
        cls=type(value)
        what=self.describe(cls)
        what["mro"]=[self.describe(member) for member in cls.__mro__[:8]]
        what["method_owner"]={}
        for method in ("__iter__","__len__"):
            declaring=next((parent for parent in cls.__mro__ if method in parent.__dict__),None)
            if declaring is not None:
                what["method_owner"][method]=self.describe(declaring)
        self.doc["objects"][name]=what

    def describe(self,cls):
        module=safe_name(cls.__module__)
        if module not in self.doc["implementations"] and module is not None:
            self.doc["implementations"][module]=implementation(module)
        return {"module":module,"class":safe_name(cls.__qualname__)}

    def get(self,value,name,key):
        try:
            found=getattr(value,name,MISSING)
        except Exception as error:
            self.doc["fields"][key]={"status":"READ_ERROR","class":safe_name(type(error).__name__)}
            return MISSING
        self.doc["fields"][key]={"status":"MISSING" if found is MISSING else "OBSERVED"}
        if found is None or type(found) is bool or type(found) is int and -1<=found<=1000000:
            self.doc["fields"][key]["value"]=found
        return found

    def test(self,key,ok):
        self.doc["checks"][key]=None if ok is None else bool(ok)
        if ok is not True and self.doc["first_failed_check"] is None:
            self.doc["first_failed_check"]=key

    def require(self):
        gate=self.doc["first_failed_check"]
        if gate is not None:
            raise ValueError("R43_STRICT_LIVE_SAMPLER_"+gate.upper())


def verify_live(loader,rank,ancestry,check,obs,types):
    expected_shard,expected_batch,expected_sampler=types
    obs.versions()
    for name,cls in (("expected_shard",expected_shard),("expected_batch",expected_batch),
                     ("expected_sampler",expected_sampler)):
        obs.doc["objects"][name]=obs.describe(cls)
    obs.object("loader",loader)
    base=obs.get(loader,"base_dataloader","loader.base_dataloader")
    obs.object("loader.base_dataloader",base)
    if base is not MISSING and base is not None:
        obs.object("loader.base_dataloader.batch_sampler",obs.get(base,"batch_sampler","base.batch_sampler"))
    try:
        length=len(loader)
    except Exception as error:
        length=None
        obs.doc["fields"]["loader.length_error"]=safe_name(type(error).__name__)
    obs.doc["fields"]["loader.length"]=length
    obs.test("loader_length_100",length==100)

    current=obs.get(loader,"dataset","loader.dataset")
    seen=set()
    for index in range(9):
        obs.object("dataset."+str(index),current)
        if current is MISSING or type(current).__module__.split(".",1)[0]!="swift":
            break
        other=obs.get(current,"dataset","dataset."+str(index)+".dataset")
        if other is MISSING:
            break
        if id(other)==id(current) or id(other) in seen or index==8:
            obs.test("dataset_wrapper_noncyclic",False)
            break
        seen.add(id(current))
        current=other
    else:
        obs.test("dataset_wrapper_noncyclic",False)
    if "dataset_wrapper_noncyclic" not in obs.doc["checks"]:
        obs.test("dataset_wrapper_noncyclic",True)

    dataset_type=type(current).__module__.startswith("datasets.")
    obs.test("real_arrow_dataset",dataset_type)
    if dataset_type:
        obs.test("dataset_length_800",len(current)==800)
        indices=obs.get(current,"_indices","dataset._indices")
        obs.test("arrow_indices_present",indices is not MISSING and indices is not None)
        table=obs.get(indices,"table","dataset.indices.table")
        columns=obs.get(table,"column_names","dataset.indices.column_names")
        obs.test("only_arrow_indices_column",columns==["indices"])
        if columns==["indices"]:
            order=table.column("indices").to_pylist()
            exact=all(type(x) is int for x in order)
            obs.doc["physical_lineage"]={"rows":len(order),"integer_type_exact":exact,
                                        "order_SHA256":digest(order) if exact else None}
            obs.test("arrow_order_matches_800",exact and len(order)==800 and
                     [x+1 for x in order]==ancestry["raw_ordinals_by_dataset_index"])
    shard=obs.get(loader,"batch_sampler","loader.batch_sampler")
    obs.object("batch_shard",shard)
    obs.test("exact_batch_sampler_shard_class",type(shard) is expected_shard)
    for key,want in (("process_index",rank),("num_processes",8),("split_batches",False),("even_batches",True)):
        actual=obs.get(shard,key,"batch_shard."+key)
        ok=None if actual is MISSING else (actual is want if type(want) is bool else actual==want)
        obs.test("shard_"+key,ok)
    batch=obs.get(shard,"batch_sampler","batch_shard.batch_sampler")
    obs.object("batch_sampler",batch)
    obs.test("exact_torch_batch_sampler_class",type(batch) is expected_batch)
    sampler=obs.get(batch,"sampler","batch_sampler.sampler")
    obs.object("sampler",sampler)
    obs.test("exact_seedable_sampler_class",type(sampler) is expected_sampler)
    for key,want in (("initial_seed",42),("epoch",0),("replacement",False),("num_samples",800)):
        actual=obs.get(sampler,key,"sampler."+key)
        ok=None if actual is MISSING else (actual is want if type(want) is bool else actual==want)
        obs.test("sampler_"+key,ok)
    drop=obs.get(batch,"drop_last","batch_sampler.drop_last")
    obs.test("no_drop_last",None if drop is MISSING else not drop)
    obs.test("batch_size_1",obs.get(batch,"batch_size","batch_sampler.batch_size")==1)
    obs.require()
    check()
    clone=deepcopy(shard,{id(sampler.data_source):sampler.data_source})
    obs.test("clone_distinct",clone is not shard and clone.batch_sampler.sampler is not sampler)
    obs.require()
    obs.doc["clone_iterated"]=True
    slots=list(islice(iter(clone),8))
    obs.test("clone_first_eight_integer_slots",len(slots)==8 and
             all(len(slot)==1 and type(slot[0]) is int and 0<=slot[0]<800 for slot in slots))
    obs.require()
    actual=[order[slot[0]]+1 for slot in slots]
    frozen=ancestry["rank_first64"][str(rank)]
    obs.doc["first_slots"]={"count":len(actual),"observed_SHA256":digest(actual),"expected_SHA256":digest(frozen)}
    obs.test("clone_slots_match_frozen_rank8_of64",actual==frozen)
    obs.require()
    return actual
