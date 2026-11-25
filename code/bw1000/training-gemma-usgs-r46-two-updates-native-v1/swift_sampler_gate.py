"""Prove the live Swift shard against the exact CPU and native input receipts."""

from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import os
import stat


SWIFT_SHA="ad05502b587d37eb4de7d7cd61f5f652eb7751403320f2b18524d21b7fbd340c"
TORCH_SHA="9edf6b378be77655e0641130e18edfc32d4415d9355cf027bef3eba54aeeecc9"
ARROW_SHA="6cfc9a36eee275885a2f01a92086db9d66d6cf879f71293cbc5dc6af96c34104"
CPU_DRAW="0571ccd9b83ba7239c831018f2b11e9d3a650dd8100b624367d0ff5f27a63e32"
CPU_FIRST64="2e6a490ac4e1ab91dfe16dc16c1e74217fd579182694741775fc894041b8f250"
CPU_READ="b6e32e773064800a697ec3dd6641916cac99e50f78634c0fcaf117cabf174d1e"
W,N,M,K=8,800,100,8
FIELD_EXPECTED={"tp_size":1,"total_samples":100,"batch_size":1,"shuffle":True,"drop_last":False,
                "base_seed":42,"curr_seed":42,"group_by_length":False,"world_size":8}


def need(gate,code):
    if not gate:
        raise ValueError(code)


def ident(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def pin_source(path,size,wanted):
    before=path.lstat()
    keys=lambda p:(p.st_dev,p.st_ino,p.st_size,p.st_mtime_ns,p.st_ctime_ns)
    need(path.resolve(strict=True)==path and stat.S_ISREG(before.st_mode) and before.st_size==size,
         "G45_SWIFT_RUNTIME_SOURCE_FILE")
    with path.open("rb") as source:
        opened=os.fstat(source.fileno())
        raw=source.read(size+1)
        after=os.fstat(source.fileno())
    need(keys(before)==keys(opened)==keys(after)==keys(path.lstat()) and len(raw)==size
         and sha256(raw).hexdigest()==wanted,"G45_SWIFT_RUNTIME_SOURCE_PIN")


def check_provider(torch):
    from swift.dataloader.shard import DataLoaderShard, BatchSamplerShard
    module=import_module("swift.dataloader.shard")
    path=Path(module.__file__)
    need(path==Path("/swift-runtime/site/swift/dataloader/shard.py"),"G45_EXACT_SWIFT_RUNTIME_MOUNT")
    pin_source(path,3553,SWIFT_SHA)
    need(str(torch.__version__)=="2.10.0" and torch.version.hip and torch.version.cuda is None,
         "G45_EXACT_HIP_TORCH_VERSION")
    pin_source(Path(torch.__file__).resolve(strict=True),107368,TORCH_SHA)
    return DataLoaderShard,BatchSamplerShard


def fields(shard,rank,record):
    expect={**FIELD_EXPECTED,"rank":rank}
    observed={}
    record["fields"]=observed
    for name,wanted in expect.items():
        try:
            found=getattr(shard,name)
        except (AttributeError,RuntimeError):
            observed[name]={"status":"UNAVAILABLE","passes":False}
            continue
        typed=type(found) is type(wanted)
        observed[name]={"status":"OBSERVED","value":found if typed else None,"passes":typed and found==wanted}
    need(all(value["passes"] for value in observed.values()),"G45_LIVE_SWIFT_FIELDS")
    return {key:v["value"] for key,v in observed.items()}


def full_arrow(loader,expected,record):
    current=loader.dataset
    seen=set()
    for _ in range(9):
        if type(current).__module__.split(".",1)[0]!="swift":
            break
        need(id(current) not in seen and hasattr(current,"dataset"),"G45_SWIFT_DATASET_CHAIN")
        seen.add(id(current))
        current=current.dataset
    need(type(current).__module__.startswith("datasets.") and len(current)==N and current._indices is not None,
         "G45_NATIVE_ARROW_800")
    table=current._indices.table
    need(table.column_names==["indices"],"G45_NATIVE_ARROW_INDEX_COLUMN")
    logical=table.column("indices").to_pylist()
    need(len(logical)==N and all(type(x) is int for x in logical) and set(logical)==set(range(N)),
         "G45_NATIVE_ARROW_COMPLETE_PERMUTATION")
    record.update(arrow_rows=N,arrow_SHA256=ident(logical))
    physical=[i+1 for i in logical]
    need(ident(logical)==ARROW_SHA and physical==expected["raw_ordinals_by_dataset_index"],
         "G45_ARROW_MATCHES_3465_AND_ORIGINAL_3406")
    return physical


def derive_expected(torch,arrow,ancestry):
    gen=torch.Generator(device="cpu")
    gen.manual_seed(42)
    draw=torch.randperm(N,generator=gen,device="cpu").tolist()
    need(len(draw)==N and set(draw)==set(range(N)) and ident(draw)==CPU_DRAW,"G45_ACTUAL_3481_CPU_DRAW_PARITY")
    ranks=[draw[rank::W] for rank in range(W)]
    all_slots=[i for r in ranks for i in r]
    need(all(len(r)==M and len(set(r))==M for r in ranks) and len(set(all_slots))==len(all_slots)==N,
         "G45_ALL_800_SLOTS_EXACT_UNION")
    scout={str(rank):[arrow[i] for i in logical[:K]] for rank,logical in enumerate(ranks)}
    need(scout==ancestry["rank_first64"] and ident(scout)==CPU_FIRST64,"G45_ACTUAL_3481_AND_3406_FIRST64_PARITY")
    return ranks,scout


def check_live(torch,loader,rank,ancestry,record,check_time):
    actual_loader,actual_shard=check_provider(torch)
    dist=torch.distributed
    need(dist.is_initialized() and dist.get_world_size()==W and dist.get_rank()==rank,
         "G45_ACTUAL_EIGHT_RANK_DISTRIBUTED_WORLD")
    shard=loader.batch_sampler
    record.update(loader_class=type(loader).__name__,shard_class=type(shard).__name__)
    need(type(loader) is actual_loader and type(shard) is actual_shard and len(loader)==len(shard)==M,
         "G45_EXACT_SWIFT_SHARD_RUNTIME_TYPE")
    live=fields(shard,rank,record)
    arrow=full_arrow(loader,ancestry,record)
    expected,first=derive_expected(torch,arrow,ancestry)
    check_time()
    lengths=shard.lengths
    need(lengths is None or (type(lengths) is list and len(lengths)<=N and
                           all(type(k) is int and k>=0 for k in lengths)),"G45_BOUNDED_REAL_LENGTHS_FIELD")
    # Original class. rank/world read the real torch.distributed group.
    separate=actual_shard(total_samples=N,batch_size=live["batch_size"],shuffle=live["shuffle"],
                         drop_last=live["drop_last"],data_seed=live["base_seed"],tp_size=live["tp_size"],
                         group_by_length=live["group_by_length"],lengths=lengths)
    need(separate is not shard and fields(separate,rank,{})==live,"G45_INDEPENDENT_REAL_SWIFT_SHARD")
    record["isolated_sampler_iterated"]=True
    derived=[]
    for batch in separate:
        check_time()
        need(len(derived)<M and type(batch) is list and len(batch)==1 and type(batch[0]) is int
             and 0<=batch[0]<N,"G45_INDEPENDENT_SHARD_BATCH_SHAPE")
        derived.append(batch[0])
    record.update(logical_slots=len(derived),logical_rank_SHA256=ident(derived))
    need(len(derived)==M and len(set(derived))==M and derived==expected[rank]
         and fields(shard,rank,{})==live,"G45_ISOLATED_100_MATCH_AND_LIVE_FIELDS_UNCHANGED")
    ordinals=[arrow[i] for i in derived[:K]]
    need(ordinals==first[str(rank)],"G45_EXACT_8_REAL_FORWARD_POSITIONS")
    record.update(rank_first8_SHA256=ident(ordinals),expected_all_rank_logical_SHA256=[ident(r) for r in expected],
                  expected_all_rank_first8_SHA256=[ident(first[str(r)]) for r in range(W)],
                  cpu_shard_reference_SHA256=CPU_READ)
    return ordinals
