"""Bind the archived GQA56 occurrence matrix and observe the live native sampler."""
from collections import Counter
from hashlib import sha256
import inspect
import json
from pathlib import Path


ARROW_ROOT="/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/internvl-gqa56-native-arrow-v4-0947b7dac1da45ac842fae8bb0adc25e/output"
ARROW_REQUEST="0947b7dac1da45ac842fae8bb0adc25e"
RESULT_PIN="0c58bb4680e1e551d29abeef07ce86fb9e8cf8ca8bb5fd9192fb7cc5f65182af"
MATRIX_PIN="a931dc4bc7152c5aa866bc257b2c0be19706ca7974d0f8e606a19e8666ff7e1d"
ARROW_PIN="5d7ec6e740dfca0cf3004fb0d979fd37ce0d8878ef9a6af1e46b24a89ee33f4e"
LOGICAL_PIN="639aeef07c7e916f1e3bc49a02ac9637547c2c654ae10e5c997e4c0bc81b43f8"
SHARD_PATH=Path("/swift-runtime/site/swift/dataloader/shard.py")
SHARD_PIN="ad05502b587d37eb4de7d7cd61f5f652eb7751403320f2b18524d21b7fbd340c"
SHARD_SIZE=3553
TORCH_DATALOADER_PIN="cb8d82b0de999f21349b7dd261d30919fcf692f0b1d7504ca2594e88e8fd0e96"


def need(ok,code):
    if not ok:
        raise ValueError(code)


def digest(raw):
    return sha256(raw).hexdigest()


def encode(data):
    return (json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def bind_matrix(exposures,original,proposed,source_rows,matrix):
    """Independently join full raw exposures and original row identities by physical position."""
    stored,full=exposures.splitlines(),original.splitlines()
    need(len(stored)==len(full)==len(proposed)==64 and len(source_rows)==56 and len(matrix)==8 and
         all(type(row) is list and len(row)==8 for row in matrix),"EXACT_ORIGINAL_AND_MATRIX_DENOMINATORS")
    by_physical={}
    for index,(raw,slot) in enumerate(zip(stored,proposed,strict=True)):
        ordinal=slot["source_ordinal"]
        need(type(ordinal) is int and ordinal in source_rows and 1<=ordinal<=64 and
             raw==full[ordinal-1] and digest(raw)==slot["row_SHA256"]==source_rows[ordinal],
             "EXACT_PHYSICAL_EXPOSURE_AND_4370_ORIGINAL")
        by_physical[index]=dict(ordinal=ordinal,row_SHA256=digest(raw))
    ordered=[[] for _ in range(8)]
    raw_order=[None]*64
    seen=set()
    counts=Counter()
    evidence=[]
    for rank,row in enumerate(matrix):
        for offset,slot in enumerate(row):
            logical,physical=slot["logical"],slot["physical"]
            need(type(logical) is type(physical) is int and 0<=logical<64 and 0<=physical<64 and
                 raw_order[logical] is None and physical not in seen,"EXACT_NO_REUSED_PHYSICAL_OR_LOGICAL_OCCURRENCE")
            base=by_physical[physical]
            need(slot==dict(rank=rank,update=offset//4+1,accumulation=offset%4+1,logical=logical,
                            physical=physical,source_ordinal=base["ordinal"],row_SHA256=base["row_SHA256"]),
                 "EXACT_4699_RANK_ACCUMULATION_PHYSICAL_SOURCE")
            raw_order[logical]=physical
            seen.add(physical)
            counts[base["ordinal"]]+=1
            ordered[rank].append(dict(base,logical=logical,physical=physical))
            evidence.append(slot)
    need(set(counts)==set(source_rows) and Counter(counts.values())=={1:48,2:8} and
         [len({slot["source_ordinal"] for slot in evidence if slot["update"]==step}) for step in (1,2)]==[31,31],
         "EXACT_4699_56_PLUS_8_AND_31_31_WINDOWS")
    return ordered,raw_order


def authority(root,exposures,original,proposed,subset,pin):
    result=json.loads(pin(root/"result.json",expected=RESULT_PIN,extent=6243,cap=8192))
    raw=pin(root/"physical-rank-matrix.private.json",expected=MATRIX_PIN,cap=32768)
    need(result["request"]==ARROW_REQUEST and result["actual_step"]=="1103606.4699" and
         result["actual_arrow_mapping_verified"] is True and result["metadata_collator"] is True and
         result["actual_native_metadata_shards_verified"]==8 and result["actual_optimizer_updates"]==0 and
         result["production_workers"]==4 and result["observed_loader_workers"]==0 and
         (result["physical_occurrences"],result["unique_source_rows"],result["repeat_occurrences"],
          result["internal_source_rows"],result["per_update_unique_sources"])==(64,56,8,0,[31,31]) and
         result["actual_rank_occurrence_SHA256"]==MATRIX_PIN and result["actual_arrow_logical_to_physical_SHA256"]==ARROW_PIN,
         "EXACT_ORIGINAL_4699_ARROW_AUTHORITY")
    matrix=json.loads(raw)
    need(encode(matrix)==raw,"EXACT_CANONICAL_4699_RANK_BYTES")
    source_rows={row["ordinal"]:row["row_SHA256"] for row in subset.values()}
    by_rank,order=bind_matrix(exposures,original,proposed,source_rows,matrix)
    need(digest(encode(order))==ARROW_PIN and
         digest(encode([[cell["logical"] for cell in row] for row in by_rank]))==LOGICAL_PIN,
         "EXACT_4699_ARROW_AND_4643_LOGICAL_ORDER")
    return by_rank,order


def live_arrow(dataset,dataset_class):
    seen=set()
    wrappers=[]
    while type(dataset) is not dataset_class:
        cls=type(dataset)
        source=inspect.getsourcefile(cls)
        need(id(dataset) not in seen and len(seen)<4 and cls.__name__=="LazyLLMDataset" and
             cls.__module__.startswith("swift.dataset.") and source is not None and
             Path(source).is_relative_to("/swift-runtime/site/swift/dataset"),
             "EXACT_NATIVE_ARROW_DATASET_WRAPPER")
        wrappers.append(cls.__module__+"."+cls.__qualname__)
        seen.add(id(dataset))
        dataset=dataset.dataset
    need(len(dataset)==64 and dataset._data.table.num_rows==64 and dataset._indices is not None and
         dataset._indices.table.column_names==["indices"],"EXACT_LIVE_FULL_64_ARROW_TABLE")
    order=dataset._indices.table.column("indices").to_pylist()
    need(type(order) is list and all(type(value) is int for value in order) and sorted(order)==list(range(64)) and
         digest(encode(order))==ARROW_PIN,"EXACT_LIVE_4699_ARROW_PERMUTATION")
    return order,wrappers


class LiveCursor:
    """Witness the live pinned sampler iterator used by the unchanged production DataLoader."""
    def __init__(self,rank,plan,order):
        need(type(rank) is int and 0<=rank<8 and len(plan)==8,"EXACT_RANK_LIVE_CURSOR_SCOPE")
        self.rank=rank
        self.plan=plan
        self.order=order
        self.emitted=[]
        self.consumed=[]
        self.original=None

    def witness(self,batches):
        for batch in batches:
            index=len(self.emitted)
            need(index<8 and type(batch) is list and len(batch)==1 and type(batch[0]) is int and
                 0<=batch[0]<64,"EXACT_NATIVE_SAMPLER_SINGLETON_INDEX")
            logical,=batch
            physical=self.order[logical]
            expected=self.plan[index]
            need((logical,physical)==(expected["logical"],expected["physical"]),
                 "EXACT_NATIVE_SAMPLER_LOGICAL_AND_PHYSICAL_SLOT")
            self.emitted.append((logical,physical))
            yield batch

    def attach(self,loader,pin,dist):
        from datasets import Dataset
        from swift.dataloader.shard import BatchSamplerShard, DataLoaderShard
        from torch.utils.data import DataLoader

        pin(SHARD_PATH,expected=SHARD_PIN,extent=SHARD_SIZE)
        pin(Path(inspect.getsourcefile(DataLoader)).resolve(strict=True),expected=TORCH_DATALOADER_PIN,cap=262144)
        need(dist.is_initialized() and (dist.get_rank(),dist.get_world_size())==(self.rank,8) and
             type(loader) is DataLoaderShard and loader.in_order is True and loader.num_workers==4 and
             Path(inspect.getsourcefile(DataLoaderShard))==Path(inspect.getsourcefile(BatchSamplerShard))==SHARD_PATH,
             "EXACT_LIVE_SWIFT_DATALOADER_AND_ORDERED_FOUR_WORKERS")
        sampler=loader.batch_sampler
        need(type(sampler) is BatchSamplerShard and len(loader)==len(sampler)==8 and
             (sampler.rank,sampler.world_size,sampler.tp_size,sampler.total_samples,sampler.batch_size,
              sampler.base_seed,sampler.curr_seed)==(self.rank,8,1,8,1,42,42) and
             sampler.shuffle is True and sampler.drop_last is sampler.group_by_length is False,
             "EXACT_LIVE_PRODUCTION_SAMPLER_FIELDS")
        order,wrappers=live_arrow(loader.dataset,Dataset)
        need(self.order==order and self.original is None,"EXACT_LIVE_ARROW_AND_NEW_CURSOR")
        cls=type(sampler)
        original=cls.__iter__
        need(inspect.getsourcefile(original)==str(SHARD_PATH),"EXACT_OWN_NATIVE_SAMPLER_ITERATOR")
        def wrapped(instance):
            batches=original(instance)
            return self.witness(batches) if instance is sampler else batches
        self.original=(cls,original,wrapped)
        cls.__iter__=wrapped
        return dict(actual_loader=True,actual_num_workers=4,in_order=True,arrow_wrappers=wrappers,
                    live_sampler_iterator_witness=True,proof_matrix_SHA256=MATRIX_PIN,
                    torch_order_source_SHA256=TORCH_DATALOADER_PIN)

    def prepare(self,offset,source):
        need(offset==len(self.consumed) and offset<len(self.emitted),"EXACT_PREPARE_CONSUMES_EMITTED_SLOT")
        logical,physical=self.emitted[offset]
        wanted=self.plan[offset]
        need((logical,physical)==(wanted["logical"],wanted["physical"]) and
             all(source.get(key)==wanted[key] for key in ("ordinal","row_SHA256")),
             "EXACT_PREPARE_FINGERPRINT_MATCHES_LIVE_OCCURRENCE")
        return dict(source,logical=logical,physical=physical)

    def forward(self,offset,source):
        landed=self.prepare(offset,source)
        need(all(source.get(key)==landed[key] for key in ("logical","physical")),
             "EXACT_FORWARD_MATCHES_PREPARED_LIVE_OCCURRENCE")
        row=self.plan[offset]
        self.consumed.append(row)
        return row

    def confirmed(self,count):
        need(count in (4,8) and self.consumed==self.plan[:count],"EXACT_LIVE_ACCUMULATED_OCCURRENCES")

    def detach(self):
        if self.original is not None:
            cls,original,wrapper=self.original
            need(cls.__iter__ is wrapper,"EXACT_NATIVE_SAMPLER_HOOK_STILL_OWNED")
            cls.__iter__=original
            self.original=None
