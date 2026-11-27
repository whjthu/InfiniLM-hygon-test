"""Observe Swift's real skip patch with metadata-only Arrow and no model."""
import functools
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import lineage
import native_route


MIXIN=Path("/swift-runtime/site/swift/trainers/mixin.py")
MIXIN_PIN=(64818,"6b6de2e7503ce414cb837961e1355be41a56787ecca502e8570bfcfa36c15ab9")
SHARD=Path("/swift-runtime/site/swift/dataloader/shard.py")
SHARD_PIN=(3553,"ad05502b587d37eb4de7d7cd61f5f652eb7751403320f2b18524d21b7fbd340c")
TRAINER=Path("/usr/local/lib/python3.10/dist-packages/transformers/trainer.py")
TRAINER_PIN=(215477,"17e212935057f58b52efe79c3f952bf044bbada8afd4f13c8e905142b1d2ef0a")
ACCELERATE=Path("/usr/local/lib/python3.10/dist-packages/accelerate/data_loader.py")
ACCELERATE_PIN=(66830,"e08c5f5cd3f7a8b8aaece03ad51f1c37994e390faf47d368281ad49c31bc84d1")
CPU_ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/"
              "internvl-gqa56-native-arrow-epochs1to12-1ccc01ddf76049dca2e32d9f013be644/output")
CPU_REQUEST="1ccc01ddf76049dca2e32d9f013be644"
CPU_RESULT="670b572a335f4d6b5a084b662058a60e16eca3a2f2f92b4f6a08a49698d75086"
CPU_PLAN="04f7385ba2b79dd42dddedf03f2ec8f3f1848ae4ee13727f8eea0a4a01a4e005"
CP3_FAILED_REQUEST="f872143f007845ba88b5f4032e8d912d"
SKIP=4


class FetchObserved(lineage.ObservedOccurrences):
    def __init__(self,*args):
        super().__init__(*args)
        self.fetches=[]

    def __getitem__(self,logical):
        value=super().__getitem__(logical)
        self.fetches.append(logical)
        return value


def prior_plan(pin):
    raw=pin(CPU_ROOT/"result.json",9579,CPU_RESULT)
    receipt=json.loads(raw)
    lineage.need((receipt["request"],receipt["actual_step"],receipt["verified_continuation_slots"],
                  receipt["actual_native_metadata_shards_verified"],receipt["verified_epochs"])==
                 (CPU_REQUEST,"1103606.4738",704,8,12) and receipt["epoch1_byte_matches_4726"] is True
                 and receipt["continuation_plan_SHA256"]==CPU_PLAN,"EXACT_ORIGINAL_4738_RESULT")
    return pin(CPU_ROOT/"steps4-25-plan.private.json",141110,CPU_PLAN)


def captured_skip(patched,plain,holder):
    closure=inspect.getclosurevars(patched).nonlocals
    lineage.need(patched is not plain and inspect.getsourcefile(patched)==str(MIXIN) and
                 patched.__code__.co_firstlineno==1218 and closure.get("self") is holder and
                 closure.get("origin_skip_first_batches") is plain,"EXACT_LIVE_SWIFT_NESTED_SKIP_CALLABLE")


def sampler_fields(shard,rank,seed):
    desired=dict(rank=rank,world_size=8,tp_size=1,total_samples=8,batch_size=1,base_seed=42,curr_seed=seed,
                 shuffle=True,drop_last=False,group_by_length=False)
    lineage.need(all(type(getattr(shard,key)) is type(value) and getattr(shard,key)==value
                     for key,value in desired.items()),"EXACT_NATIVE_SAMPLER_FIELDS_AND_SEED")


def verify_resume_slots(raw,retained,fetches,reference,order,typed):
    """Make the distinction between consumed sampler positions and dataset retrieval explicit."""
    lineage.need(raw==reference and len(raw)==8,"EXACT_NATIVE_EIGHT_PRE_SKIP_POSITIONS")
    expected=[dict(logical=logical,physical=order[logical],typed_row_SHA256=typed[order[logical]])
              for logical in raw[SKIP:]]
    lineage.need(retained==expected and fetches==raw[SKIP:],"NATIVE_SKIP_FOUR_BEFORE_REAL_DATASET_FETCH")
    return raw[:SKIP],raw[SKIP:]


def observe(torch,mixin,impl,dataset_type,parsed,load,rank,expected,reference,dataset_source,clock,pin):
    from accelerate.data_loader import SkipBatchSampler,skip_first_batches
    import transformers.trainer as hf

    for src,wanted in ((MIXIN,MIXIN_PIN),(SHARD,SHARD_PIN),(TRAINER,TRAINER_PIN),(ACCELERATE,ACCELERATE_PIN)):
        pin(src,*wanted)
    lineage.need(Path(mixin.__file__)==MIXIN and Path(impl.__file__)==SHARD and Path(hf.__file__)==TRAINER
                 and Path(inspect.getsourcefile(SkipBatchSampler))==ACCELERATE and
                 hf.skip_first_batches is skip_first_batches,"EXACT_IMPORTED_SKIP_PROVIDER_OBJECTS")
    method=mixin.DataLoaderMixin.get_train_dataloader
    patch=mixin.SwiftMixin._patch_skip_first_batches
    lineage.need(inspect.getsourcefile(inspect.unwrap(patch))==str(MIXIN) and
                 native_route.method_globals_are_actual(method,impl),"EXACT_NATIVE_MIXIN_AND_SHARD")
    dist=torch.distributed
    lineage.need(dist.is_initialized() and (dist.get_rank(),dist.get_world_size())==(rank,8)
                 and not torch.cuda.is_initialized(),"EXACT_EIGHT_RANKS_CPU_ONLY")
    unshuffled,shuffled=load(False),load(True)
    order,messages,typed=lineage.arrow_projection(unshuffled,shuffled,dataset_type,expected,dataset_source)
    lineage.need(lineage.digest(lineage.encode(order))==native_route.V4_ARROW_SHA,"EXACT_ORIGINAL_ARROW_ORDER")
    metadata=FetchObserved(messages,shuffled._data.table,order)
    traversal=native_route.MetadataTraversalArguments(parsed,dist,rank)
    holder=SimpleNamespace(template=SimpleNamespace(sequence_parallel_size=1),train_dataset=metadata,
                           _train_batch_size=1,data_collator=lineage.collate_observed,
                           accelerator=SimpleNamespace(device=torch.device("cpu")),args=traversal)
    constructions=[]

    def get_train_dataloader(skip_batches=0):
        clock()
        loader=method(holder,skip_batches=skip_batches)
        constructions.append((skip_batches,loader))
        return loader

    holder.get_train_dataloader=get_train_dataloader
    reference_epochs=native_route.references(torch,reference)
    epochs=[]
    with patch(holder):
        active=hf.skip_first_batches
        captured_skip(active,skip_first_batches,holder)
        base=holder.get_train_dataloader()
        resumed=active(base,SKIP)
        lineage.need(constructions==[(0,base),(SKIP,resumed)] and base is not resumed and
                     all(type(loader) is impl.DataLoaderShard and loader.dataset is metadata
                         and loader.num_workers==0 for loader in (base,resumed)),
                     "EXACT_TWO_REAL_SWIFT_METADATA_LOADERS")
        original=base.batch_sampler
        wrapper=resumed.batch_sampler
        lineage.need(type(wrapper) is SkipBatchSampler and wrapper.skip_batches==SKIP and
                     type(original) is type(wrapper.batch_sampler) is impl.BatchSamplerShard and
                     original is not wrapper.batch_sampler and len(base)==8 and len(resumed)==4,
                     "EXACT_FRESH_NESTED_RESUMED_SAMPLER")
        nested=wrapper.batch_sampler
        sampler_fields(original,rank,42)
        sampler_fields(nested,rank,42)
        resumed.set_epoch(1)
        sampler_fields(nested,rank,43)
        sampler_fields(original,rank,42)
        raw=[]
        saved=impl.BatchSamplerShard.__iter__

        @functools.wraps(saved)
        def witness(instance):
            for batch in saved(instance):
                if instance is nested:
                    lineage.need(type(batch) is list and len(batch)==1 and type(batch[0]) is int and
                                 len(raw)<8,"EXACT_RESUMED_RAW_SINGLETON_SAMPLER")
                    raw.append(batch[0])
                yield batch

        impl.BatchSamplerShard.__iter__=witness
        try:
            retained=native_route.read_epoch(resumed,clock)
        finally:
            lineage.need(impl.BatchSamplerShard.__iter__ is witness,"OWNED_SAMPLER_WITNESS")
            impl.BatchSamplerShard.__iter__=saved
        skipped,kept=verify_resume_slots(raw,retained,metadata.fetches,reference_epochs[1][rank],order,typed)
        # Skipped positions are sampler observations and independently verified Arrow lineage only.
        # Their values are not presented to the dataset, collator or model in this resume traversal.
        epoch1=[dict(rank=rank,update=offset//4+1,accumulation=offset%4+1,logical=logical,
                     **expected[order[logical]][1]) for offset,logical in enumerate(raw)]
        epochs.append(dict(epoch_index=1,data_seed=42,sampler_seed=43,rows=epoch1))
        for epoch in range(2,13):
            clock()
            metadata.fetches.clear()
            seeds=native_route.apply_epoch(base,epoch)
            sampler_fields(original,rank,42+epoch)
            rows=native_route.read_epoch(base,clock)
            lineage.need(base.batch_sampler is original and resumed.batch_sampler is wrapper and
                         wrapper.batch_sampler is nested and metadata.fetches==reference_epochs[epoch][rank],
                         "EXACT_ORIGINAL_BASE_LOADER_REUSED_AFTER_RESUME_EPOCH")
            epochs.append(dict(**seeds,rows=lineage.verify_rank(rank,rows,reference_epochs[epoch],order,expected,typed)))
        lineage.need(constructions==[(0,base),(SKIP,resumed)] and hf.skip_first_batches is active,
                     "EXACT_TWO_CONSTRUCTIONS_UNTIL_EPOCH12")
    lineage.need(hf.skip_first_batches is skip_first_batches and not torch.cuda.is_initialized(),
                 "ORIGINAL_PATCH_RESTORED_AND_CPU_ONLY")
    return epochs,order,dict(constructions=[0,SKIP],original_separate=True,new_nested_sampler=True,
                             skipped=skipped,retained=kept,skipped_dataset_fetches=0,
                             successful_native_patch_calls=1)
