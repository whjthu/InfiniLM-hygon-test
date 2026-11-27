"""Observe the untouched real Swift sampler, native cursor skip, and all consumed positions."""
import functools
import inspect
import json
from pathlib import Path

from series_contract import EPOCHS, need


class SeriesCursor:
    def __init__(self,rank,plan,order):
        self.rank,self.plan,self.order=rank,plan,order
        self.emitted={}
        self.epochs=[]
        self.consumed=[]
        self.skip_consumed=False
        self.by_position={(row["epoch_index"],row["rank_offset0"]):row for row in plan}
        need(len(self.by_position)==len(plan)==88,"EXACT_RANK_88_PLAN")

    def witness(self,batches,epoch):
        self.epochs.append(epoch)
        need(self.epochs==list(EPOCHS[:len(self.epochs)]),"EXACT_ORDERED_NATIVE_EPOCH_ITERATORS")
        for offset,batch in enumerate(batches):
            need(offset<8 and type(batch) is list and len(batch)==1 and type(batch[0]) is int and
                 0<=batch[0]<64,"EXACT_LIVE_SINGLETON_BATCH")
            logical=batch[0]
            physical=self.order[logical]
            position=(epoch,offset)
            expected=self.by_position.get(position)
            if expected is not None:
                need((logical,physical)==(expected["logical"],expected["physical"]),
                     "LIVE_SAMPLER_MATCHES_CPU_CONTINUATION")
            else:
                need((epoch==1 and offset<4) or (epoch==12 and offset>=4),"EXACT_BOUNDARY_NONFORWARD_SLOT")
            self.emitted[position]=(logical,physical)
            yield batch

    def attach(self,loader,pin,dist):
        import live_guard
        from accelerate.data_loader import SkipBatchSampler,skip_first_batches
        from datasets import Dataset
        from swift.dataloader.shard import BatchSamplerShard,DataLoaderShard
        from torch.utils.data import DataLoader
        import transformers.trainer as trainer_module
        from scope import HERE

        pin(live_guard.SHARD_PATH,expected=live_guard.SHARD_PIN,extent=live_guard.SHARD_SIZE)
        pin(Path(inspect.getsourcefile(DataLoader)).resolve(strict=True),
            expected=live_guard.TORCH_DATALOADER_PIN,cap=262144)
        expected=dict(rank=self.rank,world_size=8,tp_size=1,total_samples=8,batch_size=1,base_seed=42,curr_seed=42,
                      shuffle=True,drop_last=False,group_by_length=False)
        sampler=loader.batch_sampler
        need(dist.is_initialized() and (dist.get_rank(),dist.get_world_size())==(self.rank,8)
             and type(loader) is DataLoaderShard and loader.in_order is True and loader.num_workers==4
             and type(sampler) is BatchSamplerShard and len(loader)==len(sampler)==8
             and all(type(getattr(sampler,key)) is type(value) and getattr(sampler,key)==value
                     for key,value in expected.items())
             and live_guard.live_arrow(loader.dataset,Dataset)[0]==self.order,
             "EXACT_LIVE_FOUR_WORKER_ARROW_LOADER")
        need(trainer_module.skip_first_batches is skip_first_batches and
             inspect.getsourcefile(skip_first_batches)==inspect.getsourcefile(SkipBatchSampler),
             "EXACT_ACTIVE_TRAINER_ACCELERATE_SKIP")
        proof=json.loads(pin(HERE/"SKIP_SOURCE.json",cap=16384))
        need(proof["qualified"] is True,"EXACT_DEPLOYED_SKIP_SOURCE_PROOF")
        for name,actual in (("trainer",trainer_module.__file__),
                            ("accelerate",inspect.getsourcefile(skip_first_batches))):
            expected=proof["source_files"][name]
            resolved=Path(actual).resolve(strict=True)
            need(resolved.as_posix()==expected["path"],"EXACT_ACTIVE_SKIP_MODULE_PATH")
            pin(resolved,expected=expected["sha256"],extent=expected["bytes"],cap=512*1024)
        original=BatchSamplerShard.__iter__
        need(inspect.getsourcefile(original)==str(live_guard.SHARD_PATH),"EXACT_NATIVE_SWIFT_SAMPLER_ITER")

        @functools.wraps(original)
        def watched(instance):
            if instance is not sampler:
                return original(instance)
            epoch=instance.curr_seed-instance.base_seed
            need(instance.base_seed==42 and epoch in EPOCHS and self.skip_consumed,
                 "ACTUAL_NATIVE_SEED_AND_SKIP_BEFORE_DRAW")
            return self.witness(original(instance),epoch)

        @functools.wraps(skip_first_batches)
        def skip(current,num_batches=0):
            need(current is loader and num_batches==4 and not self.skip_consumed and not self.epochs
                 and sampler.base_seed==42 and sampler.curr_seed in (42,43),
                 "EXACT_ONCE_NATIVE_EPOCH1_SKIP_FOUR")
            forwarded=skip_first_batches(current,num_batches)
            inner=forwarded.batch_sampler
            need(type(inner) is SkipBatchSampler and inner.batch_sampler is sampler
                 and inner.skip_batches==4 and forwarded.dataset is current.dataset
                 and forwarded.num_workers==4 and forwarded.in_order is True,
                 "EXACT_NATIVE_SKIP_WRAPPED_SAME_SAMPLER")
            self.skip_consumed=True
            return forwarded

        BatchSamplerShard.__iter__=watched
        trainer_module.skip_first_batches=skip
        self.hooks=(BatchSamplerShard,original,watched,trainer_module,skip_first_batches,skip)

    def prepare(self,offset,source):
        need(offset==len(self.consumed) and 0<=offset<len(self.plan) and self.skip_consumed,
             "EXACT_CONSUMPTION_CURSOR_AND_NATIVE_SKIP")
        wanted=self.plan[offset]
        need(self.emitted.get((wanted["epoch_index"],wanted["rank_offset0"]))==(wanted["logical"],wanted["physical"])
             and (source["ordinal"],source["row_SHA256"])==(wanted["source_ordinal"],wanted["row_SHA256"]),
             "LIVE_COLLATOR_SOURCE_MATCHES_CPU_SLOT")
        return dict(source,logical=wanted["logical"],physical=wanted["physical"])

    def forward(self,offset,source):
        match=self.prepare(offset,source)
        need((source["logical"],source["physical"])==(match["logical"],match["physical"]),
             "ACTUAL_FORWARD_MATCHES_PREPARED_SLOT")
        witness=self.plan[offset]
        self.consumed.append(witness)
        return witness

    def detach(self):
        cls,original,watched,trainer,skip_original,skip_watched=self.hooks
        need(cls.__iter__ is watched and trainer.skip_first_batches is skip_watched,"EXACT_OWNED_LOADER_HOOKS")
        cls.__iter__=original
        trainer.skip_first_batches=skip_original
