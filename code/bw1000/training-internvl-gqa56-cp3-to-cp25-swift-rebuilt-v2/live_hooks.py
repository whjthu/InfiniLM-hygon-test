"""Attach the real Swift rebuilt sampler to the existing production occurrence cursor."""
import functools
import inspect
import json
from pathlib import Path

import swift_watch as watch


def attach(cursor,loader,pin,dist,owner):
    import live_guard
    from accelerate.data_loader import SkipBatchSampler,skip_first_batches
    from datasets import Dataset
    from swift.dataloader.shard import BatchSamplerShard,DataLoaderShard
    from swift.trainers.mixin import DataLoaderMixin
    from torch.utils.data import DataLoader
    import transformers.trainer as hf
    from scope import HERE

    need=watch.need
    proof=json.loads(pin(HERE/"SKIP_SOURCE.json",cap=16384))
    need(proof["qualified"] is True,"EXACT_DEPLOYED_SKIP_SOURCE_PROOF")
    for name,actual in (("trainer",hf.__file__),("accelerate",inspect.getsourcefile(skip_first_batches))):
        item=proof["source_files"][name]
        path=Path(actual).resolve(strict=True)
        need(path.as_posix()==item["path"],"EXACT_ACTIVE_SKIP_MODULE_PATH")
        pin(path,expected=item["sha256"],extent=item["bytes"],cap=512*1024)
    pin(watch.MIXIN,expected=watch.MIXIN_SHA,extent=watch.MIXIN_BYTES)
    pin(live_guard.SHARD_PATH,expected=live_guard.SHARD_PIN,extent=live_guard.SHARD_SIZE)
    pin(Path(inspect.getsourcefile(DataLoader)).resolve(strict=True),
        expected=live_guard.TORCH_DATALOADER_PIN,cap=262144)
    original=loader.batch_sampler
    need(cursor.hooks is None and getattr(owner,watch.SCOPE_ATTR,"missing") is None
         and dist.is_initialized() and (dist.get_rank(),dist.get_world_size())==(cursor.rank,8)
         and type(loader) is DataLoaderShard and type(original) is BatchSamplerShard
         and loader.in_order is True and loader.num_workers==4 and len(loader)==len(original)==8
         and inspect.ismethod(owner.get_train_dataloader)
         and owner.get_train_dataloader.__func__ is DataLoaderMixin.get_train_dataloader
         and live_guard.live_arrow(loader.dataset,Dataset)[0]==cursor.order,
         "EXACT_ACTIVE_OWNED_FOUR_WORKER_SWIFT_AND_ARROW")
    watch.sampler_fields(original,cursor.rank,42)
    active=watch.active_skip_callable(hf,skip_first_batches,owner)
    iterator=BatchSamplerShard.__iter__
    need(inspect.getsourcefile(iterator)==str(live_guard.SHARD_PATH),"EXACT_NATIVE_SWIFT_SAMPLER_ITER")

    @functools.wraps(iterator)
    def observed(instance):
        if instance is not original and instance is not cursor.rebuilt_sampler:
            return iterator(instance)
        epoch=instance.curr_seed-instance.base_seed
        need(cursor.skip_consumed and ((instance is cursor.rebuilt_sampler and epoch==1) or
             (instance is original and 2<=epoch<=12)),"EXACT_REAL_SAMPLER_EPOCH_OWNERSHIP")
        watch.sampler_fields(instance,cursor.rank,42+epoch)
        return cursor.witness(iterator(instance),epoch)

    @functools.wraps(active)
    def audited_skip(current,num_batches=0):
        need(current is loader and type(num_batches) is int and num_batches==4 and
             not cursor.skip_consumed and not cursor.epochs,"EXACT_ONCE_NATIVE_EPOCH1_SKIP_FOUR")
        # Native Swift constructs the new loader. The pinned HF caller sets epoch 1 next.
        forwarded=active(current,num_batches)
        cursor.rebuilt_sampler=watch.require_rebuilt(
            loader,forwarded,original,SkipBatchSampler,BatchSamplerShard,cursor.rank)
        cursor.skip_consumed=True
        return forwarded

    cursor.hooks=(BatchSamplerShard,iterator,observed,hf,active,audited_skip)
    setattr(owner,watch.SCOPE_ATTR,cursor)
    BatchSamplerShard.__iter__=observed
    hf.skip_first_batches=audited_skip


def detach(cursor,strict=True):
    if cursor.hooks is None:
        return
    klass,original,our_iter,hf,swift_active,our_skip=cursor.hooks
    iterator_owned=klass.__iter__ is our_iter
    skip_owned=hf.skip_first_batches is our_skip
    if iterator_owned:
        klass.__iter__=original
    if skip_owned:
        hf.skip_first_batches=swift_active
    cursor.hooks=None
    if strict:
        watch.need(iterator_owned and skip_owned,"EXACT_OWNED_LOADER_HOOKS")
